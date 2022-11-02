#!/usr/bin/env python3

import os, sys
import argparse
import numpy
import tifffile
import pyexiv2
import email.base64mime
import struct
import time
from contextlib import contextmanager
try:
    import pillow_heif
except ImportError:
    import pi_heif as pillow_heif
pillow_heif.register_avif_opener()


@contextmanager
def Timer(msg):
    try:
        start = time.time()
        yield
    finally:
        end = time.time()
        print('%s: %.3f s' % (msg, end - start))


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', nargs='?', default=0, type=int)
    p.add_argument('height', nargs='?', default=0, type=int)
    return p.parse_args()

ACES_AP0_coords = ((0.735, 0.265),
                   (0.0, 1.0),
                   (0.0, -0.077),
                   (0.322, 0.338))

# ACES AP0 v4 ICC profile with linear TRC    
ACES_AP0 = b'AAACsGxjbXMEMAAAbW50clJHQiBYWVogB+YABgAIAAcAOQAAYWNzcEFQUEwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1sY21zAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMZGVzYwAAARQAAABMY3BydAAAAWAAAABMd3RwdAAAAawAAAAUY2hhZAAAAcAAAAAsclhZWgAAAewAAAAUYlhZWgAAAgAAAAAUZ1hZWgAAAhQAAAAUclRSQwAAAigAAAAQZ1RSQwAAAjgAAAAQYlRSQwAAAkgAAAAQY2hybQAAAlgAAAAkZG1uZAAAAnwAAAAybWx1YwAAAAAAAAABAAAADGVuVVMAAAAwAAAAHABBAEMARQBTAC0AQQBQADAALQBEADYAMABMAGkAbgBlAGEAcgBfAGcAPQAxAC4AMG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAMAAAABwATgBvACAAYwBvAHAAeQByAGkAZwBoAHQALAAgAHUAcwBlACAAZgByAGUAZQBsAHlYWVogAAAAAAAA9tYAAQAAAADTLXNmMzIAAAAAAAEIvwAABE7///ZoAAAFiQAA/gP///y+///+OQAAAuYAANAhWFlaIAAAAAAAAP2rAABcpf///05YWVogAAAAAP//9gn//+pkAADRwlhZWiAAAAAAAAADIgAAuPYAAAIdcGFyYQAAAAAAAAAAAAEAAHBhcmEAAAAAAAAAAAABAABwYXJhAAAAAAAAAAAAAQAAY2hybQAAAAAAAwAAAAC8FQAAQ+sAAAAAAAEAAAAAAAf//+xKbWx1YwAAAAAAAAABAAAADGVuVVMAAAAWAAAAHABSAGEAdwBUAGgAZQByAGEAcABlAGUAAA=='


def compute_xyz_matrix(key):
    def xyz(xy): return xy[0], xy[1], 1.0 - xy[0] - xy[1]
    r, g, b, w = map(xyz, key)
    w = [w[0]/w[1], 1.0, w[-1]/w[1]]
    m = numpy.array([[r[0], g[0], b[0]],
                     [r[1], g[1], b[1]],
                     [r[2], g[2], b[2]]])
    coeffs = numpy.linalg.solve(m, w)
    return m @ numpy.diag(coeffs)


class NclxProfile:
    def __init__(self, t):
        self.color_primaries = t[1]
        self.transfer_characteristics = t[2]
        self.matrix_coefficients = t[3]
        self.red_xy = t[5], t[6]
        self.green_xy = t[7], t[8]
        self.blue_xy = t[9], t[10]
        self.white_xy = t[11], t[12]

    def __str__(self):
        def xy(t): return tuple(map(lambda n: round(n, 3), t))
        return f'nclx: {self.color_primaries}/{self.transfer_characteristics}/{self.matrix_coefficients} - r: {xy(self.red_xy)}, g: {xy(self.green_xy)}, b: {xy(self.blue_xy)}, w: {xy(self.white_xy)}'

# end of class NclxProfile

sRGB_nclx = NclxProfile(
    (1, 1, 13, 5, 1,
     0.6399999856948853, 0.33000001311302185,
     0.30000001192092896, 0.6000000238418579,
     0.15000000596046448, 0.05999999865889549,
     0.3127000033855438, 0.32899999618530273))


def get_nclx(info):
    try:
        return NclxProfile(struct.unpack('BiiiBffffffff', info['nclx_profile']))
    except:
        return None

def getmatrix(nclx):
    if nclx:
        return compute_xyz_matrix([nclx.red_xy, nclx.green_xy, nclx.blue_xy,
                                   nclx.white_xy])
    else:
        return None


def pq(a, inv):
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 107.0 / 128.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    if not inv:
        # assume 1.0 is 100 nits, normalise so that 1.0 is 10000 nits
        a /= 100.0
        # apply the PQ curve
        aa = numpy.power(a, m1)
        res = numpy.power((c1 + c2 * aa)/(1.0 + c3 * aa), m2)
    else:
        p = numpy.power(a, 1.0/m2)
        aa = numpy.fmax(p-c1, 0.0) / (c2 - c3 * p)
        res = numpy.power(aa, 1.0/m1)
        res *= 100
    return res


def srgb(a, inv):
    if not inv:
        a = numpy.fmax(numpy.fmin(a, 1.0), 0.0)
        return numpy.where(a <= 0.0031308,
                           12.92 * a,
                           1.055 * numpy.power(a, 1.0/2.4)-0.055)
    else:
        return numpy.where(a <= 0.04045, a / 12.92,
                           numpy.power((a + 0.055) / 1.055, 2.4))


def rec709(a, inv):
    if not inv:
        a = numpy.fmax(numpy.fmin(a, 1.0), 0.0)
        return numpy.where(a < 0.018,
                           4.5 * a,
                           1.099 * numpy.power(a, 0.45) - 0.099)
    else:
        return numpy.where(a < 0.081,
                           a / 4.5,
                           numpy.power((a + 0.099) / 1.099, 1.0/0.45))


def hlg(a, inv):
    h_a = 0.17883277
    h_b = 1.0 - 4.0 * 0.17883277
    h_c = 0.5 - h_a * math.log(4.0 * h_a)
    if not inv:
        rgb = a
        rgb /= 10.0
        rgb = numpy.fmin(numpy.fmax(rgb, 1e-6), 1.0)
        rgb = numpy.where(rgb <= 1.0 / 12.0, numpy.sqrt(3.0 * rgb),
                          h_a * numpy.log(
                              numpy.fmax(12.0 * rgb - h_b, 1e-6)) + h_c)
        return rgb
    else:
        rgb = a
        rgb = numpy.where(rgb <= 0.5, rgb * rgb / 3.0,
                          (numpy.exp((rgb - h_c)/ h_a) + h_b) / 12.0)
        rgb *= 10
        return rgb


def linearize(data, nclx):
    if not nclx:
        return data
    shape = data.shape
    data = data.reshape(-1)
    if nclx.transfer_characteristics in (1, 6, 14, 15):
        # Rec.709
        data = rec709(data, True)
    elif nclx.transfer_characteristics == 13:
        # sRGB
        data = srgb(data, True)
    elif nclx.transfer_characteristics == 16:
        # PQ
        data = pq(data, True)
    elif nclx.transfer_characteristics == 18:
        # HLG
        data = hlg(data, True)
    else:
        pass
    return data.reshape(shape)


def read(opts):
    heif = pillow_heif.open(opts.input, convert_hdr_to_8bit=False)
    width, height = heif.size
    print(f'found image: {width}x{height} pixels, {heif.bit_depth} bits')
    if opts.width and opts.height:
        heif = pillow_heif.thumbnail(heif, max(opts.width, opts.height))
    with Timer('decoding'):
        #heif.convert_to('RGB;16')
        rgb = numpy.asarray(heif, dtype=numpy.float32) / (2**heif.bit_depth - 1)
        end = time.time()
    nclx = get_nclx(heif.info)
    if nclx:
        print('nclx profile: %s' % nclx)
    else:
        print('no nclx profile found, assuming sRGB')
        nclx = sRGB_nclx
    with Timer('linearization'):
        to_xyz = getmatrix(nclx)
        rgb = linearize(rgb, nclx)
        if to_xyz is not None:
            ap0_to_xyz = compute_xyz_matrix(ACES_AP0_coords)
            to_ap0 = numpy.linalg.inv(ap0_to_xyz) @ to_xyz
            shape = rgb.shape
            rgb = rgb.reshape(-1, 3).transpose()
            rgb = to_ap0 @ rgb
            rgb = rgb.transpose().reshape(*shape).astype(numpy.float32)
            profile = email.base64mime.decode(ACES_AP0)
        else:
            profile = None
    with Timer('saving'):
        tifffile.imwrite(opts.output, rgb)
        if profile is not None:
            md = pyexiv2.Image(opts.output)
            md.modify_icc(profile)
            md.close()
    

def main():
    opts = getopts()
    read(opts)


if __name__ == '__main__':
    main()
