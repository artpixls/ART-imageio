#!/usr/bin/env python3

import os, sys, io
import argparse
import numpy
import OpenEXR, Imath
import tifffile
import pyexiv2
import email.base64mime


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', nargs='?', default=0, type=int)
    p.add_argument('height', nargs='?', default=0, type=int)
    return p.parse_args()

def a_tp(t):
    if t == Imath.PixelType(Imath.PixelType.FLOAT):
        return numpy.float32
    elif t == Imath.PixelType(Imath.PixelType.HALF):
        return numpy.half
    else:
        raise ValueError(str(t))

def e_tp(t):
    if t == numpy.float32:
        return Imath.PixelType(Imath.PixelType.FLOAT)
    elif t == numpy.half:
        return Imath.PixelType(Imath.PixelType.HALF)
    else:
        raise ValueError(str(t))

ACES_AP0_coords = ((0.735, 0.265),
                   (0.0, 1.0),
                   (0.0, -0.077),
                   (0.322, 0.338))
REC_709_coords = ((0.640, 0.330),
                  (0.3, 0.6),
                  (0.15, 0.06),
                  (0.313, 0.329))
# ACES AP0 v4 ICC profile with linear TRC    
ACES_AP0 = b'AAACsGxjbXMEMAAAbW50clJHQiBYWVogB+YABgAIAAcAOQAAYWNzcEFQUEwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1sY21zAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMZGVzYwAAARQAAABMY3BydAAAAWAAAABMd3RwdAAAAawAAAAUY2hhZAAAAcAAAAAsclhZWgAAAewAAAAUYlhZWgAAAgAAAAAUZ1hZWgAAAhQAAAAUclRSQwAAAigAAAAQZ1RSQwAAAjgAAAAQYlRSQwAAAkgAAAAQY2hybQAAAlgAAAAkZG1uZAAAAnwAAAAybWx1YwAAAAAAAAABAAAADGVuVVMAAAAwAAAAHABBAEMARQBTAC0AQQBQADAALQBEADYAMABMAGkAbgBlAGEAcgBfAGcAPQAxAC4AMG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAMAAAABwATgBvACAAYwBvAHAAeQByAGkAZwBoAHQALAAgAHUAcwBlACAAZgByAGUAZQBsAHlYWVogAAAAAAAA9tYAAQAAAADTLXNmMzIAAAAAAAEIvwAABE7///ZoAAAFiQAA/gP///y+///+OQAAAuYAANAhWFlaIAAAAAAAAP2rAABcpf///05YWVogAAAAAP//9gn//+pkAADRwlhZWiAAAAAAAAADIgAAuPYAAAIdcGFyYQAAAAAAAAAAAAEAAHBhcmEAAAAAAAAAAAABAABwYXJhAAAAAAAAAAAAAQAAY2hybQAAAAAAAwAAAAC8FQAAQ+sAAAAAAAEAAAAAAAf//+xKbWx1YwAAAAAAAAABAAAADGVuVVMAAAAWAAAAHABSAGEAdwBUAGgAZQByAGEAcABlAGUAAA=='

REC_709 = b'AAACwGxjbXMEMAAAbW50clJHQiBYWVogB+YACQANABMAKQAUYWNzcEFQUEwAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1sY21zAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANZGVzYwAAASAAAAA4Y3BydAAAAVgAAAA2d3RwdAAAAZAAAAAUY2hhZAAAAaQAAAAsclhZWgAAAdAAAAAUYlhZWgAAAeQAAAAUZ1hZWgAAAfgAAAAUclRSQwAAAgwAAAAQZ1RSQwAAAgwAAAAQYlRSQwAAAgwAAAAQY2hybQAAAhwAAAAkZG1kZAAAAkAAAABaZG1uZAAAApwAAAAibWx1YwAAAAAAAAABAAAADGVuVVMAAAAcAAAAHABMAGkAbgBlAGEAcgAgAFIAZQBjAC4ANwAwADltbHVjAAAAAAAAAAEAAAAMZW5VUwAAABoAAAAcAFAAdQBiAGwAaQBjACAARABvAG0AYQBpAG4AAFhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//8yUAAAeTAAD9kP//+6H///2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAAD4QAALbDWFlaIAAAAAAAAGKXAAC3hwAAGNlwYXJhAAAAAAAAAAAAAQAAY2hybQAAAAAAAwAAAACj1wAAVHsAAEzNAACZmgAAJmYAAA9cbWx1YwAAAAAAAAABAAAADGVuVVMAAAA+AAAAHAAjAHUAcwAvAHAAaQB4AGwAcwAvAEEAUgBUACMAMQAuADAAMAAwADAAMAAwADoAMAAuADAAMAAwADAAMAAhAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAYAAAAcAEEAUgBUAAA='

_profile_dict = {
    ACES_AP0_coords : ACES_AP0,
    REC_709_coords : REC_709,
}
def getprofile(header):
    try:
        c = header['chromaticities']
        def mkt(p):
            return (round(p.x, 3), round(p.y, 3))
        key = (mkt(c.red), mkt(c.green), mkt(c.blue), mkt(c.white))
        data = _profile_dict.get(key)
        if data is not None:
            return email.base64mime.decode(data)
    except KeyError as e:
        sys.stderr.write('ERROR: %s\n' % e)
    return None


def compute_xyz_matrix(key):
    def xyz(xy): return xy[0], xy[1], 1.0 - xy[0] - xy[1]
    r, g, b, w = map(xyz, key)
    w = [w[0]/w[1], 1.0, w[-1]/w[1]]
    m = numpy.array([[r[0], g[0], b[0]],
                     [r[1], g[1], b[1]],
                     [r[2], g[2], b[2]]])
    coeffs = numpy.linalg.solve(m, w)
    return m @ numpy.diag(coeffs)


def getmatrix(header):
    try:
        c = header['chromaticities']
        def mkt(p):
            return (round(p.x, 3), round(p.y, 3))
        key = (mkt(c.red), mkt(c.green), mkt(c.blue), mkt(c.white))
    except KeyError as e:
        # Rec.709 as default
        key = REC_709_coords
    if key in _profile_dict:
        return None
    else:
        return compute_xyz_matrix(key)


def read(opts):
    exr = OpenEXR.InputFile(opts.input)
    header = exr.header()
    box = header['dataWindow']
    width, height = box.max.x - box.min.x + 1, box.max.y - box.min.y + 1
    channels = header['channels']
    assert len(channels) in (3, 4)
    assert 'R' in channels and 'G' in channels and 'B' in channels
    t = a_tp(channels['R'].type)
    rgb = numpy.dstack([numpy.frombuffer(d, t).reshape(height, width)
                        for d in exr.channels('RGB')])
    if opts.width and opts.height:
        sw = int(width / opts.width)
        sh = int(height / opts.height)
        skip = max(min(sw, sh), 1)
        if skip != 1:
            rgb = numpy.array([row[::skip] for row in rgb[::skip]])
    to_xyz = getmatrix(header)
    if to_xyz is not None:
        ap0_to_xyz = compute_xyz_matrix(ACES_AP0_coords)
        to_ap0 = numpy.linalg.inv(ap0_to_xyz) @ to_xyz
        shape = rgb.shape
        rgb = rgb.reshape(-1, 3).transpose()
        rgb = to_ap0 @ rgb
        rgb = rgb.transpose().reshape(*shape).astype(numpy.float32)
        profile = email.base64mime.decode(ACES_AP0)
    else:
        profile = getprofile(header)
    tifffile.imwrite(opts.output, rgb)
    if profile is not None:
        md = pyexiv2.Image(opts.output)
        md.modify_icc(profile)
        md.close()


def write(opts):
    data = tifffile.imread(opts.input)
    r, g, b = data[...,0], data[...,1], data[...,2]
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    # assume ACES AP0 primaries and white point
    header['chromaticities'] = Imath.Chromaticities(
        Imath.chromaticity(0.7347, 0.2653),
        Imath.chromaticity(0.0, 1.0),
        Imath.chromaticity(0.0001, -0.0770),
        Imath.chromaticity(0.32168, 0.33767)
    )
    tp = e_tp(r.dtype)
    header['channels'] = {'R': Imath.Channel(tp),
                          'G': Imath.Channel(tp),
                          'B': Imath.Channel(tp)}
    exr = OpenEXR.OutputFile(opts.output, header)
    exr.writePixels({'R' : r.tobytes(), 'G' : g.tobytes(), 'B' : b.tobytes()})
    exr.close()
    

def main():
    opts = getopts()
    if opts.mode == 'read':
        read(opts)
    else:
        write(opts)


if __name__ == '__main__':
    main()
