#!/usr/bin/env python3

import os, sys
import argparse
import numpy
import OpenEXR, Imath
import tifffile
import subprocess


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('-c', '--compression',
                   choices=sorted(
                       a[:-len('_COMPRESSION')] for a in
                       dir(Imath.Compression)
                       if a.endswith('_COMPRESSION')),
                   default='ZIP')
    p.add_argument('-H', '--half', action='store_true')
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

def pth(fn):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), fn))

_profile_dict = {
    ACES_AP0_coords : pth('ap0.icc'),
    REC_709_coords : pth('rec709.icc'),
}
def getprofile(header):
    try:
        c = header['chromaticities']
        def mkt(p):
            return (round(p.x, 3), round(p.y, 3))
        key = (mkt(c.red), mkt(c.green), mkt(c.blue), mkt(c.white))
    except KeyError as e:
        key = REC_709_coords
    return _profile_dict.get(key)


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
    #assert len(channels) in (3, 4), channels
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
        profile = _profile_dict[ACES_AP0_coords]
    else:
        profile = getprofile(header)
    tifffile.imwrite(opts.output, rgb)
    if profile is not None:
        subprocess.run(['exiftool', '-icc_profile<=' + profile,
                        '-overwrite_original', opts.output], check=True)


def write(opts):
    data = tifffile.imread(opts.input)
    if opts.half:
        data = data.astype(numpy.half)
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
    header['compression'] = Imath.Compression(
        eval(f'Imath.Compression.{opts.compression}_COMPRESSION'))
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
