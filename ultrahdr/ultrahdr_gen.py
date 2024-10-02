#!/usr/bin/env python3

import os, sys
import tifffile
import numpy
import subprocess
import argparse
import tempfile
import math
import time


sRGB_to_xyz = numpy.array([
    [0.4360747,  0.3850649, 0.1430804],
    [0.2225045,  0.7168786,  0.0606169],
    [0.0139322,  0.0971045,  0.7141733]
    ], dtype=numpy.float32)

lum = sRGB_to_xyz[1]
to_yuv = numpy.array([lum, lum - [0, 0, 1], [1, 0, 0] - lum],
                     dtype=numpy.float32)
to_rgb = numpy.linalg.inv(to_yuv)


def pq(a, inv=False):
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 107.0 / 128.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    if not inv:
        a /= 100.0
        aa = numpy.power(a, m1)
        res = numpy.power((c1 + c2 * aa)/(1.0 + c3 * aa), m2)
    else:
        p = numpy.power(a, 1.0/m2)
        aa = numpy.fmax(p-c1, 0.0) / (c2 - c3 * p)
        res = numpy.power(aa, 1.0/m1)
        res *= 100
    return res


def srgb(a, inv=False):
    if not inv:
        a = numpy.fmax(numpy.fmin(a, 1.0), 0.0)
        return numpy.where(a <= 0.0031308,
                           12.92 * a,
                           1.055 * numpy.power(a, 1.0/2.4)-0.055)
    else:
        return numpy.where(a <= 0.04045, a / 12.92,
                           numpy.power((a + 0.055) / 1.055, 2.4))


def tonemap(x):
    c = 0 
    a = 1.0 - c
    mid = 0.18
    b = (a / (mid - c)) * (1.0 - ((mid - c) / a)) * mid
    gamma = math.pow((mid + b), 2.0) / (a * b)
    
    def rolloff(x):
        return a * x / (x + b) + c
    def contrast(x):
        return mid * numpy.power(x / mid, gamma)

    x = x.reshape(-1, 3).transpose()

    y, u, v = numpy.split(to_yuv @ x, 3, 0)

    h = numpy.max(y)
    if h <= 1:
        return x.transpose().reshape(-1).copy()

    def tm(a):
        return rolloff(contrast(a))

    hue = numpy.arctan2(u, v)
    rgb = tm(x)
    y, u, v = numpy.split(to_yuv @ rgb, 3, 0)
    sat = numpy.hypot(u, v)

    hue = 0.6 * hue + 0.4 * numpy.arctan2(u, v)

    u = sat * numpy.sin(hue)
    v = sat * numpy.cos(hue)
    oY = y

    yuv = numpy.stack([oY.transpose(), u.transpose(), v.transpose()], -1)
    rgb = to_rgb @ yuv.reshape(-1, 3).transpose()
    rgb = rgb.transpose().reshape(-1)
    return rgb


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('--sdr')
    p.add_argument('hdr')
    p.add_argument('output')
    return p.parse_args()


def save_hdr(data, outname):
    r, g, b = numpy.split(pq(data.reshape(-1)).reshape(-1, 3), 3, 1)
    d = 2**10-1
    packed = ((b * d).astype(numpy.uint32) << 20) \
        | ((g * d).astype(numpy.uint32) << 10) \
        | ((r * d).astype(numpy.uint32) << 0)
    with open(outname, 'wb') as out:
        out.write(packed.astype('<u4').tobytes())


def save_sdr(data, outname):
    r, g, b = numpy.split(srgb(data.reshape(-1)).reshape(-1, 3), 3, 1)
    d = 2**8-1
    packed = ((b * d).astype(numpy.uint32) << 16) \
        | ((g * d).astype(numpy.uint32) << 8) \
        | ((r * d).astype(numpy.uint32) << 0)
    with open(outname, 'wb') as out:
        out.write(packed.tobytes())


def read(filename):
    data = tifffile.imread(filename)
    h, w, p = data.shape
    if w & 1:
        data = numpy.delete(data, -1, 1)
    if h & 1:
        data = numpy.delete(data, -1, 0)
    return data


def main():
    opts = getopts()
    hdrdata = read(opts.hdr)
    height, width, planes = hdrdata.shape
    hdrdata = numpy.fmax(hdrdata.reshape(-1), 0)
    if not opts.sdr:
        sdrdata = tonemap(hdrdata)
    else:
        sdrdata = read(opts.sdr)
        h, w, p = sdrdata.shape
        assert height == h and width == w and planes == p
        sdrdata = numpy.fmax(sdrdata.reshape(-1), 0)
    with tempfile.TemporaryDirectory() as d:
        save_hdr(hdrdata, os.path.join(d, 'out.hdr'))
        save_sdr(sdrdata, os.path.join(d, 'out.sdr'))
        subprocess.run(['ultrahdr_app', '-m', '0',
                        '-p', os.path.join(d, 'out.hdr'),
                        '-y', os.path.join(d, 'out.sdr'),
                        '-w', str(width), '-h', str(height),
                        '-C', '0', '-t', '2', '-R', '1',
                        '-z', opts.output], check=True)
    subprocess.run(['exiftool', '-tagsFromFile', opts.hdr,
                    '-all', '-overwrite_original', opts.output],
                   check=True)

if __name__ == '__main__':
    main()
