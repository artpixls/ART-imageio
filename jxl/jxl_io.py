#!/usr/bin/env python3

import os, sys
import argparse
import subprocess
import tempfile
import re
import numpy
import tifffile


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', type=int, nargs='?')
    p.add_argument('height', type=int, nargs='?')
    p.add_argument('--hdr', action='store_true')
    return p.parse_args()


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


def hlg(a, inv):
    h_a = 0.17883277
    h_b = 1.0 - 4.0 * 0.17883277
    h_c = 0.5 - h_a * numpy.log(4.0 * h_a)
    if not inv:
        rgb = a
        rgb /= 12.0
        rgb = numpy.fmin(numpy.fmax(rgb, 1e-6), 1.0)
        rgb = numpy.where(rgb <= 1.0 / 12.0, numpy.sqrt(3.0 * rgb),
                          h_a * numpy.log(
                              numpy.fmax(12.0 * rgb - h_b, 1e-6)) + h_c)
        return rgb
    else:
        rgb = a
        rgb = numpy.where(rgb <= 0.5, rgb * rgb / 3.0,
                          (numpy.exp((rgb - h_c)/ h_a) + h_b) / 12.0)
        rgb *= 12.0
        return rgb   


def get_profile(opts):
    res = subprocess.run(['jxlinfo', opts.input], stdout=subprocess.PIPE,
                         check=True, encoding='utf-8')
    profiles = {
        ('D65', 'sRGB primaries', 'sRGB transfer function') : ('rec709.icc', srgb),
        ('D65', 'Rec.2100 primaries', 'PQ transfer function') : ('rec2100.icc', pq),
        ('D65', 'Rec.2100 primaries', 'HLG transfer function') : ('rec2100.icc', hlg)
        }
    for line in res.stdout.splitlines():
        if line.startswith('Color space: '):
            bits = line[13:].split(', ')
            if bits[0] == 'RGB':
                key = tuple(bits[1:-1])
                return profiles.get(key)
    return None


def linearize(img, fun):
    shape = img.shape
    img = img.reshape(-1)
    img = fun(img, True)
    return img.reshape(shape)
    

def read(opts):
    fd, name = tempfile.mkstemp(suffix='.ppm')
    os.close(fd)    
    subprocess.run(['djxl', '--bits_per_sample=16',
                    opts.input, name], check=True)
    with open(name, 'rb') as f:
        data = f.read()
        info = re.split(b'\\s+', data, 4)
    os.unlink(name)
    
    img = numpy.frombuffer(info[-1],
                           dtype=numpy.dtype(numpy.uint16).newbyteorder('>'))
    img = img.reshape((int(info[2]), int(info[1]), 3))
    img = img.astype(numpy.float32) / 65535.0

    profile = get_profile(opts)
    if profile:
        img = linearize(img, profile[1])
    tifffile.imwrite(opts.output, img)
    if profile:
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), profile[0]))
        subprocess.run(['exiftool', '-icc_profile<=' + p,
                        '-overwrite_original', opts.output], check=True)
    

def write(opts):
    fd, name = tempfile.mkstemp(suffix='.ppm')
    os.close(fd)

    data = tifffile.imread(opts.input)
    if not opts.hdr:
        data = numpy.fmax(numpy.fmin(data, 1.0), 0.0)
    data *= 65535.0
    data = data.astype(numpy.dtype(numpy.uint16).newbyteorder('>'))
    with open(name, 'wb') as out:
        out.write(b'P6 ')
        out.write(str(data.shape[1]).encode('utf-8'))
        out.write(b' ')
        out.write(str(data.shape[0]).encode('utf-8'))
        out.write(b' ')
        out.write(b'65535\n')
        out.write(data.tobytes('C'))

    colorspace = [] if not opts.hdr else ['-x', 'color_space=RGB_D65_202_Per_PeQ']
    subprocess.run(['cjxl', '--container=1', name, opts.output] + colorspace,
                   check=True)
    os.unlink(name)
    
    subprocess.run(['exiftool', '-tagsFromFile', opts.input,
                    '-overwrite_original', opts.output], check=True)

        
def main():
    opts = getopts()
    if opts.mode == 'read':
        read(opts)
    else:
        write(opts)


if __name__ == '__main__':
    main()
