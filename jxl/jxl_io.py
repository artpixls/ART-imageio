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
    return p.parse_args()


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
    tifffile.imwrite(opts.output, img)
    

def write(opts):
    fd, name = tempfile.mkstemp(suffix='.ppm')
    os.close(fd)

    data = tifffile.imread(opts.input)
    data = data.astype(numpy.dtype(numpy.uint16).newbyteorder('>'))
    with open(name, 'wb') as out:
        out.write(b'P6 ')
        out.write(str(data.shape[1]).encode('utf-8'))
        out.write(b' ')
        out.write(str(data.shape[0]).encode('utf-8'))
        out.write(b' ')
        out.write(b'65535\n')
        out.write(data.tobytes('C'))

    subprocess.run(['cjxl', '--container=1', name, opts.output], check=True)
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
