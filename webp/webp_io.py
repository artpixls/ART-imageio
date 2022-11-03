#!/usr/bin/env python

from PIL import Image
import webp
import argparse
import subprocess


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['read', 'write'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', type=int, default=0, nargs='?')
    p.add_argument('height', type=int, default=0, nargs='?')
    return p.parse_args()


def copy_metadata(opts):
    subprocess.run(['exiftool', '-tagsFromFile', opts.input,
                    '-all', '-icc_profile', '-overwrite_original', opts.output],
                   check=True)


def read(opts):
    src = webp.load_image(opts.input, 'RGB')
    if opts.width and opts.height:
        src.thumbnail((opts.width, opts.height))
    src.save(opts.output)
    copy_metadata(opts)


def write(opts):
    src = Image.open(opts.input)
    out = webp.WebPPicture.from_pil(src)
    out.save(opts.output)
    copy_metadata(opts)


def main():
    opts = getopts()
    if opts.mode == 'read':
        read(opts)
    else:
        write(opts)


if __name__ == '__main__':
    main()
