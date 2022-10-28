#!/usr/bin/env python

from PIL import Image
import webp
import pyexiv2
import argparse


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', choices=['load', 'save'], required=True)
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('width', type=int, default=0, nargs='?')
    p.add_argument('height', type=int, default=0, nargs='?')
    return p.parse_args()


def copy_metadata(opts):
    md = pyexiv2.Image(opts.input)
    with pyexiv2.Image(opts.output) as out:
        icc = md.read_icc()
        if icc:
            out.modify_icc(icc)
        out.modify_exif(md.read_exif())
        out.modify_iptc(md.read_iptc())
        out.modify_xmp(md.read_xmp())


def load(opts):
    src = webp.load_image(opts.input, 'RGB')
    if opts.width and opts.height:
        src.thumbnail((opts.width, opts.height))
    src.save(opts.output)
    copy_metadata(opts)


def save(opts):
    src = Image.open(opts.input)
    out = webp.WebPPicture.from_pil(src)
    out.save(opts.output)
    copy_metadata(opts)


def main():
    opts = getopts()
    if opts.mode == 'load':
        load(opts)
    else:
        save(opts)


if __name__ == '__main__':
    main()
