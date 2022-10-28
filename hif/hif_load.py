#!/usr/bin/env python

from PIL import Image
import argparse
import subprocess


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("width", type=int, nargs='?', default=0)
    p.add_argument("height", type=int, nargs='?', default=0)
    return p.parse_args()


def thumbnail(opts):
    res = subprocess.run(['heif-info', opts.input], stdout=subprocess.PIPE,
                         encoding='utf-8', check=True)
    w, h = 0, 0
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith('thumbnail:'):
            w, h = [int(t) for t in line[10:].split('x', 1)]
            break
    if w >= opts.width and h >= opts.height:
        subprocess.run(['heif-thumbnailer', opts.input, opts.output], check=True)
        thumb = Image.open(opts.output)
        thumb = thumb.crop((0, 0, w, h))
        thumb.save(opts.output)
    else:
        subprocess.run(['heif-thumbnailer', '-p',
                        '-s', f'{opts.width}x{opts.height}',
                        opts.input, opts.output], check=True)


def load(opts):
    subprocess.run(['heif-convert', opts.input, opts.output], check=True)


def main():
    opts = getopts()
    if opts.width and opts.height:
        thumbnail(opts)
    else:
        load(opts)


if __name__ == '__main__':
    main()
