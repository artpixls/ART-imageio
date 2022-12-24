#!/usr/bin/env python3
import subprocess
import argparse
import os


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    return p.parse_args()


def main():
    opts = getopts()
    pq = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pq.icc'))
    subprocess.run(['exiftool', '-icc_profile<=' + pq, '-overwrite_original',
                    opts.input], check=True)
    subprocess.run(['cjxl', opts.input, opts.output], check=True)


if __name__ == '__main__':
    main()
