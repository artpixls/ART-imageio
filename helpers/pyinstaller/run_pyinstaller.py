#!/usr/bin/env python

"""
Script to generate a binary installation for Windows of the ART-imageio scripts
using PyInstaller.
(Note: this doesn't collect the external binary dependencies
needed to run the scripts)
"""

import os, sys, argparse, tempfile, glob
import PyInstaller.__main__


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--outdir', required=True, help='output directory')
    return p.parse_args()


def main():
    mydir = os.path.abspath(os.path.dirname(__file__))
    imageio_dir = os.path.abspath(os.path.join(mydir, '..', '..'))
    opts = getopts()
    hidden_imports = []
    for fname in glob.glob(imageio_dir + '/*/*.py'):
        dirpath = os.path.dirname(fname)
        name = os.path.basename(fname)
        if dirpath != mydir:
            if name.endswith('.py'):
                sys.path.append(dirpath)
                m = os.path.splitext(name)[0]
                hidden_imports.append('--hidden-import=' + m)

    outdir = os.path.abspath(opts.outdir)
    sep = os.pathsep
    tool = os.path.join(mydir, 'driver.py')

    with tempfile.TemporaryDirectory() as d:
        args = ['--name=python',
                '--clean',
                '--exclude-module=tkinter',
                '--workpath=' + os.path.join(d, 'build'),
                '--distpath=' + outdir,
                '--specpath=' + d] + hidden_imports + \
                (['--strip'] if sys.platform != 'win32' else []) + \
                [tool]
        PyInstaller.__main__.run(args)    


if __name__ == '__main__':
    main()
