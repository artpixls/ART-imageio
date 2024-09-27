#!/usr/bin/env python

"""
Script to generate a binary installation for Windows of the ART-imageio scripts
using PyInstaller.
(Note: this doesn't collect the external binary dependencies
needed to run the scripts)
"""

import os, sys, argparse
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
    for (dirpath, dirnames, filenames) in os.walk(imageio_dir):
        if dirpath != mydir:
            for name in filenames:
                if name.endswith('.py'):
                    sys.path.append(dirpath)
                    m = os.path.splitext(name)[0]
                    hidden_imports.append('--hidden-import=' + m)
        else:
            dirnames[:] = []
                
    os.chdir(opts.outdir)
    print(';; directory is: %s' % os.getcwd())

    sep = os.pathsep
    tool = os.path.join(mydir, 'driver.py')

    args = ['--name=python', '--clean'] + hidden_imports + [tool]
    PyInstaller.__main__.run(args)    


if __name__ == '__main__':
    main()
