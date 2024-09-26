#!/usr/bin/env python

"""
windows driver to build a stand-alone python executable for the
ART-imageio plugins with PyInstaller
"""

import sys

if len(sys.argv) < 2:
    print('no script specified')
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    with open(script) as f:
        code = f.read()
    globals()['__file__'] = script
    exec(code)
