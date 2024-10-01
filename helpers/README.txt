This directory contains miscellaneous helper files that can be useful for
packaging, installing, or distributing the ART-imageio plugins.

Currently the following helpers are provided:

- python/requirements.txt

    a pip-compatible list of required python modules required by the
    plugins. This can be used as follows:

     $ python -m pip install -r python/requirements.txt

- pyinstaller/

    a driver and installation script to build a stand-alone windows python
    executable with the required modules using PyInstaller. This can be used
    as follows:

     c:\> python pyinstaller/run_pyinstaller.py -o output_dir
