# README #

This repository contains a list of [custom image format](https://bitbucket.org/agriggio/art/wiki/Customformats) plugins for the [ART](https://bitbucket.org/agriggio/art) raw processor.

### Structure of the repository ###

Each plugin consists of two parts: 

* A `format.txt` configuration file
* An optional `format` directory containing auxiliary files,

where `format` is the name of the file format handled by the plugin (e.g. `exr.txt`).

### Installation in ART ###

Each plugin can be installed separately, by simply copying both the `format.txt` and `format` directory to the `imageio` directory in the ART config folder (e.g. on Linux that would be `$HOME/.config/ART/imageio`).

Each plugin uses different external programs and dependencies, that are listed at the beginning of the `format.txt` file. Such dependencies must be installed separately.
Most of them require at least [Python](http://www.python.org) and [Exiftool](http://exiftool.org).

### License ###

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)
