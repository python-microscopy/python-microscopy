#!python
"""py2exe wrapper script for the win32 screensaver library.

USAGE:
  buildScr.py [options] filename

Options are:
    -g, --pygame        pygame mode
    -d, --data          copy data file

"filename" is the name of a python module that contains the screensaver
main code. This script creates an exe and renames it to scr.
The result is placed in the dist subdirectory.

The --pygame option is required when you build screensavers based on
pysrc_pygame. The default font is missing otherwise and the screensaver
will not run.

The --data option appends data files to the list of files that are
copied. Wildcards are allowed and the option may be specified multiple
times.

Happy screensaving.

(C) 2003 Chris Liechti <cliechti@gmx.net>
This is distributed under a free software license, see license.txt."""

from distutils.core import setup
import getopt, glob, py2exe, sys, os


pygameMode = False
data_files = []

try:
    opts, args = getopt.getopt(sys.argv[1:],
        "hgd:",
        ["help", "pygame", "data="]
    )
except getopt.GetoptError:
    # print help information and exit:
    print __doc__
    sys.exit(2)

for o, a in opts:
    if o in ("-h", "--help"):
        print __doc__
        sys.exit()
    elif o in ("-g", "--pygame"):
        pygameMode = True
    elif o in ("-d", "--data"):
        for filename in glob.glob(a):
            #~ print filename
            data_files.append(filename)

if not args:
    print __doc__
    sys.exit(1)

name = args[0]
#cut of file ending
if name.endswith(".py"):
    name = name[:-3]
elif name.endswith(".pyw"):
    name = name[:-4]

if pygameMode:
    #copy additional files for pygame
    import pygame
    pygamedir = os.path.split(pygame.base.__file__)[0]
    data_files.append(os.path.join(pygamedir, pygame.font.get_default_font()))

sys.argv[1:] = ['py2exe']
setup(
    options = {'py2exe': {
        'optimize': 2,
        'dist_dir': 'dist/%s' % name,
        }
    },
    name=name,
    windows=[args[0]],
    data_files = data_files,
)

#rename exe -> scr
if os.path.exists(os.path.join('dist', name, name+'.scr')):
    os.remove(os.path.join('dist', name, name+'.scr'))
os.rename(os.path.join('dist', name, name+'.exe'), os.path.join('dist', name, name+'.scr'))
