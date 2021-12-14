"""

Modified from conda's cwp.py, this script attempts to set paths appropriately and then launch dh5view or VisGUI
depending on what it gets passed as a filename. Used for setting up file associations under windows.

"""

import os
import sys
import subprocess
from os.path import pathsep

if __name__ == '__main__':
    # TODO - is this safe if we are installed within a non-root conda env?
    prefix = sys.exec_prefix

    # call as: python cwp.py PREFIX ARGs...
    args = sys.argv[1:]

    new_paths = pathsep.join([prefix,
                            os.path.join(prefix, "Library", "mingw-w64", "bin"),
                            os.path.join(prefix, "Library", "usr", "bin"),
                            os.path.join(prefix, "Library", "bin"),
                            os.path.join(prefix, "Scripts")])

    env = os.environ.copy()
    env['PATH'] = new_paths + pathsep + env['PATH']
    env['CONDA_PREFIX'] = prefix


    filename = args[-1]
    extension = os.path.splitext(filename)[-1]

    try:
        os.chdir(os.path.dirname(filename))
    except:
        pass

    if extension in ['.h5r', '.csv', '.txt', 'hdf']:
        # VisGUI
        args = [os.path.join(prefix, 'Scripts', 'VisGUI.exe')] + args
    else:
        # Try dh5view for anything else

        args = [os.path.join(prefix, 'Scripts', 'dh5view.exe')] + args

    sys.exit(subprocess.call(args, env=env))