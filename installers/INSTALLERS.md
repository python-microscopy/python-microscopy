# Single-click / non expert installers
This folder contains the code to create installers for windows and mac.


## Commonalities
- A script (or potentiall similar bash/powershell/bat scritps) to construct a self contained folder with python and PYME.
- the script should use uv to create an enviroment and install the current pip PYME package in it, relying on that package to determine any additional dependencies
- the script should accept a parameter for the output folder, defaulting to creating a `PYME` directory in the users home directory.
- download (curl) uv if not already installed.
- The script should idealy create executable top level aliases to PYMEAcquire, PYMEImage, PYMEVis and PYMEClusterOfOne (as defined in the pyproject.toml scripts section). The script should also create a alias activating venv for console use.
- Python version is defined by an easily editable TARGET_PYTHON, with an initial target of 3.13 - potentially in installer_defines.yaml or a an alternative readily (minimal external deps) cross-platform format. We ahould also put the list of entry points to expose in here (which are expected to be a subset of the entrypoints defined in pyproject.toml).
- this script acts as the default installer while we build the more native windows and mac options, and remains the installation script for linux.


## Windows
- Use InnoSetup to create an executable installer, from the bundle created by the common script
- Start menu group PYME, with shortcuts to the top level entry points as described above, and a `PYME Console` shortcut to create a shell with the venv active.
- File associations to be revisited after basics working.
- Ideally install for me only and all users options.
- This will eventually run in a github action on windows-latest (which should already have inno setup available).

## Mac
- Ultimate goal is an app package, derived from bundle, with appropriate entry points.
- Parked for now pending to a bit more research, but avoid things in script structure which will bite us on that later.
