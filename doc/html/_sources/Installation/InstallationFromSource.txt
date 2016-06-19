.. _installationFromSource:

Installation From Source
#########################

Prerequisites
=============

PYME requires:

- Python 2.7
- a c compiler (on windows I recommend the free *Visual C for python*, on linux or OSX just use the platform gcc)
- python-dev (only on Linux - has the Python development header files)

and the following Python packages:

- Numpy
- Scipy
- Matplotlib
- wxPython (>2.8.11)
- pytables
- Pyro (any 3.x version should work - the newer version 4 won't)
- PyOpenGL
- PIL (Python Imaging Library)
- pywin32 (only req. on windows)

For full functionality, the following are also useful:

- PySerial       [acquisition with real hardware]
- PyFFTW3
- MySQL-python   [sample DB server]
- Django (>1.2)  [sample DB server]
- Django-south   [sample DB server]
- Mayavi2
- traits
- traits-ui
- Delny          [some *very* rarely used segmentation code]
- jinja2
- cherrypy
- scikit-image
- scikit-learn
- networkx
- toposort
- shapely
- zeroconf
- requests
- pandas
- yaml

You'll also want some form of mercurial client if checking out of the repository.


On Windows and OSX the easiest way to get the pre-requisites is to
install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ and to pull the remaining
dependencies using `conda install` (see :ref:`installationAnaconda`). Other pre-packaged 
scientific python distributions (e.g. WinPython, PythonXY, Canopy) can also be used, although
Anaconda seems to be the easiest. I would not reccomend trying to use the system python on OSX.

Under linux you have the choice of using Anaconda (or similar), or using the default python install and 
the distribution packages for the dependencies. Depending on how old your distro is you might want to use
pip/easy_install for some of the packages to get a more recent version instead
(notably Pyro, and WxPython). Which choice you make is going to depend on your use case. If using it 
just for data analysis, then Anaconda might be easier. If you want to use the sample information database
with apache and mod-python, then its likely to be easier to use the system python. Until recently I've always
used the system python. 

Dependencies which cannot be fount using either the Anaconda channels, or system packages
can be installed using `pip <http://pypi.python.org/pypi/pip>`_. If using Anaconda, add the `david_baddeley`
channel before resorting to pip as this should have most/all of the dependencies not included in Anaconda. 

There are a couple of (mostly historical) caveats:

- I have had problems with getting Delny to compile/install on Win/OSX, although
  this might have been fixed in the interim. It's only required in PYME for some very rarely 
  used functionality, so can usually be safely ignored.
- On some (most?) versions of windows, the network stack is broken. Pyro needs 
  to be told about this by setting an environment variable - ``PYRO_BROKEN_MSGWAITALL=1``. 
  I can confirm that this is the case for pretty much every XP system, but can't comment on Vista/7). 
  Pyro falls back on another (slower) piece of code when this the flag is set, 
  so it should be safe in any case. Can't remember exactly how to diagnose the 
  problem other than that Pyro falls over with cryptic error messages.
- All nodes on the network need to have the same version of Pyro


Installing
==========

Get the code
------------

The code is stored in a mercurial repository. Get the current copy by doing
::

    hg clone https://bitbucket.org/david_baddeley/python-microscopy 

or the equivalent using a gui client (eg `TortoiseHG <http://tortoisehg.bitbucket.org/>`_). 
Alternatively download the tarball corresponding to a particular release and extract.

Build and install
-----------------------------

Open a terminal, change to the directory where you extracted the source and execute:

::

    python setup.py install
. 

Alternatively, if you're going to fiddle with the code
-------------------------------------------------------

This is mostly applicable when using the software for microscope control, where you
are going to have to tweak some of the configuration codes.

::
    
    python setup.py develop


Windows
-------

If not already done by the setup script, create shortcuts somewhere (eg the start menu), to the following scripts:

- ``PYTHONDIR\Scripts\PYMEAquire.py`` (data acquisition)
- ``PYTHONDIR\Scripts\launchWorkers.py`` (real time analysis)
- ``PYTHONDIR\Scripts\dh5view.cmd`` (raw data viewer)
- ``PYTHONDIR\Scripts\LMVis\VisGUI.cmd`` (analysed data viewer)

Where  ``PYTHONDIR`` is the location of your python installation (typically ``c:\Python27`` or similar)
Optionally associate .h5 files with dh5view (will also open .tif,  .kdf & .psf if you want) 
and .h5r files with VisGUI. I typically do this by clicking on one of the files, 
using the 'Open With' option, and telling it to remember. If you're using the 
sample database, you can also associate .pmu files with ``PYTHONDIR\Scripts\pymeUrlOpener.cmd``.

Linux (Gnome)
-------------

Change to the ``PYME/gnome`` directory and run ``install_gnome.sh``. This should 
(hopefully) set up 
associations and :) thumbnailing! With any luck, file permissions should be OK 
out of the repository, but there's a chance you're going to have to make a 
couple of the scripts executable.



.. _basicconfig:

Basic Configuration
-------------------

In addition to the setup detailed above, PYME has a couple of configuration 
options which are controlled by environment variables. These are:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|


==================    ======================================================
PYMEDATADIR           Default location where PYMEAcquire saves data. Eg
                      ``D:\``. Also place where other parts of analysis
                      chain look for data.

PYMEMICRPATH          Used with sample database to know where (on the local
                      machine) the microscope computer is mapped.

PYMENASPATH           As for PYMEMICRPATH, but for the NAS
==================    ======================================================

When useing PYME for data acquisition you probably want to set at least PYMEDATADIR, as the default is not particularly useful.

You should now have a setup which works for simulation*, data analysis, & visualisation. Interfacing with hardware obviously requires a little more work - see :ref:`ConfiguringPYMEAcquire`.

\* simulation probably won't work perfectly until you've done the EMGain calibration section of :ref:`ConfiguringPYMEAcquire`.
