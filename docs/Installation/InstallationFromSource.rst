.. _installationFromSource:

Installation for development or instrument control
**************************************************

.. warning::

    These installation instructions are outdated, but retained as they have a little more explanation for some of the
    choices. Most notably we now recommend python 3.6 or 3.7 for new installs. Use the :ref:`new instructions<installation>`
    as a first stop and read this in conjunction with the updated instructions.

If you want to contribute to PYMEs development, or you want to use PYME in a context (e.g. instrument control) which
requires modification to some of the scripts you will want to have the PYME sources somewhere you can easily find, modify
and update them.

If you just want to use the data analysis features then you are strongly advised to follow the instructions in :ref:`installationanaconda`
instead.

The easiest (and only supported) way of getting a development build up and running also uses
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_. Although PYME will work with other python 2.7 distributions,
this will involve considerably more effort to navigate the many :ref:`dependencies <prereqnoanaconda>`.

STEP 1: Installing Anaconda
===========================

Download and install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ from
https://store.continuum.io/cshop/anaconda/. Choose the **64 bit** version of **Python 2.7**
for your prefered platform.

.. warning::

    Anaconda is available in both Python 2.7 and Python 3.x flavours. PYME will **only** work with the Python 2.7 version.

STEP 2: Add the ``david_baddeley`` channel and install dependencies
===================================================================

We maintain a conda *channel* with packaged versions of a number of dependencies which are either not available through
the standard conda libraries or for which the required version is newer than the anaconda default. This channel also has
a *metapackage* ``pyme-depends`` which lists should install the dependencies [#pymedepends]_.

To tell anaconda about our channel, and to install the dependencies we need to open a terminal window [#terminal]_  and run the following two commands:

.. code-block:: bash

	conda config --add channels david_baddeley
	conda install pyme-depends

This should download and install the required dependencies.

STEP 3: Install git
===================

If you are contemplating a source install you probably don't need explicit instructions here, and might well already
have it, but as a fallback git is conda installable: ``conda install git``.

STEP 4: Install a c/c++ compiler
================================

Building PYME requires a c compiler. On Linux, make sure gcc, g++ and the standard development tools are installed. On OSX,
install XCode, and on windows download and install `MS Visual C for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_.

STEP 5: Get the code
====================

The code is stored in a github repository. Get the current copy by doing
::

    git clone https://github.com/python-microscopy/python-microscopy

If you are going to be doing significant development and plan on issuing pull requests, it might be reasonable to create
a private fork on github at this point and clone from that (this is reasonably easy to set up later, so if you're unsure
just pull the code from the default repository and go from there).


STEP 6: Install
===============

Change to the directory where you cloned the code and execute the following.

::

    python setup.py develop

.. note::
    If you are on OSX, you might want to do the following instead to avoid problems running GUI scripts:
    ::

        /PATH/TO/ANACONDA/python.app/Contents/MacOS/python setup.py develop

    this will ensure that the correct "shebang" is used to ensure that you don't run into issues with using a
    non-framework build.


At this point you should have a functioning install.

.. note::
    Due to some slightly weird interplay between numpy distutils and setuptools ``python setup.py develop`` works from
    the base directory of the repository, but if you want to run ``python setup.py install`` instead you'll need to drop
    down one directory into the `PYME` directory first.


Further configuration (mostly optional)
=======================================

Windows
-------

If not already done by the setup script, create shortcuts somewhere (eg the start menu), to the following scripts:

- ``PYTHONDIR\Scripts\PYMEAquire.exe`` (data acquisition)
- ``PYTHONDIR\Scripts\launchWorkers.exe`` (real time analysis)
- ``PYTHONDIR\Scripts\dh5view.exe`` (raw data viewer)
- ``PYTHONDIR\Scripts\LMVis\VisGUI.exe`` (analysed data viewer)

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


OSX
---

Build the opener stubs (to allow file association) by executing the following:

::

    cd osxLaunchers
    xcodebuild -alltargets



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


.. _prereqnoanaconda:

Prerequisites for the adventurous
=================================

The prefered way of getting pre-requisites is to use the ``pyme-depends`` package as noted above. If using a non-anaconda
python distribution, the pre-requisites will need to be sources and installed manually. Below is an **outdated and
unmaintained** list of pre-requisites. A more up to date list can be found by looking at ``meta.yaml`` file used to generate
the ``conda`` package.

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


.. rubric:: Footnotes

.. [#pymedepends] This package should track with the current state of the dependencies. At present, however, the dependencies
    for the python-microscopy package are likely to be updated earlier and more often. If a dependency seems to be missing,
    check the ``meta.yaml`` file in the python-microscopy directory and if necessary use ``conda install`` to install the
    missing package. Please report any missing dependencies so I can fix them.

.. [#terminal] On OSX use spotlight to launch the **Terminal** app, on Windows, launch the **Anaconda Command Prompt**
    from the "Anaconda" group in the start menu.