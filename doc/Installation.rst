.. _installation:

Installation
############

Prerequisites
=============

PYME requires:

- Python (2.5 or 2.6, might work with 2.7) *
- a c compiler*
- python-dev (only on Linux - has the Python development header files)

and the following Python packages:

- Numpy*
- Scipy*
- Matplotlib*
- wxPython* (>2.8.11)
- pytables*
- Pyro (3.9.1)
- PyOpenGL*
- PIL (Python Imaging Library)*
- pywin32* (only req. on windows)
- multiprocessing* (Python 2.5 only - part of standard libraries for newer versions on Python)

For full functionality, the following are also useful:

- PySerial
- PyParallel
- MySQL-python
- Django (>1.2)
- Mayavi2*
- Delny

\* part of Enthought python.

On Windows (and probably MacOS) the easiest way to get the pre-requisites is to
install the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_
and then pull the remaining packages using ``pip`` or ``easy_install``.
Under linux I'd use the default python install and the distribution packages for
the rest. Depending on how old your distro is you might want to use
pip/easy_install for some of the packages to get a more recent version instead
(notably Pyro, and WxPython).

There are a couple of little caveats though:

- We need a very recent version of wxPython. This means you'll probably have to upgrade the wxPython in EPD (either using pip/easy_install, or by downloading the installer). I found I had to delete the old wx directories and egg file before the new ones would be found because EPD is doing something wierd. In Linux, its just a matter of using ``easy_install -U wxpython``. On Macs (OS X, the port is generally debugged by Christian) updating wx in EPD involved downloading the latest python2.6 build for OS X (at the time that was wxPython2.8-osx-unicode-2.8.11.0-universal-py2.6.dmg downloaded from `www.wxpython.org <http://www.wxpython.org/>`_). I expanded that into a temporary directory and manually moved the relevant directories into the EPD file hierarchy which reside somewhere below /Library/Frameworks/Python.framework/Versions/Current.
- Delny is a bit of a mess (not in in the python package archive so no easy_install, iffy licensing, need to google, download source, & build, but default sources don't build on Win or MacOS). I'm trying to remove the dependency (currently only used by one minor component of the visualisation/postprocessing), but in the meantime the easiest thing would probably be to grab a copy of the sources I've hacked to work on Win/ Mac off me.
- On some (most?) versions of windows, the network stack is broken. Pyro needs to be told about this by setting an environment variable - ``PYRO_BROKEN_MSGWAITALL=1``. I can confirm that this is the case for pretty much every XP system, but can't comment on Vista/7). Pyro falls back on another (slower) piece of code when this the flag is set, so it should be safe in any case. Can't remember exactly how to diagnose the problem other than that Pyro falls over with cryptic error messages.


To make this whole process of finding and installing dependencies a little less painful,
I've put together a ``.deb`` file which depends on the modules available available
through the Ubuntu/Debian package manager. This should work with modern versions
of Ubuntu/Debian to give you the bulk of the dependencies.

There's also now lists of required and recommended modules that can be used with
``pip`` to pull any remaining dependencies. Thus one would execute::

 pip install -r recommended-modules.txt

to get everything I think is going to be useful, or::

 pip install -r required-modules.txt

to get the bare essentials. Obviously this requires `pip <http://pypi.python.org/pypi/pip>`_
to be installed first. I would suggest using the ``.deb`` first, or, if on windows,
manually installing at least numpy and scipy because pip defaults to building
everything from source, which is likely to be somewhat painful for numpy and scipy.


Pyro Nameserver
---------------

You need to run a `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_ nameserver somewhere on your network segment. For testing, the easiest thing is to run ``pryo_ns`` (or ``pyro-nsd``) from the command line. There can, however, only be one nameserver on the network segment, so long term you might want to find a computer that's always on and run it on that. If it's a linux box, there might be some trickery involved to make sure it binds to the the external interface rather than localhost (specifically, the hostname has to resolve to the external interface).

You'll also want some form of mercurial client if checking out of the repository.

Installing
==========

Create a directory to hold the code. This directory is going to be added to the Python path, so it's probably not a good idea to just use your home directory, or ``c:``. On Windows I use ``c:\PYME``, on Linux I tend to use ``~/PYME``.

Get the code
------------

The code is stored in a mercurial repository. If you're on the local network you can get the current copy by doing
::

    cd ~/PYME
    hg clone http://lmsrv1/hg/PYME

or the equivalent using a gui client (eg `TortoiseHG <http://tortoisehg.bitbucket.org/>`_). If you're not on the network, then extract the archive I've sent you into this directory.

Build the c extension modules
-----------------------------

Open a terminal and execute the following:

::

    c:
    cd c:\PYME\PYME
    python setup.py build_ext -i

obviously replacing the path with the relevant one. The -i flag tells python to do an inplace build, which is what we want for a development install.

Tell python where to find it
----------------------------

So that python can use the modules it needs to know how to find them. This is done by setting the ``PYTHONPATH`` environment variable, by, for example, adding the following line to ``.profile``.
::

    export PYTHONPATH=/home/david/PYME


Note that these instructions are for installing a development copy of PYME (ie in a local directory to which you have write access to). This is usually what you want, there are however a couple of situations (eg when using on a multi-user linux box, or as part of a web server) where you might want to do a more conventional install and have PYME wind up in your Python site-packages directory. To do this, just run ``sudo python setup.py install`` rather than ``python setup.py build_ext -i`` and dont worry about setting ``PYTHONPATH``.

Windows
-------

Create shortcuts somewhere (eg the start menu), to the following scripts:

- ``PYME\Acquire\PYMEAquire.py`` (data acquisition)
- ``PYME\ParallelTasks\launchWorkers.py`` (real time analysis)
- ``PYME\DSView\dh5view.cmd`` (raw data viewer)
- ``PYME\Analysis\LMVis\VisGUI.cmd`` (analysed data viewer)

Optionally associate .h5 files with dh5view (will also open .tif,  .kdf & .psf if you want) and .h5r files with VisGUI. I typically do this by clicking on one of the files, using the 'Open With' option, and telling it to remember. If you're using the sample database, you can also associate .pmu files with ``PYME\FileUtils\pymeUrlOpener.cmd``.

Linux (Gnome)
-------------

Change to the ``PYME/FileUtils`` directory and run ``install_gnome.sh``. This should (hopefully) create links to the relevant programs in ``~/bin`` and set up associations and :) thumbnailing! With any luck, file permissions should be OK out of the repository, but there's a chance you're going to have to make a couple of the scripts executable.

.. _basicconfig:

Basic Configuration
-------------------

In addition to the setup detailed above, PYME has a couple of configuration options which are controlled by environment variables. These are:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|


==================    ======================================================
PYME_TASKQUEUENAME    Specifies the name of the task queue which processes
                      (analysis/ acquisition) bind to. Needed if you want
                      to run independent analyses on multiple computers.

PYMEDATADIR           Default location where PYMEAcquire saves data. Eg
                      ``D:\``. Also place where other parts of analysis
                      chain look for data.

PYMEMICRPATH          Used with sample database to know where (on the local
                      machine) the microscope computer is mapped.

PYMENASPATH           As for PYMEMICRPATH, but for the NAS
==================    ======================================================

You probably want to set at least PYMEDATADIR, as the default is not particularly useful.

You should now have a setup which works for simulation*, data analysis, & visualisation. Interfacing with hardware obviously requires a little more work - see :ref:`ConfiguringPYMEAcquire`.

\* simulation probably won't work perfectly until you've done the EMGain calibration section of :ref:`ConfiguringPYMEAcquire`.
