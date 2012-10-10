.. _installationFromSource:

Installation From Source
#########################

Prerequisites
=============

PYME requires:

- Python 2.6 or 2.7(reccomended) *
- a c compiler*
- python-dev (only on Linux - has the Python development header files)

and the following Python packages:

- Numpy*
- Scipy*
- Matplotlib*
- wxPython* (>2.8.11)
- pytables*
- Pyro (3.9.1 - any 3.x version should work - the newer version 4 won't)
- PyOpenGL*
- PIL (Python Imaging Library)*
- pywin32* (only req. on windows)

For full functionality, the following are also useful:

- PySerial
- PyParallel
- PyFFTW3
- MySQL-python
- Django (>1.2)
- Django-south
- Mayavi2*
- Traits*
- Delny
- jinja2*
- cherrypy

\* part of Enthought python.

On Windows (and probably MacOS) the easiest way to get the pre-requisites is to
install the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_
and then pull the remaining packages using ``pip`` or ``easy_install``.
Under linux I'd use the default python install and the distribution packages for
the rest. Depending on how old your distro is you might want to use
pip/easy_install for some of the packages to get a more recent version instead
(notably Pyro, and WxPython).

There are a couple of little caveats though:

- We need a very recent version of wxPython. This means that you'll probably have to 
  upgrade the wxPython found in EPD and/or older linux distros. See upgrading wxPython below.
- I have had problems with getting Delny to compile/install on Win/OSX, although
  this might have been fixed in the interim. It's only required in PYME for some very rarely 
  used functionality, so can usually be safely ignored.
- On some (most?) versions of windows, the network stack is broken. Pyro needs 
  to be told about this by setting an environment variable - ``PYRO_BROKEN_MSGWAITALL=1``. 
  I can confirm that this is the case for pretty much every XP system, but can't comment on Vista/7). 
  Pyro falls back on another (slower) piece of code when this the flag is set, 
  so it should be safe in any case. Can't remember exactly how to diagnose the 
  problem other than that Pyro falls over with cryptic error messages.


To make this whole process of finding and installing dependencies a little less painful,
I've put together lists of required and recommended modules that can be used with
``pip`` to pull any remaining dependencies. Thus one would execute::

 pip install -r recommended-modules.txt

to get everything I think is going to be useful, or::

 pip install -r required-modules.txt

to get the bare essentials. Obviously this requires `pip <http://pypi.python.org/pypi/pip>`_
to be installed first. I would suggest installing at least numpy and scipy manually
because pip defaults to building everything from source, which is likely to be
somewhat painful for numpy and scipy. On Ubuntu/Debian systems running the
``install_dependencies.py`` script will try to install the dependencies using system
packages first and then resort to *pip* for anything left over.



Pyro Nameserver
---------------

You need to run a `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_ nameserver somewhere on your network segment. For testing, the easiest thing is to run ``pryo_ns`` (or ``pyro-nsd``) from the command line. There can, however, only be one nameserver on the network segment, so long term you might want to find a computer that's always on and run it on that. If it's a linux box, there might be some trickery involved to make sure it binds to the the external interface rather than localhost (specifically, the hostname has to resolve to the external interface).

You'll also want some form of mercurial client if checking out of the repository.

Installing
==========

Get the code
------------

The code is stored in a mercurial repository. Get the current copy by doing
::

    hg clone https://code.google.com/p/python-microscopy/ 

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

Upgrading wxPython
------------------

Linux (system python on Ubuntu/Debian)
**************************************
The easiest way is to add the relevant 
`repositories <http://wiki.wxpython.org/InstallingOnUbuntuOrDebian>`_ and do 
an ``apt-get upgrade``.

Enthought Python Distribution (windows)
***************************************
1) Delete the existing wxPython distribution incluing all .egg files.
   This can be done by executing the ``remove_old_wx.py`` script that ships with PYME
   (this should be on the path after installing PYME, and can otherwise be found 
   in the ``scripts`` folder of the source distribution).
2) Download and install a newer wxPython from `www.wxpython.org <http://www.wxpython.org/>`_

Enthought Python Distribution (OSX) (thanks to Christian)
*********************************************************
1) Delete the existing wxPython distribution incluing all .egg files. You're going
   to have to dive in and manually remove the files from your ``site-packages`` directory.
2) Download the most recent OSX build for the version of python that
   came in your EPD distribution, and extract this to a temporary directory.
3) Copy the relevant directories across to your EPD site packages directory. This
   is likely to be somewhere below ``/Library/Frameworks/Python.framework/Versions/Current.``



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

You probably want to set at least PYMEDATADIR, as the default is not particularly useful.

You should now have a setup which works for simulation*, data analysis, & visualisation. Interfacing with hardware obviously requires a little more work - see :ref:`ConfiguringPYMEAcquire`.

\* simulation probably won't work perfectly until you've done the EMGain calibration section of :ref:`ConfiguringPYMEAcquire`.
