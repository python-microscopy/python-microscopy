.. _installation:

Installation of PYME on windows with Enthought python
#####################################################

STEP 1: Installing EPD
======================

PYME requires python (ideally version 2.7) and a number of additional packages.
The easiest way to get a system up and running is to install the
`Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ (EPD)
which is free for academic use. This has most of the required dependencies, although
a couple will still need to be installed manually. These instructions are based on EPD 7.1.2, which 
ships with python 2.7, other versions of EPD might work, but a change in the 
underlying python version will need a new installer. 

Both 32 bit and 64 bit versions
are available and PYME works with either. If you want to access hardware (e.g. cameras)
and are running 64 bit windows you might need the 64 bit version (I've only tested it
with the 64 bit version). If given the option chose 'install for everyone / all users',
rather than doing a personal install. Under Win7/Vista? you might need to right click the EPD
installer and select 'run as administrator' to do an install for everyone.

STEP 2: Installing PYME using the installer
===========================================

This should be as simple as running the installer which is appropriate to your
version of EPD. ie ``PYME-X.X.X.win32-py2.7.exe`` for 32 bit EPD and 
``PYME-X.X.X.win-amd64-py2.7.exe`` for 64 bit EPD.
The installer will also attempt to download and install Pyro, a required dependency, 
for you as well. Automatically installing Pyro will only work if the computer that 
you are installing on 
has an internet connection, otherwise see specific instructions below.


STEP 3: Upgrading wxPython
==========================

PYME needs wxPython >= 2.8.11. EPD comes with an
older version of wx and we need to remove this before installing the up to date one.
When you install PYME, it installs a script (``remove_old_wx.py``) to help with 
the messy removal part of the process:

- Open a command prompt (start menu, run, cmd)
- type ``remove_old_wx.py`` and hit enter 
- install the new wx which you can download from `http://www.wxpython.org/download.php. <http://www.wxpython.org/download.php>`_
  Grab the unicode version which matches your python version (python version number and 32 / 64 bits).

At this stage you should have a working basic installation.

STEP 4: Other Dependencies (mostly optional)
============================================

In addition to those provided with EPD and which EPD, PYME can make use of a few
additional packages which need to be manually installed. Luckily Python comes with a package manager called ``pip``
(or alternatively a slightly older one called ``easy_install`` if you don't have pip on
your system) which makes this relatively easy if the computer you are using has an
internet connection.

To use ``pip``, open a command prompt and type::

 pip install <module name>

alternatively, using ``easy_install``::

 easy_install <module name>

If you don't have an internet connection on the computer you're trying to install on,
you'll have to grab the installers seperately for each module you need. Here google is
your friend and googling "python <module name>" will usually get you there pretty quickly.
Once you have an installer, just double click on it to install the module.

4a: Pyro
--------

This is required and should have been installed by the PYME installer. 
If this didn't work, try installing manually by entering::

 easy_install Pyro

at the command prompt. If you don't have internet access you'll need to download
it from `http://pypi.python.org/pypi/Pyro/ <http://pypi.python.org/pypi/Pyro/>`_
, unzip, change to the relevant directory, and run::

 python setup.py install

More information can be found at `http://www.xs4all.nl/~irmen/pyro3/ <http://www.xs4all.nl/~irmen/pyro3/>`_.
Notably this should be one of the 3.X versions rather than the recently released 
Pyro4.

You can test that Pyro is installed and functioning by typing::

    pyro-ns -h

at the command prompt - you should get usage information for the pyro nameserver.

4b: Extras
----------
None of these are required for core functionality, so are probably best deferred
until you are sure you want/need that feature:

- PyFFTW3 (widefield/confocal deconvolution)
- PySerial (interfacing some hardware)
- PyParallel (ditto)
- MySQL-python (needed for Django and sample database) 
- Django (>1.2) (sample database)
- Delny  (triangle based segmentation)

It's possible that I've also forgotten something, so if PYME complains that it can't
find a module, try ``pip install``ing it.


STEP 5: Configuration
=====================

.. _basicconfig:

Basic Configuration
-------------------

In addition to the setup detailed above, PYME has a couple of configuration options 
which are controlled by environment variables. These are:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|


==================    ======================================================
PYMEDATADIR           Default location where PYMEAcquire saves data. Eg
                      ``D:\``. Also place where other parts of analysis
                      chain look for data.

PYMEMICRPATH          Used with sample database to know where (on the local
                      machine) the microscope computer is mapped. Not relevant
                      unless you're using the sample information database.

PYMENASPATH           As for PYMEMICRPATH, but for the NAS
==================    ======================================================

You probably want to set PYMEDATADIR, as the default is not 
particularly useful. Environment variables can be set by right clicking on 
`My Computer` selecting `Properties` and then `Advanced System Settings`.

You should now have a setup which works for simulation*, 
data analysis, & visualisation. Interfacing with hardware 
requires a little more work - see :ref:`ConfiguringPYMEAcquire`.

\* simulation probably won't work perfectly until you've done the 
EMGain calibration section of :ref:`ConfiguringPYMEAcquire`.


Pyro Nameserver
---------------

You need to run a `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_ nameserver somewhere 
on your network segment. For testing, the easiest thing is to let PYME launch one for you. 

There can, however, only be one nameserver on the network segment and once you start
running PYME on multiple machines a somewhat more sophisticated solution is needed.
The nameserver can be started seperately from PYME by running ``pryo_ns`` 
(or ``pyro-nsd``) from the command line and my recommendation is to find a machine
which is always on (e.g. a server) and run it on that. Several linux distributions 
have packages for Pyro which set the nameserver up as a service, although there might be some 
trickery involved to make sure it binds to the the external interface rather 
than localhost (specifically, the hostname has to resolve to the external interface).

STEP 6: Testing
===============

The installer should have added a ``PYME`` folder to the start menu and each of the scripts
should launch some form of GUI - see the main documentaion for more details.

TODO - expand this with a few simple tests and example data.
 