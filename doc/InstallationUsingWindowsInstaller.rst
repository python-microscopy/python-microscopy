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

STEP2: Installing PYME using the installer
==========================================

This should be as simple as running the installer which is appropriate to your
version of EPD. ie ``PYME-X.X.X.win32-py2.7.exe`` for 32 bit EPD and ``PYME-X.X.X.win-amd64-py2.7.exe``.
The installer will also attempt to download and install Pyro, a required dependency 
for you as well. 


Upgrading wxPython
------------------

PYME needs wxPython >= 2.8.11. EPD comes with an
older version of wx and we need to remove this before installing the up to date one.
When you install PYME, it installs a script (``remove_old_wx.py``) to help with the messy
bits of the process:

- Open a command prompt (start menu, run, cmd)
- type ``remove_old_wx.py`` and hit enter 
- install the new wx which you can download from `http://www.wxpython.org/download.php. <http://www.wxpython.org/download.php>`_
  Grab the unicode version which matches your python version.

Other Dependencies
------------------

In addition to wx, PYME needs a couple of other packages which aren't in EPD and which
need to be manually installed. Luckily Python comes with a package manager called ``pip``
(or alternatively a slightly older one called ``easy_install`` if you don't have pip on
your system) which makes this relatively easy if the computer you are using has an
internet connection.

To use ``pip``, open a command prompt and type::

 pip install <module name>

alternatively, using ``easy_install``:

 easy_install <module name>

If you don't have an internet connection on the computer you're trying to install on,
you'll have to grab the installers seperately for each module you need. Here google is
your friend and googling "python <module name>" will usually get you there pretty quickly.
Once you have an installer, just double click on it to install the module.

Pyro
++++
There's only one required extra module, which is `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_.
Notably this should be one of the 3.X versions rather than the recently released Pyro4.
This should have been installed by the PYME installer but if you get errors try running `easy_install Pyro`.

Extras
++++++
For full functionality, however, the following are useful:

- PyFFTW3 (widefield/confocal deconvolution)
- PySerial (interfacing some hardware)
- PyParallel (ditto)
- MySQL-python (needed for Django and sample database) 
- Django (>1.2) (sample database)
- Delny  (triangle based segmentation)

Last time I tried, MySQL-python didn't play well with pip/easy_install on windows
and there is no official build for Win7. An unofficial one can be found 
`here <http://www.codegood.com/archives/129>`_.
MySQL and Django are only needed for interacting with the sample database, however,
which requires quite a lot of additional setup.

Delny used to be problematic as well, although I think it's better in the 
current version. If you run into problems I can provide a patched version 
which I know ought to work. This is only used in a small measurement component 
of the visualisation software, so it can usually be safely ommitted.

It's possible that I've also forgotten something, so if PYME complains that it can't
find a module, try ``pip install``ing it.






Pyro Nameserver
---------------

You need to run a `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_ nameserver somewhere 
on your network segment. For testing, the easiest thing is to run ``pryo_ns`` 
(or ``pyro-nsd``) from the command line, or skip this entirely and let PYME launch 
one for you. There can, however, only be one 
nameserver on the network segment, so long term you might want to find a computer 
that's always on and run it on that. If it's a linux box, there might be some 
trickery involved to make sure it binds to the the external interface rather 
than localhost (specifically, the hostname has to resolve to the external interface).


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
