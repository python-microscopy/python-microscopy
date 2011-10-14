.. _installation:

Installation of PYME on windows with Enthought Python
#####################################################

STEP 1: Prerequisites - installing EPD
======================================

PYME requires python (ideally version 2.7) and a number of additional packages.
The easiest way to get a system up and running is to install the
`Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ (EPD)
which is free for academic use. This has most of the required dependencies, although
a couple will still need to be installed manually. Both 32 bit and 64 bit versions
are available and PYME works with either. If you want to access hardware (e.g. cameras)
and are running 64 bit windows you might need the 64 bit version (I've only tested it
with the 64 bit version). If given the option chose 'install for everyone / all users',
rather than doing a personal install. Under Win7/Vista? you might need to right click the EPD
installer and select 'run as administrator' to do an install for everyone.

For example, at the time of writing we downloaded and executed ``epd-7.1-2-win-x86_64.msi`` and just accepted all defaults
when queried by the installer.

STEP 2: Installing PYME using the installer
===========================================

This should be as simple as running the installer appropriate for your system. As explained above (prerequisites)
pick the installer that corresponds to your python version (this would be the python version that came with EPD) and
the type of windows that you run (32 or 64 bit, see above for more about 32 vs 64 bits).

To give a concrete example, with the current EPD 7.1.2, python version 2.7 is installed. We are using a 64 bit Win 7
installation on our latest machines, so the correct installer version for this setup should be named something
like ``PYME-XXX.win-amd64-py2.7.exe``.

Double click the installer, currently there is minimal user interaction required and it
should be safe to accept any defaults when prompted. Likely you will have to give permission to the installer
to modify the system (most installers will do that). Just say yes, and you should be ok.

The installer will attempt to install the additional module ``Pyro`` for you as well, which is a required module,
but that requires a working internet connection on your machine.

STEP 3: Upgrading wxPython
==========================

PYME needs a wxPython version >= 2.8.11. EPD comes with an
older version of wx and we need to remove this before installing a current wx.

The installer that you run in STEP 2 installs a script (``remove_old_wx.py``) to help with the removal,
the potentially messy bit of the process. Do the following to run the script:

- Open a command prompt (XP: start menu, run, cmd; Win 7: start menu, into ``Search programs and files`` type ``cmd``, hit enter)
- in the cmd-window that opens type ``remove_old_wx.py`` and hit enter 

Now install the new wxPython which you can download from `http://www.wxpython.org/download.php. <http://www.wxpython.org/download.php>`_
Grab the unicode version which matches your python version and windows version. To stay with our example, we have python 2.7 as a result
of installing EPD 7.2.1 and run a 64 bit win 7, so we grabbed the installer for ``64-bit python 2.7``, a file named ``wxPython2.8-win64-unicode-2.8.12.1-py27.exe``. Again it should be safe to just accept defaults when prompted by the installer.

At this stage you should have a working basic PYME installation. See below (``TESTING PYME``) how to test
that things are working basically ok.

STEP 4: Other Dependencies (mostly optional)
============================================

If everything went well up to here, this step should be truly optional. This mostly involves getting some
other packages which aren't in EPD and which are required for some extra functionality of PYME (see STEP 4b).

If, however, for some reason Pyro was not successfully installed during STEP 2, you can try to install Pyro
manually once more, using the methods described next.

TODO: we need to tell you how to test if Pyro has been installed successfully.

Python comes with some package managers to help you get these additional dependencies easily. One is called ``pip``,
alternatively you can use a slightly older one called ``easy_install``. (Note: ``pip`` does not seem to be part of EPD, so you can
try to get that one first with the command ``easy_install pip``). The computer you are using needs an internet connection
to pull the modules from the net.

To use ``pip``, open a command prompt (see STEP 3 how to get a prompt window) and type::

 pip install <module name>

alternatively, using ``easy_install``::

 easy_install <module name>

If you don't have an internet connection on the computer you're trying to install on,
you'll have to grab the installers seperately for each module you need. Here google is
your friend and googling "python <module name>" will usually get you there pretty quickly.
Once you have an installer, just double click on it to install the module.

STEP 4a: Pyro
+++++++++++++

There's only one required extra module, which is `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_.
Notably this should be one of the 3.X versions rather than the recently released Pyro4.
This should have been installed by the PYME installer (STEP 3) but if you get errors try running ``easy_install Pyro``.

STEP 4b: Extras
+++++++++++++++

None of these extras is required for core functionality. You seriously might want to skip this first
time around and come back to this step later if you want to try some of this additional functionality.

For full functionality the following are useful:

- PyFFTW3 (widefield/confocal deconvolution)
- PySerial (interfacing some hardware)
- PyParallel (ditto)
- MySQL-python (needed for Django and sample database) 
- Django (>1.2) (sample database)
- Delny  (triangle based segmentation)

Last time I tried, MySQL-python didn't play well with pip/easy_install on windows
and there is no official build for Win7. An unofficial one could be found at the time
of writing `here <http://www.codegood.com/archives/129>`_.
MySQL and Django are only needed for interacting with the sample database, however,
which requires quite a lot of additional setup. So don't try this unless you really know
why you want this.

The module ``Delny`` used to be problematic as well, although I think it's better in the 
current version. If you run into problems I can provide a patched version 
which I know ought to work. This is only used in a small measurement component 
of the visualisation software, so it can usually be safely ommitted.

It's possible that I've also forgotten something, so if PYME complains that it can't
find a module, try ``pip install``ing it.


Pyro Nameserver
===============

TODO: This is not really clear enough and will lead to questions. Distill the minimal
instructions that should work on a system which had no Pyro use before (which is ideally: 
``just do nothing and let PYME do it for you``). Then follow that by more advanced instructions
in a separate para.

You need to run a `Pyro <http://www.xs4all.nl/~irmen/pyro3/>`_ nameserver somewhere 
on your network segment. For testing, the easiest thing is to run ``pryo_ns`` 
(or ``pyro-nsd``) from the command line, or skip this entirely and let PYME launch 
one for you. There can, however, only be one 
nameserver on the network segment, so long term you might want to find a computer 
that's always on and run it on that. If it's a linux box, there might be some 
trickery involved to make sure it binds to the the external interface rather 
than localhost (specifically, the hostname has to resolve to the external interface).

TESTING the basic PYME installation
===================================

TODO: Some basic command to execute from the start menu: Launch PYME/LMVis and a window should open
that looks roughly like this...

TODO: provide simple example data set generated with simulator, both H5 and H5R for basic test run

.. _basicconfig:

Basic Configuration
===================

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
