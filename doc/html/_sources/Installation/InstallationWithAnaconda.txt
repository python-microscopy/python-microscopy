.. _installationAnaconda:

Installation of PYME on 64 bit Windows or OSX
#####################################################

PYME requires python (version 2.7) and a number of additional scientific packages.
Although it is possible to install all packages individuall, and then install PYME,
by far the easiest way to get a system up and running is to install a pre-packaged 
'scientfic python' distribution. `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ 
is one such distribution which is free for both academic and commercial use and includes 
extensive package management capabilities which allow us to easily distribute and update 
PYME on a variety of platforms. We currently provide compiled packages for 64 bit windows and OSX. 

PYME also runs on linux, but is not currently available as an Anaconda package, and 
will instead need to be built from source.

STEP 1: Installing Anaconda
===========================

Download and install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ from 
https://store.continuum.io/cshop/anaconda/. Choose the 64 bit version of Python 2.7
for your prefered platform. 

STEP 2: Installing PYME using conda
===========================================

Anaconda comes with a built in package manager called **conda** which can be used to
install additional packages. In addition to the main set of packages maintained
by Continuim Analytics (the developers of anaconda) conda can install packages which
members of the community have uploaded to **binstar.org**. The python-microscopy package 
and a number of it's dependencies are available through the `david_baddeley` binstar channel. 
To install PYME, we first need to tell conda to use the `david_baddeley` channel
in addition to it's existing channels. We can simply tell conda to install the package
named `python-microscopy`.

This is accomplished by opening a terminal window (On OSX use spotlight to launch the **Terminal** 
app, on Windows, launch the **Anaconda Command Prompt** from the "Anaconda" group in the 
start menu) and running the following two commands:

.. code-block:: bash
	
	conda config --add channels david_baddeley
	conda install python-microscopy

This should download and install PYME, along with a number of it's dependencies.

STEP 3: Verifying the Installation
==================================

From the command prompt, launch any of the following programs, which should have been
installed as part of PYME.

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

========================	==================================================================================================================
``dh5view -t -m lite``		This is the data viewer for image files (also used to launch localization analysis). The **-t -m lite** options 
							initiates a test mode which should display a image consisting of random noise. 

``PYMEAcquire``				This is the data acquistion component, which when launched without any options will start with simulated hardware.

``VisGUI``					This is a viewer for point data sets. When launched without any parameters it will show a large pink triangle.
========================	==================================================================================================================



STEP 4: Configuration
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
