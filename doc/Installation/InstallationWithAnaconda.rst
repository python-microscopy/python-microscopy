.. _installationanaconda:

Installation of PYME on 64 bit Windows, OSX, or Linux
*****************************************************

PYME requires python (version 2.7) and a number of additional scientific packages.
Although it is possible to install all packages individually, and then install PYME,
by far the easiest way to get a system up and running is to install a pre-packaged 
'scientfic python' distribution. `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ 
is one such distribution which is free for both academic and commercial use and includes 
extensive package management capabilities which allow us to easily distribute and update 
PYME on a variety of platforms. We currently provide compiled packages for 64 bit windows, OSX, and Linux.

.. note::

    This is the recommended way of installing PYME in most circumstances. If you absolutely don't want to deal with the
    command line there is also a :ref:`completely graphical way of doing the installation <installationanacondagui>`. If
    you are looking to actively develop PYME or want to use it to control microscope hardware, see :ref:`installationFromSource`.

.. note::

    The instructions here assume a clean anaconda install. If you already use anaconda for other work, consider installing
    PYME in a dedicated conda environment e.g. `conda create -n PYME python=2.7` (see https://conda.io/docs/user-guide/tasks/manage-environments.html for details).
    The downside of this is that you will need to run `source activate PYME` before you can run any of the PYME programs.
    You might also not be able to associate files to open with dh5view or VisGUI on windows.

STEP 1: Installing Anaconda
===========================

Download and install `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ from
https://store.continuum.io/cshop/anaconda/. Choose the **64 bit** version of **Python 2.7**
for your prefered platform.

.. warning::

    Anaconda is available in both Python 2.7 and Python 3.x flavours. PYME will **only** work with the Python 2.7 version.

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

.. note::

    **Troubleshooting:** There appears to be a dependency conflict between the `mayavi` (which we use for 3D
    visualization) and `navigator-updater` packages in recent versions of Anaconda. As `navigator-updater`
    is installed by default, this can prevent `python-microscopy` from installing. If the installation above fails
    with an error message about dependencies, try running ``conda uninstall navigator-updater`` and then re-running
    ``conda install python-microscopy``.

    Other dependency issues can result in an old version of PYME being installed (most likely in older anaconda installs)
    A good sanity check is to look at what version conda wants to install when you run `conda install python-microscopy`.
    If it's older than a month or two (PYME uses date based versions) something is going wrong.


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


STEP 4: Setting up bioformats importing [optional]
==================================================

PYME (or specifically dh5view) can use bioformats to load data formats it doesn't natively support. For this to work you need to have java (JRE should be enough, but as the JDK is needed to compile the interface modules I have only tested with that) and the following 2 python modules installed:

- python-javabridge
- python-bioformats

For OSX, I have compiled versions of these in the `david_baddeley` channel which you can get using ``conda install``. On other platforms you will have to download the JDK and build these from source (both are on github). You might also get away with ``pip install`` ing them.


