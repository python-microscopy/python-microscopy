.. _installationanaconda:

Installation of PYME on 64 bit Windows, OSX, or Linux
*****************************************************

.. warning::

    These installation instructions are outdated, but retained as they have a little more explanation for some of the
    choices. Use the :ref:`new instructions<installation>` as a first stop instead.

PYME requires python (version 2.7, 3.6, or 3.7) and a number of additional scientific packages.
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

STEP 1: Installing Miniconda
============================

Python-microscopy can either be installed on top of the full Anaconda distribution, or on top of a stripped down version,
called Miniconda. We reccomend the later as it is less likely to result in dependency issues. Download and install
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ from https://docs.conda.io/en/latest/miniconda.html .
Choose the **64 bit** version of **Python 2.7** for your prefered platform.

.. note::

    As of mid 2019, PYME will no longer install cleanly on top of a full anaconda install due to a broken dask package
    in the default conda channels (see https://github.com/dask/dask/issues/4846 - this fix does not seem to have propagated
    back to the anaconda default channels). We strongly recommend using miniconda instead (or a bare conda environment
    which does not include dask). If you need scikits-image (which has dask as a dependency), install python-microscopy
    first and then explicity install compatible versions of numpy, skimage, and dask.

.. warning::

    Miniconda is available in both Python 2.7 and Python 3.x flavours. PYME will **only** work with the Python 2.7 version.



STEP 2: Installing PYME using conda
===========================================

Anaconda comes with a built in package manager called **conda** which can be used to
install additional packages. In addition to the main set of packages maintained
by Continuim Analytics (the developers of anaconda) conda can install packages which
members of the community have uploaded to **binstar.org**. The python-microscopy package 
and a number of it's dependencies are available through the `david_baddeley` binstar channel. 
To install PYME, we need to tell conda to use the `david_baddeley` channel in addition to it's existing channels.
With miniconda (and potentially more recent versions of anaconda), we also need to add
the `anaconda` channel, which should be added before the `david_baddeley` channel to ensure that the `david_baddeley`
channel gets a higher priority and can over-ride the broken fftw package in the anaconda channel. We can then tell
conda to install the package named `python-microscopy`.

This is accomplished by opening a terminal window (On OSX use spotlight to launch the **Terminal** 
app, on Windows, launch the **Anaconda Command Prompt** from the "Anaconda" group in the 
start menu) and running the following three commands:

.. code-block:: bash
	
    conda config --append channels anaconda
    conda config --add channels david_baddeley
    conda install python-microscopy

This should download and install PYME, along with a number of it's dependencies.

.. note::

    **Troubleshooting:** Installing on top of a recent full anaconda distribution will likely fail due to dependency
    conflicts. The easiest solution is to use miniconda instead, or alternatively install PYME into a clean conda
    environment (e.g. ``conda create -n PYME python-microscopy``). Doing the latter will require you to activate the
    environment (e.g. ``conda activate PYME``) before running any of the PYME commands. It might also break the GUI
    shortcuts and file ascociations on windows.

    Other dependency issues can result in an old version of PYME being installed (most likely in older anaconda installs)
    A good sanity check is to look at what version conda wants to install when you run `conda install python-microscopy`.
    If it's older than a month or two (PYME uses date based versions) something is going wrong.

.. warning::

    **Avoid the `conda-forge` channel**. This mostly applies to people who want to use PYME with an existing anaconda
    installation or who are doing further development. Whilst `conda-forge` is appealing due to the large
    number of packages available, my experience is that it often results in a broken conda installation.
    My recommendations are thus:

    * Never add conda-forge to your channels
    * If you must install a package from conda-forge, use conda-forge just for that one package, e.g.
      `conda install -c conda-forge package` and the default channels for everything else. Double check
      what other packages it wants to download/ update to satisfy it's dependencies.


STEP 3: Verifying the Installation
==================================

From the command prompt, launch any of the following programs, which should have been
installed as part of PYME.

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

+------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``dh5view -t -m lite`` | This is the data viewer for image files (also used to launch localization analysis). The **-t -m lite** options      |
|                        | initiates a test mode which should display a image consisting of random noise.                                       |
+------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``PYMEAcquire``        | This is the data acquistion component, which when launched without any options will start with simulated hardware.   |
+------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``VisGUI``             | This is a viewer for point data sets. When launched without any parameters it will show a large pink triangle.       |
+------------------------+----------------------------------------------------------------------------------------------------------------------+


STEP 4: Setting up bioformats importing [optional]
==================================================

PYME (or specifically dh5view) can use bioformats to load data formats it doesn't natively support. For this to work you need to have java (JRE should be enough, but as the JDK is needed to compile the interface modules I have only tested with that) and the following 2 python modules installed:

- python-javabridge
- python-bioformats

For OSX, I have compiled versions of these in the `david_baddeley` channel which you can get using ``conda install``. On other platforms you will have to download the JDK and build these from source (both are on github). You might also get away with ``pip install`` ing them.


