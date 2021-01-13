.. _installation:

Installation
************

The best to install PYME will depend on your background and whether you are already using python on your computer.

Executable installers (Windows and OSX)
=======================================

Recommended if you don't already have python on your computer and/or are unfamiliar with python. Download the latest installer from https://python-microscopy.org/downloads/. Double-click the installer and follow instructions. 


Installing using conda
======================

Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Then, open the *Anaconda prompt* [#anacondaprompt]_ and enter

.. code-block:: bash
	
    conda config --append channels anaconda
    conda config --add channels david_baddeley
    conda install python=3.7 python-microscopy

.. note::

    **Which Python version?** As of 2020/12/07, we recommend python 3.6 or 3.7 for new installs. We plan to add support for
    python=3.8 (as installed by default in miniconda) in the near future, but until such time downgrading to either 3.6 or 3.7
    with and explicit python=3.x argument to conda install is required.

    Having recently made the py2-> py3 transition, there are a still few components which might work better / are better tested on python 2.7
    (likely to only effect instrument control). As we are no longer building updated packages on 2.7 we recommend a source
    install in this case.


Updating
========

Assuming that you've installed using either the executable or conda routes, you can update PYME by dropping into the *Anaconda prompt* [#anacondaprompt]_ and entering:

.. code-block:: bash

    conda update python-microscopy


Development installs
====================

This assumes a basic familiarity with python and conda. We maintain a conda metapackage, ``pyme-depends`` for PYMEs dependencies, and reccomend a separate conda environment for development installs. Entering the following at the command prompt should get you a functional system, alter to suit your needs:

.. code-block:: bash
    
    conda config --add channels david_baddeley
    conda create -n pyme pyme-depends python=X.X
    conda activate pyme

    git clone https://github.com/python-microscopy/python-microscopy.git
    cd python-microscopy
    python setup.py develop

On OSX, use ``/path/to/conda/environment/python.app/Contents/MacOS/python setup.py develop`` instead  of ``python setup.py develop`` so that the PYME programs can access the screen. 


Enable bioformats data importers
================================

Install a JAVA JDK or JRE. Open a command prompt in the installation ``conda`` 
environment and enter

.. code-block:: bash

    conda install javabridge
    conda install python-bioformats

**Caveat:** This currently only works on OSX. If conda packages for javabridge and bioformats don't work, try pip. 



Verify installation
*******************

Locate the **PYMEVisualize (VisGUI)** desktop shortcut. Double-click it and confirm the program launches. If you don't have a desktop shortcut, launch any of the following programs from an anaconda prompt, which should have been
installed as part of PYME.

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``PYMEImage -t``        | This is for viewing images. The **-t** option initiates a test mode which displays an image of random noise.         |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``PYMEAcquire``         | This for acquiring data from a custom microscope. When launched without any options, it will start with simulated    |
|                         | hardware. It will display a live image of random noise, streamed from a simulated camera.                            |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| ``PYMEVisualize``       | This is for viewing point data sets. It shows a blank canvas when launched without any parameters.                   |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+

Troubleshooting
***************

Executable installers
=====================
If prompted with **Windows protected your PC**, click **More info** and then **Run anyway**. 

If prompted with **Installation error**, press **OK** and then **Ignore**.

Developer installs [OSX]
========================

On OSX, the following error may appear when launching a PYME application from the command line.

.. code-block:: bash

    This program needs access to the screen. Please run with a Framework build of python, 
    and only when you are logged in on the main display of your Mac.

This can be solved by the following.

.. code-block:: bash

    cd /path/to/python-microscopy/
    /path/to/mininconda/install/python.app/Contents/MacOS/python setup.py develop


Additional resources
********************

- Detailed developer installation docs are located at :ref:`installationFromSource`
- A step by step walkthough of installation using anaconda along with some troubleshooting tips can be found at :ref:`installationanaconda`


pip installation [EXPERIMENTAL]
===============================

You can also install PYME using pip, although we recommend this as a last resort as a conda based installation will generally give better performance and should be easier. When using pip, you might need to manually hunt down some dependencies, and for dependencies which don't have binary wheels, you might need to spend a lot of time setting up the development evironment and finding the DLLs etc which dependencies link against. Some of our dependencies also need to be compiled using gcc (rather than MSVCC), even on windows. Because we view this as a fallback when, e.g. conda can't come up
with a resolvable set of dependencies, or when you are installing on top of a bunch of existing packages, the pip packages depend only on numpy, with the rest of the dependencies being installed separately through the use of a requirements.txt file. 

.. code-block:: bash

    pip install -r https://raw.githubusercontent.com/python-microscopy/python-microscopy/master/requirements.txt
    pip install python-microscopy


If installing in a tricky evironment, you can manually edit requirements.txt before installing. You can also use the top line to setup for a development install.

Installation on python 2.7
==========================

On some instrument control computers, or when debugging potential regressions, it still makes sense to install PYME on
python 2.7. We have stopped building packages on py2.7, so you'll need a source install to get the most recent functionality
and fixes. Unfortunately it is becoming increasingly difficult to `conda` install a consistent environment on python 2.7.
As we are now focussed on py3 and things seem to change every couple of weeks we have given up on maintaining updated
py 2.7 installation instructions. It is still possible to get things running, but it will be a bit of trial and error and you will need to manually
up or downgrade some of the dependency packages. Good candidates for package conflicts would be `traitsui`, `pyface`, and
`wxpython`. You might also need to use the full MS visual studio (community edition should suffice) rather than the stripped down
msvc for python.

.. rubric:: Footnotes

.. [#anacondaprompt] On OSX or linux this is the command prompt. On Windows, this is accessed from the "Miniconda" or "PYME" folder in the start menu.


