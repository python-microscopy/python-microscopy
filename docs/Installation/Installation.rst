.. _installation:

Get started
***********

Single-click installation for users (Windows-only)
==================================================

Download the latest installer from https://python-microscopy.org/downloads/. Double-click
the installer and follow instructions.


conda installation for users
============================

Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Then, open a command prompt and enter

.. code-block:: bash
	
    conda create -n pyme python=2.7
    source activate pyme
    conda config --add channels david_baddeley
    conda install python-microscopy

conda installation for developers
=================================

Create a Python 3.6 virtual environment with ``conda``. Then, open a command prompt and enter

.. code-block:: bash
	
    conda config --add channels david_baddeley
    conda install pyme-depends
    git clone git@github.com:python-microscopy/python-microscopy.git
    cd python-microscopy
    python setup.py develop

pip installation for developers
=================================

Create a Python 3.6 virtual environment with ``venv`` or ``conda``. Then, open a command prompt and enter

.. code-block:: bash
	
    git clone git@github.com:python-microscopy/python-microscopy.git
    cd python-microscopy
    pip install requirements.txt
    python setup.py develop

Enable bioformats compatibility (optional)
==========================================

Install Java Development Kit 1.8. Open a command prompt in the installation ``conda`` or ``venv`` 
environment and enter

.. code-block:: bash

    pip install javabridge
    pip install python-bioformats


Verify installation
*******************

single-click
============
Locate the **PYMEVisualize (VisGUI)** desktop shortcut. Double-click it and confirm the program launches.

conda or pip
============

From the command prompt, launch any of the following programs, which should have been
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

Single-click
============
If prompted with **Windows protected your PC**, click **More info** and then **Run anyway**. 

If prompted with **Installation error**, press **OK** and then **Ignore**.

conda for developers
====================

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
- Additional PYME installation docs can be found at https://github.com/csoeller/pyme-install-docs