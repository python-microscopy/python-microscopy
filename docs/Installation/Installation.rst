.. _installation:

Installation
************

The best PYME installation method will depend on your background and whether you are already using python on your computer.

If you have an existing PYME installation, we recommend :ref:`removing it <removing>` before installing a new one.

Executable installers (Windows and Mac)
=======================================

Recommended if you don't already have Python on your computer and/or are unfamiliar with Python. Download the latest
installer from https://python-microscopy.org/downloads/. Double-click the installer and follow instructions.

Updating
---------

Open the *Anaconda prompt* [#anacondaprompt]_ and enter:

.. code-block:: bash

    conda update python-microscopy


Installing using conda
======================

Recommended if you already use Python. Note that while we recommend conda over pip, we do additionally offer :ref:`instructions for pip <pip>`.
Download and install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Then, open the *Anaconda prompt* [#anacondaprompt]_ and enter

.. code-block:: bash
	
    conda config --append channels anaconda
    conda config --add channels david_baddeley
    conda create -n pyme python=3.7 pyme-depends python-microscopy

To run programs from the command prompt, you will need to run `conda activate pyme` before running the program.

.. note::

    **Which Python version?** As of 2022/6/01, we recommend python 3.7 for new installs. Python 3.8 is also now well tested, but may need a little manual intervention and/or the use of conda-forge to 
    satisfy all dependencies. For a native install on apple-silicon (M1 etc ...) you need to do a development install using python 3.8 (see :ref:`apple_silicon`), although the standard mac packages should still work
    through rosetta.

Updating
---------

Open the *Anaconda prompt* [#anacondaprompt]_ and enter:

.. code-block:: bash

    conda update python-microscopy


Development & instrument computer installs
===========================================

This assumes a basic familiarity with python and conda. We maintain a conda metapackage, ``pyme-depends`` for PYME's dependencies, and reccomend a separate conda environment for development installs. Entering the following at the command prompt should get you a functional system, alter to suit your needs:

.. code-block:: bash
    
    conda config --add channels david_baddeley
    conda create -n pyme pyme-depends python=X.X
    conda activate pyme

    git clone https://github.com/python-microscopy/python-microscopy.git
    cd python-microscopy
    python setup.py develop

On OSX, use ``/path/to/conda/environment/python.app/Contents/MacOS/python setup.py develop`` instead  of ``python setup.py develop`` so that the PYME programs can access the screen. 

Windows users who do not already have MSVC build tools need to install them. On some verisons of Python this can be done using conda, however a more general approach is to download Visual Studio (the free, community version - the installer is also used for downloading build tools). 
Customize as needed, but for a 64 bit Windows 10 computer you will likely need the following individual components:

* Windows 10 SDK
* MSVC v142 - VS 2019 C++ x64/x86 build tools (latest)
* C++/CLI support for v142 build tools (latest)
* Windows Universal C runtime
* C++ Universal Windows Platform runtime
* C++ Build Tools core Features
* C++ core features
* .NET Framework 4.8 SDK
* .NET Framework 4.6.1 targeting pack 



Building/Editing documentation
---------------------------------

Building PYME documentation (thank you for helping!) requires additional packages which can be installed via conda:

.. code-block:: bash

    conda install mock numpydoc sphinx_rtd_theme

The documentation htmls can then be built by running 

.. code-block:: bash

    sphinx-build <path/to/python-microscopy/docs> <destination-directory>



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
| ``PYMEVis``             | This is for viewing point data sets. It shows a blank canvas when launched without any parameters.                   |
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

- Legacy developer installation docs are located at :ref:`installationFromSource`
- A step by step walkthough of installation using anaconda along with some troubleshooting tips can be found at :ref:`installationanaconda`


.. _pip:

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

.. _removing:

Removing a PYME install
=======================

To remove an executable installer on Windows 10, go to **Start Menu > Settings > Apps**, find `python-microscopy` under
**Apps & Features**, select it and press *Uninstall*. 

To remove an executable installer on Mac, delete the `python-microscopy` folder, either in Finder or via the Terminal.

For conda installations on Windows, Mac and Linux, removing the conda envrionment 
(i.e. ``conda remove --name pyme --all``, see the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment>`__
for additional help) is the preferred method to delete PYME. If you want to completely remove
any trace, you may also need to modify or remove `.condarc` and `.bash_profile`.


.. _apple_silicon:

Apple Silicon (M1) native
=========================

PYME will now build and run natively on apple silicon, and is significantly faster than a rosetta based installation. The installation process is, however, not particularly smooth
and should probably only be attempted by someone who is familiar with python. M1 installs require python >=3.8 and the use of conda-forge to find native versions of many
or our dependencies. 

These instructions are starting from an i386 (Rosetta) miniconda install. If starting from scratch it might be simpler to use
a miniforge install (https://github.com/conda-forge/miniforge)

#. Create a new, **empty**, conda environment:
    
    .. code-block:: bash
        
        conda create -n pyme_aarm64

#. Activate the new environment:

    .. code-block:: bash

        conda activate pyme_aarm64

#. Setup so that this environment pulls arm64 packages:

    .. code-block:: bash

        conda env config vars set CONDA_SUBDIR=osx-arm64
        conda deactivate pyme_aarm64
        conda activate pyme_aarm64

#. Install (base) dependencies. Note, this list is incomplete and additional dependencies will likely need to be installed to resolve ``ImportErrors`` in some functionality:

    .. code-block:: bash

        conda install -c conda-forge python=3.8 numpy scipy matplotlib pytables pyopengl jinja2 cython pip requests pyyaml psutil pandas scikit-image scikit-learn sphinx
        conda install -c conda-forge traits traitsui==7.1.0 pyface==7.1.0

#. build wxpython from source (the wxpython package on conda-forge is broken):

    **NOTE 1:** This has to be done in a native (not rosetta) terminal for the wx configuration to detect the architecture correctly. 
    
    
    **NOTE 2:** This may be machine specific, but autoconf doesn't distinguish between native and x64 libraries, and was trying to link to an x64 (rather than arm64) 
    copy of libtiff. I fixed this by hacking ``wxPython-4.1.1/buildtools/build_wxwidgets.py`` to add ``"--with-libtiff=builtin"`` to the ``configure_options``.

    .. code-block:: bash

        pip download wxpython
        tar -xzf wxPython-4.1.1.tar.gz
        cd wxPython-4.1.1
        conda install -c conda-forge graphviz
        python build.py dox
        python build.py etg
        python build.py sip
        python build.py build
        python setup.py install


#. change to base python-microscopy directory, find relevant python.app executable, and do a development install

    .. code-block:: bash

        cd python-microscopy
        which python
        /Users/david/opt/miniconda3/envs/pyme_as/python.app/Contents/MacOS/python setup.py develop
    
    (modifying as appropriate)

#. Try running ``dh5view -t``, ``PYMEVis`` etc ... 

#. chase down any additional dependencies (e.g. toposort, pyfftw, zeroconf)

**Extra - optimised numpy**
Build numpy from source, linking against Accelerate, vecLib (https://stackoverflow.com/questions/69848969/how-to-build-numpy-from-source-linked-to-apple-accelerate-framework)