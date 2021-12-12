.. _configuringpymeacquire:

Configuring PYME Acquire
************************

Specifying your Hardware
========================

The hardware in PYME is configured and initialised in a customised initialisation script. These typically start with ``init`` and live in ``PYME/Acquire/Scripts``. You can tell PYME Acquire which script to use with the ``--init-script`` (short form ``-i``) command line option eg:

.. code-block:: bash

  python PYMEAquire.py -i init_script.py

If you don't specify a script, it'll use ``init.py`` by default, which contains a setup for the simulator.

The easiest way to get your hardware up and running is to copy and modify one of the existing scripts. There is a little bit of magic going on in places, so I'll attempt to explain the relevant bits.

The ``scope`` Object
--------------------

``scope`` is an object representing the microscope. It serves a a place to accumulate all the various hardware bits as well as being home to a few utility functions. By the end of the script, scope must have the following properties defined:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

==================   ============================================================================================================
``scope.cam``        The camera object. (``scope.cameras`` is a dictionary of cameras used for multiple camera support, but can
                     safely be ignored unless you really need to drive two cameras on one rig)
``scope.chaninfo``   Information about colour channels. This is mostly left over cruft from previous versions of the software
                     and specifies how to deal with colour (bayer mask) cameras, and the sequential acquisition of different
                     colour channels when using shuttered laser excitation sources, a multiband dichroic/blocking filter, and a
                     black & white camera. Just copy across from one of the existing scripts.
``scope.shutters``   Cruft again, associated with chaninfo above
==================   ============================================================================================================

In addition to the mandatory items above, there are a couple of optional items that will be recognised and used if present:

==================   ============================================================================================================
``scope.lasers``     This should be a list of laser objects, and is where most of the shuttering is now controlled
``scope.joystick``   A joystick object.
==================   ============================================================================================================

Positioning - ``scope.piezos``
------------------------------

``scope.piezos`` is a list to which positioning devices which export the piezo interface (see the scripts in ``PYME\Acquire\Hardware\Piezos``) should be added. Note that the piezo bit is historic, and this should probably be called ``scope.positioning`` (or similar) - there's nothing to say that they can't actually be stepper motors. The entries in this list should be tuples of the form ``(positioningObject, channelNum, displayName)``.

``InitBG`` and ``InitGUI``
--------------------------

This is where things start to get a bit complicated. To improve startup times, PYME Acquire supports a threaded initialisation, such that different hardware components can be initialised in parallel. This is what all the ``InitBG`` blocks are.

The ``InitBG`` function takes a name (to display on the splash screen) and a string containing the code to execute, and fires off a new thread to execute the code. It returns the thread created, so, if you have one bit of hardware which requires another to be initialised first, you can use the join function to force a wait. This is purely a performance tweak, so you can quite happily just put the code in without the ``InitBG`` wrappers.

The ``InitGUI`` blocks serve a different function - the main PYME Acquire GUI is not properly created when the init script is run, and any GUI operations associated with the various bits of hardware need to be deferred to such time as the main window is there. ``InitGUI`` blocks add the code contained to a list of things to be executed when the main window is ready.

PYMESettings.db
===============

PYME Acquire stores a lot of it's settings in ``PYME/Acquire/PYMESettings.db``. This is an sqllite database and will be created the first time PYME Acquire is run. It should then have it's permissions changed so that all users who are going to be using the software can write to it.

Calibration
===========

CCD Pixel Size
--------------

PYME stores it's pixel sizes in a two step process - first there is a named list of
pixel size settings, and then an index to the setting that is currently active.
This is to facilitate the easy changing of cameras / objectives etc. To set the
pixel size you thus have to create a new setting, and then make that active.

This can be done by selecting **Controls > Camera > Set Pixel Size** from the menu.

Alternatively one can execute the following commands in the console:

.. code-block:: python

  scope.AddVoxelSizeSetting(name, x_size, y_size)
  scope.SetVoxelSize(name)

where ``x_size`` and ``y_size`` are the x and y pixel sizes **in the sample** in um.

Camera Noise Properties
-----------------------

The analysis software wants to know about the camera noise properties, which can be obtained from the performance sheet shipped with the camera. Noise characteristics
are stored in a database, keyed by camera serial number. To add the noise characteristics for you camera(s), add a .yaml file to the ``~/.PYME/cameras/`` directory (or
te corresponding install or site-directory for multi-user installs - see :py:mod:`PYME.config`). The exact name of the file is your choice - all .yaml files in the ``.PYME/cameras``
directory will be read and ammalgamated. The exact format of an entry differs slightly between camerase (see examples below), but follows the basic pattern of a top-level dictionary
keyed on serial number, with each entry having a ``noise_properties`` entry which is in turn a dictionary keyed by gain mode. See also :py:mod:`PYME.Acquire.Hardware.camera_noise`


.. code-block:: yaml
    
    # An Andor Zyla entry
    VSC-00954:
        noise_properties:
            12-bit (high well capacity):
                ADOffset: 100
                ElectronsPerCount: 6.97
                ReadNoise: 5.96
                SaturationThreshold: 2047
            12-bit (low noise):
                ADOffset: 100
                ElectronsPerCount: 0.28
                ReadNoise: 1.1
                SaturationThreshold: 2047
            16-bit (low noise & high well capacity):
                ADOffset: 100
                ElectronsPerCount: 0.5
                ReadNoise: 1.33
                SaturationThreshold: 65535

    # An Andor IXon entry:
    5414:
        default_preamp_gain: 0
        noise_properties:
            Preamp Gain 0:
                ADOffset: 413
                DefaultEMGain: 90
                ElectronsPerCount: 25.24
                NGainStages: 536
                ReadNoise: 61.33
                SaturationThreshold: 16383

    # A HamamatsuORCA entry:
    '100233':
        noise_properties:
            fixed:
                ADOffset: 100
                DefaultEMGain: 1
                ElectronsPerCount: 0.47
                NGainStages: 0
                ReadNoise: 1.65
                SaturationThreshold: 65535



EMCCD Gain
----------

The old Andor EMCCD cameras use a method of setting the gain with is non-linear, and uncalibrated (basically you just set a value between 0 and 255 which is sent through a D to A convertor and used to control the gain register voltage). This needs to be calibrated if we want to know what our actual EM gain is. More recent Andor cameras give you 4 different ways of setting the gain, some of which are linearised / calibrated. PYME uses the default mode, which is similar to that of the older cameras (with some differences in scaling), and does it's own calibration for these as well. The Steps for doing this are outlined below:

1. Set up a uniform illumination using transmitted light (a uniform fluorescent field can also be used as long as there is NO bleaching & the illumination source is stable). If there are residual non-uniformities, a region of interest can be selected. If using a ROI it shouldn't be too small.

2. Wait for the CCD temperature to settle

3. Decide what range of gain values you want to calibrate over (the default is 0 to 220, but this might be too much for newer cameras - I'd recommend 0 to 150 for these). Set the illumination intensity and/or integration time such that the maximum brightness in the image is at ~50% of saturation when using the highest gain you want to calibrate for. Note that this WILL saturate the display (the display saturates at  4096 counts, the camera at ~16000). Use the histogram instead  -  you want the upper bound somewhere between 8000 & 12000.

4. In the console window, execute the following commands:

.. code-block:: python

  from PYME.Acquire.Hardware import ccdCalibrator
  ccdCalibrator.ccdCalibrator()

 or (if you want to calibrate over a range other than 0 to 220):
 
.. code-block:: python

  import numpy
  from PYME.Acquire.Hardware import ccdCalibrator
  ccdCalibrator.ccdCalibrator(numpy.arange(0, <max_gain>, 5))


