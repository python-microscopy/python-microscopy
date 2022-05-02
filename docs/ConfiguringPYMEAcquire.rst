.. _configuringpymeacquire:

Configuring PYME Acquire
************************

Configuring PYME Acquire for your microscope is relatively straightforward and involves minimal programming (1 file!) if we already support your hardware components.
No PYME code needs to be altered to run your microscope, however you will likely find it helpful to use a development installation of PYME (see :ref:`installation`),
in order to find the hardware classes you need to specify.


Specifying your Hardware
========================

The hardware in PYME is configured and initialised in a customised initialisation script, or `init script` for short. These typically start with ``init`` and several examples can be found in ``PYME/Acquire/Scripts``. 
You can tell PYME Acquire which script to use with the ``--init-script`` (short form ``-i``) command line option eg:

.. code-block:: bash

  python PYMEAquire.py -i init_script.py

PYME will look in ``PYME/Acquire/Scripts`` if you specify a relative path, however you may find it convenient to keep your init script elsewhere so you can easily back it up or keep it in version control separate of PYME.
If you don't specify a script, PYME will use ``init.py`` by default, which contains a setup for the simulator.

init script structure
---------------------
The easiest way to get your hardware up and running is to copy and modify one of the existing scripts. 
There you will see functions with two decorators, ``init_hardware`` and ``init_gui``. 
As their names suggest, these correspond to functions which either initialize a hardware component, or set up a GUI panel. Each ``init_hardware`` function runs in it's own background thread, effectively parallelising the hardware initialisation phase and improving startup time. To wait for all the hardware initialisation tasks to complete, we add a ``joinBGInit()`` call at the end of the init script. ``init_gui`` functions, in contrast are defferred until all the hardware is initialised (the init script completes) **and** the main PYMEAcquire GUI has been created.

For example, the blocks to initialize an AOTF-shuttered laser, and its GUI controls could look like:

.. code-block:: python

    @init_hardware('Lasers & Shutters')
    def lasers(scope):
        from PYME.Acquire.Hardware.Coherent import OBIS
        from PYME.Acquire.Hardware.AAOptoelectronics.MDS import AAOptoMDS
        from PYME.Acquire.Hardware.aotf import AOTFControlledLaser
        from PYME.config import config
        import json

        calib_file = config['aotf-calibration-file']
        with open(calib_file, 'r') as f:
            aotf_calibration = json.load(f)

        scope.aotf = AAOptoMDS(aotf_calibration, 'COM14', 'AAOptoMDS', n_chans=4)
        scope.CleanupFunctions.append(scope.aotf.Close)

        l405 = OBIS.CoherentOBISLaser('COM10', name='OBIS405', turn_on=False)
        scope.CleanupFunctions.append(l405.Close)
        scope.l405 = AOTFControlledLaser(l405, scope.aotf, 0)
        scope.l405.register(scope)
    
    @init_gui('Laser controls')
    def laser_controls(MainFrame, scope):
        from PYME.Acquire.ui import lasersliders

        lsf = lasersliders.LaserToggles(MainFrame.toolPanel, scope.state)
        MainFrame.time1.WantNotification.append(lsf.update)
        MainFrame.camPanels.append((lsf, 'Laser Powers'))
    

Note that any code can be imported during these functions, which makes it easy to add your own hardware or GUI panels.

The arguments for these functions, ``MainFrame`` and ``scope`` are the main PYME GUI and the microscope instance it interfaces with, respectively.



The ``scope`` Object
--------------------

``scope`` is an object representing the microscope. It serves as a place to accumulate all the various hardware bits as well as being home to a few utility functions.
The ``scope`` variable will be accesible from the PYME Acquire shell. The typical process is to initialize a component, such as your camera, and then call it's registration method if available.
If there is no registration method, you can still add a component to the ``scope`` object as an attribute so you can access it later. 

Important registration methods include:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

===========================         ============================================================================================================
``scope.register_camera``           see PYME.Acquire.microscope.register_camera for details. 
                                    This method allows configuration of camera orientation with respect to stages such that mouse center-click and drag moves the field of view in the correct direction.
``scope.register_piezo``            see PYME.Acquire.microscope.register_piezo for details. 
                                    This method configures each stage axis, base unit, and postive/negative directions
``Laser.register``                  Lasers, and other hardware, can implement their own registration methods. 
                                    These can be used to set up state handlers such that e.g. laser powers and shutters can be easily changed.
                                    See PYME.Acquire.Hardware.lasers.Laser.register for an example.
===========================         ============================================================================================================

These registration methods make it possible to query or set the microscope state very conveniently. 
The ``scope.state`` property will return a dictionary describing the state of the microscope (stage positions, camera frame rate, laser powers, etc.).
What makes this powerful for hardware control is the state can also be set. 
For example ``scope.state.update({'Lasers.l640.On': True, 'Positioning.x': 30, 'Positioning.y': 30})``
would reposition the scope and make sure the laser named `l640` is turned on/unshuttered. 


Key attributes which which will be set up include:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

==================   ============================================================================================================
``scope.cam``        The active camera object. ``scope.cameras`` is a dictionary of cameras used for multiple camera support, but can
                     safely be ignored unless you really need to drive two cameras on one rig)
``scope.lasers``     A list of laser objects, which is also where most of the shuttering is now controlled
``scope.piezos``     A list of positioning devices (piezos, stepper stages, etc.) which use the piezo interface (see ``PYME\Acquire\Hardware\Piezos``).
                     The entries in this list are tuples of the form ``(positioningObject, channelNum, displayName)``.
==================   ============================================================================================================


Settings and Camera Calibrations 
================================

PYME Acquire stores a lot of it's settings in ``PYME/Acquire/PYMESettings.db``. This is an sqllite database and will be created the first time PYME Acquire is run.
It should then have it's permissions changed so that all users who are going to be using the software can write to it.

Several calibrations are either strongly encouraged or effectively required.

Pixel Size
--------------

PYME stores it's pixel sizes in a two step process - first there is a named list of
pixel size settings, and then an index to the setting that is currently active.
This is to facilitate the easy changing of cameras / objectives etc. To set the
pixel size you have to create a new setting, and then make that active.

This can be done by selecting **Controls > Camera > Set Pixel Size** from the PYME Acquire menu.

Alternatively one can execute the following commands in the console:

.. code-block:: python

  scope.AddVoxelSizeSetting(name, x_size, y_size)
  scope.SetVoxelSize(name)

where ``x_size`` and ``y_size`` are the x and y pixel sizes **in the sample** in um.

Camera Noise Properties
-----------------------

The analysis software wants to know about the camera noise properties, which can often be obtained from the performance sheet shipped with the camera. Noise characteristics
are stored in a database, keyed by camera serial number. To add the noise characteristics for you camera(s), add a .yaml file to the ``~/.PYME/cameras/`` directory (or
the corresponding install or site-directory for multi-user installs - see :py:mod:`PYME.config`). The exact name of the file is your choice - all .yaml files in the ``.PYME/cameras``
directory will be read and ammalgamated. The exact format of an entry differs slightly between cameras (see examples below), but follows the basic pattern of a top-level dictionary
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

These entries will propagate into the metadata of acquired images (see :ref:`metadata`). Some values can be summaries of per-pixel quantities if using camera maps
as described in the following table:

.. tabularcolumns:: |p{4.5cm}|p{11cm}|

==================   ============================================================================================================
ADOffset             Analog-digital offset, in analog-digital units (ADU). May be specified by the camera data sheet. 
                     Can be calibrated using ``PYME\Analysis\gen_sCMOS_maps.py``, and then taking the median dark-map value to be the ADOffset.
ReadNoise            Gaussian amplifier noise, as a standard deviation in units of photoelectrons. May be specified by the camera data sheet. 
                     Can be calibrated using ``PYME\Analysis\gen_sCMOS_maps.py``, and then taking the square-root of the median variance-map value.
ElectronsPerCount    Conversion between ADU and photoelectrons (units of [e-/ADU]). May be specified by the camera data sheet.
==================   ============================================================================================================


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


