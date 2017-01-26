.. _modifyingpymeacquire:

Modifying PYMEAcquire and writing hardware drivers
**************************************************

Initial setup
=============

- Download and install a *development* build following the instructions at
  http://python-microscopy.org/doc/Installation/InstallationFromSource.html

- Run PYMEAcquire (either from the console, or by making a shortcut to it). When run without parameters, it uses
  `PYME/Acquire/Scripts/init.py` as a startup script and should run with a simulated camera. Copy and rename this script
  keeping it in the `Acquire/Scripts` directory (this will be your new hardware config script, and you can slowly edit
  it to use real hardware rather than the simulated bits) [NB - support for init scripts in other directories is on our
  wishlist]. You can specify which initialization script to use with the `-i` command line option to `PYMEAcquire`

Making a camera driver
======================

This can be a little involved, and is not as 'nice' as it could be due to how the camera support evolved (we started out
with a driver for the PCO Sensicam and then made other camera drivers look like this one).

- Copy PYME/Acquire/Hardware/CameraSkeleton.py. This gives stubs for the most important functions a camera class should
  provide, and some minimal (poor) documentation of the camera interface. Making this cleaner and making all cameras
  inherit from a common base class is fairly high on our wishlist, but it's hard to know when that will happen.

- Modify the camera skeleton, filling in implementation details for each of the functions. Also look at the implemented
  camera drivers, particularly for similar classes of camera (e.g. sCMOS).

- change your `init_XXX.py` script to initialize your new camera class instead of the fake camera. Make sure to both
  assign to `scope.cam` and add to `scope.cameras`. [NOTE: this is a horrible way of initializing things, and should
  probably be replaced by a `scope.add_camera(...)` function call or similar at some point in the future].

Making a driver for a stage / piezo
===================================

Copy one of the existing piezo drivers (in `PYME/Acquire/Hardware/Piezos`) and modifiy the method implementation whilst
keeping the signatures the same. Should really be re-factored to use a common Piezo/Positioning base class. Both
piezos and motorized stages use the same interface.

Making a driver for a laser
===========================

This one actually has a base class! Copy one of the existing laser drivers and modify accordingly.

Other plugins / general comments
================================

Other hardware doesn't currently have a defined interface, so there is a fair bit of flexibility about how something can
be implemented. See filter wheel code for an example.

In general we are moving from an imperative (we tell the piezo to move to a position x) to a state based (we tell the
piezo that it should be at position x) way of talking to hardware. This is to aid future automation (we can save and
restore the 'state' of the microscope / tell the microscope to assume a certain state, rather than making a number of
imperative commands to each of the hardware components). This is still a work in progress, but new hardware should
ideally support this way of doing things - see :mod:`PYME.Acquire.microscope`, and in particular the
:class:`PYME.Acquire.microscope.StateHandler` code for more info.

Timing
======

Timing is currently 'lazy' by default  - i.e. there is not tight synchronization between hardware movements and the
camera, with the preferred approach being to generate and save a timestamped event (:mod:`PYME.Acquire.eventLog` -
although this still needs documentation) for any hardware motion and compare this to frame timestamps in post-processing
(currently inferred from a start time and frame rate, but ideally provided directly from the camera in the future - both
Andor and Hamamatsu cameras support hardware timestamps, although these are not supported in our current software).
The rationale behind the asynchronous acquisition is to allow the maximum frame rate possible. That said, nothing about
our design prevents you from slaving other hardware off the camera trigger, and triggering the camera externally should
be possible with minor modifications to the camera drivers.

Using the state based control, it is also possible to force camera synchronization at a software level, albeit at a
significant loss of speed (synchronized operations involve stopping and restarting the camera).

