
Cameras / Detectors
===================
* Andor IXon
* Andor SDK3 Cameras (tested on Zyla and Neo)
* Hamamatsu DCAM cameras (tested on Orca Flash and Fusion BT)
* PCO Cameras (tested on pco.edge 4.2 LT)
* Thorlabs DCC1240 (maybe also DCC3240)
* IDS uEye industrial cameras (UI306x, UI327x, UI324x, probably others. Thorlabs-branded IDS)
* Ocean Optics spectrometers (probably suffering from bitrot)

Stages / Translation
====================
* PI piezos (using e255, e662, e709 and e816 controller interface boards, covers most PI piezos, likely easily modifyable to new controllers)
* PI piezo linear motor stages using the C867 controller (M686 stage and similar)
* PI stepper motor stages using the Mercury command set
* Marzhauser Tango translation stages and joystick
* Thorlabs MG17?? piezos (this was a 3-axis stage, somewhat dated now)

Lasers / Light Sources
======================
* Coherent OBIS lasers
* MPB Lasers
* Toptica iBeam lasers
* Cobolt lasers
* Matchbox lasers
* Omicron Phoxx lasers
* Prior Lumen arclamp
* Oriel Cornerstone arclamp and monochrometer

Filters / Shutters
==================
* AA Optoelectronics AOTF
* Simple voltage controlled AOTF (via Arduino)
* Thorlabs FW102B filter wheels

Miscellaneous
=============
* TI LightCrafter DMD
* Thorlabs PM100USB power meter
* Arduino-based analog, & digital IO
* Nikon TE2000 stand
* Nikon Ti stand
* 3D Connexion "Space Navigator" 3D mouse

.. note::

    **Hardware support using other packages:** 
    * [Oxford Micron's microscope](https://pypi.org/project/microscope/) lasers 
    and filterwheels 
