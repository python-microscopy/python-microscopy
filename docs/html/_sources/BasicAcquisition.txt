Basic Acquisiton
****************

These instructions assume that you are imaging using near infrared dyes and the 671nm laser. While the same principles apply to imaging using other dye and laser combinations, there will be subtle differences. See :ref:`Protocols` and AdvancedImaging_.

Before starting
===============

Mount slides using the appropriate switching buffer. There are a few important points to consider:

- slides and coverslips should be thoroughly cleaned
- we can only access a small area of the slide, so the coverslip must be within this area. There is a template on the bench in the shared lab area.
- You should have a consistent slide naming scheme – I tend to use ``day_month_year_A``, ``_B``, etc...
- the nail polish must be dry before putting the sample on the microscope

Initial start up
================
- Turn on the following:
    + Computer, Pifoc, powerboard (stepper motors & cooling for 2nd camera), microscope body, filter wheel, arc lamp, laser, and laser shutters
    + Check that 2nd PC & Digidata are on (these should be on the whole time)
- log on
- fire up **'PYME Acquire'** and **'Launch Workers – 6 cores'** note that launch workers sometimes gives an error on startup (race condition). If this happens, close and restart (you can leave PYME Acquire running)

Initial configuration
=====================

By default, the system is set up for ratiometric imaging using the splitter, and any ROI is constrained to cover equal areas on both sides of the centre line. If you are only intending to image a single label, you should disable this feature by unchecking the ``Constrain ROI`` option under the ``Controls->Splitter`` menu.

If you are doing multi-colour imaging, make sure you have a current shift field. If not, acquire a new one using a beads slide (see the ShiftField_ section). Now might also be a good time to let the software know about the shiftfield using the ``Controls->Splitter->Set Shift Field`` menu item. This will mean that shifts get corrected in the live unmixing view. It will also mean *(NEW)* that the shift field data gets propagated to the analysis for you and you don't need to enter it every time.

It is useful to setup the acquisition protocol at this stage. For standard imaging with IR dyes this should be ``Prebleach671``. If you need to do something different, see :ref:`Protocols` or AdvancedImaging_.


Putting your sample on the scope
================================

This is simple – carefully put a drop of oil on the objective and place the slide into the holder. Remember that this is an inverted microscope and the coverslip needs to face downwards.

Setting up the slide information
--------------------------------

The important thing is to let the software know what slide you put on and which structures are stained using the sample information dialog. Bring up the dialog by pressing the **Set** button in the slide pane. This is where your carefully chosen naming scheme comes in, and you should also enter the structures you labelled and the dyes you used. Trying to keep a fairly consistent naming scheme for the dyes is important (as the dye information is used later in the analysis). We're currently using e.g. A680 or A750 for the Alexa dyes.

Finding cells
=============

Pretty much as for any normal microscope, with the following caveats:

- arc lamp / laser slider should be in (ie to arc lamp)
- you need to select both excitation and emission filters (see list on rack)
- you can only focus when PYME Acquire is in the foreground. Focus position is usually roughly in the centre of the travel. Focus sensitivity can be adjusted using the buttons on the focus wheel, and read out from the menu bar.

Finding cells in multi-colour samples
-------------------------------------

Note that there are a number of useful functions for multi-colour samples. Below the Preview window you can open a live unmixing preview window using the ``Controls->Splitter->Unxmix`` menu item (shortcut F7). Note that you should load the proper shiftfield as described in `Initial configuration`_ above. When the unmix window comes up first you might have to set the unmixing matrix values correctly. A suitable unmixing matrix for Alexa 680 and Alexa 750 is:

====  ====
0.89  0.11
0.13  0.87
====  ====

Preparing to acquire
====================

- Switch to the near-ir filter cube (#3) and excitation filter
- switch from eyepieces to camera
- if the signal is weak, switch to an EM gain of 150 (for old camera, ~100 for new camera)
- if still weak, increase integration time
- fine tune focus & position
- ensure neutral density filter is at the ND4.5 position
- close shutter to arc lamp, pull illumination slider to laser position, and open laser shutter
- fine tune again, and set ROI to cover illuminated area
- if doing a z-stack, focus and set top and bottom
- set the integration time to 50ms and the EMGain to 150 (assuming this doesn't saturate the detector)

Acquiring
=========

Before you start the acquisition make sure you have selected the appropriate acquisition protocol for your sample (usually ``Prebleach671``).

Click on **Series** or **Z-Series** for a single slice or a stack respectively,
wait for the analysis program to come up, test the threshold, and click go

Note that some protocols, including ``Prebleach671``, will perform a sanity check of acquisition parameters and will warn you when they think they have detected a problem. You should only override this warning if you know what you are doing.

Shutting down
=============

- Close the software (waiting for the camera(s) to warm up)
- turn off all the pieces of hardware you switched on during start up (this can be done while the software is shutting down)
- Do **not** turn off the computer without asking people (it gets used as a server to access recently acquired files
- Do not turn off the Digidata or 2nd computer
