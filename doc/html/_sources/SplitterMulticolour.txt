.. _imagesplitter:

Using an image splitting device for multi-colour ratiometric imaging
********************************************************************

Calibrating the image splitting device
======================================

These instructions are designed for use with an image splitting device and a single CCD camera. They should be able to be applied to a multi-camera scenario if the images are stiched together first. The code was originally written for a home-built splitter where the CCD was split vertically and the two halves were mirror image views. It has since been modified to work with more generic splitters, although these modifications are not yet well tested (we're currently in the process of assembling a new splitter which split's horizontally and doesn't flip, so any bugs are likely to be ironed out over the course of the next couple of weeks).

1. Prepare a medium density bead slide (minimum separation between beads should be on the order of ~ 500 nm, more is OK). The density should be low enough that beads in both channels can be unambiguously assigned (i.e. there should only be one bead in any given 15x15 pixel ROI), a small number of clusters is permissible as these will be discarded in post-processing. A suitable density usually works out at ~20 beads in the field of view (our FOV is half the camera).
2. Because the density used above is relatively low, a single image will not give us a particularly good estimate of the vector shift field between the two channels. To enable a better coverage of the field of view we take multiple shifted images of the beads, achieving better coverage through post-processing. If using PYMEAcquire for data acquisition on a microscope with a motorised stage there is a protocol (called 'shiftfield') which will do this for you. If using 3rd party software, or using a microscope without a motorised stage you can simply move the stage manually whilst recording a sequence of images (the analysed data is filtered for bead width and blurred beads will be discarded – I find a series of short moves with brief pauses to be effective).
3. Make sure that the distributed data analysis platform (``launchWorkers``) is running.
4. Open the data using ``dh5view`` (if using PYMEAcquire it should open automatically). 
5. From the *'Set defaults for'* menu chose *'Calibrating the splitter'*. This sets the fit model to one in which the separation between red and green images of the same bead is a free parameter. It also turns off temporal background subtraction and increases the detection threshold. 
6. **[Non-PYME data only]** If the data was acquired in 3rd party software you will also need to set a number of metadata parameters to tell the software about how the splitting is carried out (these can be entered from the console within ``dh5view``, but it might make more sense to put them in the .md file used to get the data to load – see the loading external data bit of the PYME documentation). The relevant parameters are ``Splitter.Channel0ROI``, ``Splitter.Channel1ROI``, and ``Splitter.Flip``. The ROI parameters take a list of values in the form ``[x0, y0, width, height]``. The Flip parameter is either ``True`` or ``False``. It is important that the width and height is the same for both ROIs. Eg (if entered in the .md file – if executed at the console, replace ``md`` with ``image.mdh``)

::

  md['Splitter.Channel0ROI'] = [0, 0, 512, 256]
  md['Splitter.Channel1ROI'] = [0, 256, 512, 256]
  md['Splitter.Flip'] = False

7. Click on *'Test'* to see if the detection threshold is suitable – if necessary try a higher or lower threshold.
8. Once happy with the threshold, click *'Go'*. This will send the frames into the distributed analysis system which should churn through and perform the fits [#]_. 
9. One all the analysis tasks are complete, go to the analysis folder (if you haven't set the ``PYMEDataDir`` environment varible it should be under ``c:\Users\<username>\PYMEData\Analysis\<name of folder containing raw data>``) and find the ``.h5r`` file corresponding to the raw data. Open this in ``VisGUI``.
10. Check the data in ``VisGUI`` to see if it looks reasonable - good coverage of the field of view, reasonable looking distribution of shifts if you set the point colour to be ``FitResults_dx`` or ``FitResults_dy`` (the x and y shifts). Try adjusting the filter if this is not the case (a good place to start is sigma – the PSF std deviation, which can be set to a reasonably narrow window around the expected bead width). A few erroneous vectors are still permissible as these will be filtered out in subsequent steps.
11. From the *'Extras'* menu choose *'Calculate shiftmap'*. This will attempt to interpolate the shift vectors obtained at the bead locations across the field of view. The algorithm first checks to see if each vector points in approximately the same direction as it's neighbours. 'Wonky' vectors which dramatically differ from their neighbours but have somehow made if through prior filtering steps are discarded at this point. Bivariate smoothing splines are then fitted to the x and y shift vectors. The resulting interpolated shift field (and residuals) is shown, and the user given the opportunity to save the shiftmap (effectively the spline coefficients) in a .sf file. Unfortunately the 'save' dialog is modal and you don't get a chance to examine the shift field before being prompted to save. I usually cancel the save request the first time, examine the result, and if happy, run *'Calculate shiftmap'* again, saving the result. This interpolated shift field should be smooth, although it's common to see magnification differences as well as rotation in the field resulting in a spiral or vortex like appearance. If you're unhappy with the generated shiftmap, you can go back to the filter (or if really bad try acquiring and analysing a new data set).
  **New**: If the above does not yield a good shiftmap (shifts should be mostly translation, rotation, and some scaling, which results in smoothly varying shiftmaps) you can also try the *'Calculate Shiftmap (Model Based)'* option (also on the *'Extras'* menu) which fits the coefficients of a global affine transform rather than trying to interpolate shifts. The resulting shiftmap will be less flexible than one calculated using the *'Calculate shiftmap'* function, but captures the most likely transformations and is better behaved (particularly at the corners of the field where errors can be common).

Large shifts
------------

If you have very large shifts, you might need to increase the size of the ROI used to fit each bead when performing the callibration. This can be achieved by overriding the ``ROISize`` parameter in the analysis - e.g. by entering: 
::

  image.mdh['Analysis.ROISize'] = 10

in the ``dh5view`` console. The ``ROISize`` setting is the size of a 'half ROI', with the size of the actual ROI being :math:`2n + 1`. The default for shift estimation is 7 (15x15), and the default for fitting is 5 (11x11).
  

Analysing ratiometric images
============================

Whilst not as complicated as the calibration procedure, the analysis procedure for multi-colour images is also a little different to that for single colour images.

1. Load the data in dh5view
2. Choose ``SplitterFitFNR`` as the fit module (**Note:** this assumes that the default temporal background subtraction has done it's thing and doesn't fit the background at all. If you have disabled background subtraction, try using the older ``SplitterFitFR``) [#]_.
3. Set the Splitter parameters as in 6 above
4. Set the shift field to the .sf file saved in the calibration step (click on the 'Set' button)
5. Test the threshold
6. Click 'Go' to start the analysis. 

.. note:: The shifts are corrected as part of the fitting process (they should be absent from the fitted data)

Visualising ratiometric data
============================

If you have analysed data using one of the *'SplitterFit...'* modules, VisGUI will show a colour tab with a scattergram of the ratios. Before you can render the images as multi-colour, you will need to add species to this scattergram by clicking add and setting the ratio (which can then be tweaked by clicking on the ratio value in the table). You can also try and automagically guess what components are present by using the 'Guess' button. The points assigned to a certain ratio will be given the same colour as that component. After the ratios have been defined, you will have new selections in the colour filter selector in the main window, and rendering options will default to producing multi-channel images. 

.. note:: It is also possible to define species and ratios in the metadata, but that is beyond the scope of what we can go into here.


.. [#] Extra for experts:  in this case it will probably only make use of 1 or 2 cores as the distributed analysis uses a chunk size of at least 50 frames to allow the data to be cached for efficient background subtraction on the workers when analysing binking datasets.
.. [#] ``SplitterFitFNR`` is a new routine and is preferred as it performs shift-corrected ROI extraction and thus works for larger shift values. The previous versions only worked if the shift was sufficently small that the ROI co-ordinates for the 1st channel could also be used to extract a ROI for the second channel which completely enclosed the 2nd image of a molecule. As well as coping with almost arbitrarily large shifts, the new routine allows a smaller ROI to be used for moderate shifts, improving speed and tolerable localisation density. 