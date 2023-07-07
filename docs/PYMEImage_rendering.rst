.. _pymeimage_rendering:


Rendering Images and Movies with PYMEImage
===========================================

PYMEImage facilities exporting image data, and rendering, two similar but distinct tasks.
Exporting images via `File > Save As` and `File > Export Cropped` will save the image data to disk,
while here we will cover exporting and saving rendered views of these data.

Sizes and overlays
------------------
Exporting will include the current view of the image as displayed in PYMEImage, including any overlays such as the scalebar or crosshairs.
These can be toggled on/off in the side-docking `Overlays` panel.

The size (number of pixels) of the exported image will vary with aspect ratio displayed. Currently, exporting an image which goes beyond
the visible bounds of your PYMEImage GUI will result in a large image being exported. 

Single Images
-------------
The current view in PYMEImage can be copied to clipboard via `View > Copy display to clipboard` and saved to disk via
`View > Save image as PNG`. 

Generating a movie from a 3D image or timeseries
-------------------------------------------------
PYMEImage does not directly export movies, however, can produce the individual frames of a movie which can then be combined into a movie using
a suitable tool such as `ffmpeg <https://ffmpeg.org/>`_. 
To save a series of PNG files, use `View > Save movie frames as PNG`. This will again export the current display, but will iterate through
each Z or T index of the displayed series.

