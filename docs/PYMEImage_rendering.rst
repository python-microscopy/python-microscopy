.. _pymeimage_rendering:


Exporting Colour-mapped Images and Movies with PYMEImage
=========================================================

Exporting images via `File > Save As` and `File > Export Cropped` will save the image data to 
disk as raw data without any colour-mapping or visual manipulations, and is best option if you want
to perform any further processing on the data.
For visualisation tasks (e.g. incorporating in manuscript figures and presentations) it can be desirable
to export images or movies as they appear in PYMEImage with lookup-tables and scaling applied, as described
in this document.

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
Consult ffmpeg or other software for usage documentation, though note that PYMEImage will not necessarily export images
with an even width or height, which can sometimes require padding. 
An example ffmpeg command which will pad the image if necessary, find the numbered PNG files in the current directory, and output an mp4 video with 5 FPS playback is:
`ffmpeg -framerate 5 -i "imagefilename_%d.png" -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" imagefilename.mp4`
