TIFF support in PYME
********************

Default TIFF support
====================

PYME supports the TIFF file format, with some peculiarities as described below. In general, we expect TIFF files to
use the `.tif` extension, with the `.tiff` extension being reserved for 'odd' TIFFs which won't load or save through
the default code paths. As a few 'normal' TIFFs use the `.tiff` extension and several 'odd' TIFFs use the `.tif`
extension we recognize that this behaviour is not ideal and this should be a target for refinement in the future.
At present manually changing the file extension is the best way of letting PYME know if the file is 'odd'
or 'normal'.

Loading TIFFs
-------------

By default (TIFFs with the `.tif` or `.lsm` extension), PYME uses a slightly modified version of Christoph Gohlke's excellent
**tifffile** module. This will read 90% of all biologically interesting TIFF formats and is fast and robust. The `.tif`
reader will extract basic OME metadata (pixel sizes and dimension ordering) if present. Unfortunately the **tifffile**
module can't read all TIFFs.

If reading fails, adding an 'f' to the file extension to make it `.tiff` will mean that the TIFF doesn't get caught by
our default TIFF handler, instead falling through to our handler of last resort - i.e. **bioformats**. This must be
set up independently of PYME (at least on Windows and Linux), and is somewhat painful to install. It has the advantage
of being able to load pretty much anything, but is not as stable or fast as we'd like.

Saving TIFFs
------------

PYME supports saving as floating point TIFF with minimal OME metadata, with the extension `.tif`, and with colour
and z channels interleaved according to the OME standard. Additional metadata is packaged within the OME
metadata using an XML annotation format. These floating point TIFFs are readable with ImageJ/FIJI, Matlab, and a few other
special purpose image processing programs. They are not readable by most general purpose software (word, powerpoint,
etc ..).

.. note:: **Why Floating Point?**

    PYME treats TIFF as a raw or intermediate image format which might be subjected to future quantitative image
    processing. As such we want to make sure that no information is lost during saving. If saving as the more widely
    readable 8 bit or 16 bit integer TIFF formats, we would need to quantize the data, resulting in loss of
    information. An additional problem with having to quantize the data is that the upper and lower bounds of
    the valid, e.g. 8 bit range would need to be determined and are unlikely to be the same across different images
    and image types. Saving as floating point removes the need to chose a scaling factor and any potential for loss
    due to quantization. **It is the only safe option for an intermediate data format**, despite lack of widespread
    software support.

Deprecated `.tiff` format saving
================================

In addition to the default `.tif` format, using the `.tiff` extension will force PYME to use legacy TIFF export
code that was written before we adopted the OME TIFF standard. The main differences to the standard `.tif` export
are:

    * Colour channels are saved in separate files
    * Metadata is saved in a stand-alone XML
    * There is no OME header

.. warning::

    **This format should not normally be used**. ImageJ will not be able to extract the voxel size or other metadata,
    and the chance of metadata getting separated from the individual files is high. The main use case for this form
    of export is for software that can't handle OME interleaved colour channels.


Deprecated TIFF series (.xml) saving
====================================

There is additionally support for saving image stacks as a folder containing series of individual TIFF files and
an XML metadata file. Like the `.tiff` legacy format, **this shouldn't normally be used**, but may be useful when
exporting data for exceptionally picky subsequent analysis programs.

Exporting images for use in reports and publications
====================================================

Saving as `.tif` is not the way you should export data for inclusion in a figure as the default PYME `.tif` format
is unreadable in most publishing programs. Instead you should export the image that is displayed (complete with
colourmaps, scaling etc ...). This is accomplished by choosing *View->Save image as PNG* from within an image view.

Alternatively you can copy the displayed image to the clipboard, either from the *View* menu, or using keyboard shortcuts.

* <cmd>-c (<ctrl>-c on windows/linux)  will copy the entire area of the currently displayed image to the clipboard
* <cmd><shift>-c (<ctrl><shift>-c on windows/linux) will copy only the region which is visible

