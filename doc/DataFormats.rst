.. _dataformats:

PYME Data Formats
*****************

PYME HDF5 (``.h5``)
===================

This is the default format for saving streamed image data, and is based on
`HDF5 <http://www.hdfgroup.org/HDF5/>`_, an open format for scientific data storage.
HDF5 is a very flexible format, and PYME defines a root level dataset called `ImageData`
containg the raw image data, stores metadata in a group called `MetaData`, containing a
number of nested groups, and optionally an additional dataset called `Events` which
details events which happened during the acquisition (eg focus changes and protocol tasks).

Running `h5ls` on an example file gives the following output. Note that the dimension order for ImageData is Z/T, X, Y:

.. code-block:: bash

    DB3:~ david$ h5ls -r /Users/david/PYMEData/david/2016_11_30/30_11_series_A.h5
    /                        Group
    /Events                  Dataset {3/Inf}
    /ImageData               Dataset {258/Inf, 1024, 256}
    /MetaData                Group
    /MetaData/Camera         Group
    /MetaData/Lasers         Group
    /MetaData/Lasers/l405    Group
    /MetaData/Lasers/l488    Group
    /MetaData/Positioning    Group
    /MetaData/Protocol       Group
    /MetaData/StackSettings  Group
    /MetaData/voxelsize      Group

.. topic:: Why HDF5?

   Whilst HDF5 is used extensively for scientific data in the areas of geophysics
   and astronomy, it is not currently particularly popular amoungst microscopists
   with the default microscopy format being tiff. In deciding to use HDF5, the more
   pertinent question might be **Why not tiff?** There are a number of quite
   compelling reasons not to use tiff:

   * Although TIFF is nominally a standardised format, very few (if any) programs
     support the full tiff standard, making writing portable tiffs a non-trivial
     proposition
   * Tiffs are limited in size to 2GB. Our raw data files are often ~ 6GB or more.
     In principle this can be circumvented by saving each frame as an individual
     file rather than in a multi-page tiff, but this runs into scalability issues
     well before the 2GB limit (at ~1000 frames on windows/NTFS) due to filesystem issues
     (file access becomes very slow due to the time taken to search through all
     the file nodes in the directory and the disk becomes very fragmented).
   * Support for metadata and other accompanying information such as events is poor,
     with the only real options being to write out an accompanying metadata file,
     or to bastardise some of the existing tags (ala ImageJ) both of which negate
     any portability advantages and invite data loss when copying/editing images.
   * Python support for TIFF leaves much to be desired, with the methods for
     writing multi-page tiffs being poor and clunky at best, as well as usually
     requiring the entire image sequence to be held in memory.

   By contrast, HDF5 offers:

   * A flexible, open, self describing format
   * Well supported in Python, ImageJ (with a plugin), Matlab, and IDL (although
     the IDL support is broken in some versions)
   * Unlimited file sizes
   * Transparent lossless compression (we get a factor of ~3 on image data)
   * High performance IO with atomic writes (ie if the acquisition program crashes
     the data taken up to the point of the crash will be safe)



HDF5 Results (``.h5r``)
=======================

This is the format in which analysis is stored. Like PYME H5 it is based on HDF5,
but rather than having an *ImageData* dataset, it has one called *FitResults* which
contains the fitted positions of all single molecule events. The *MetaData* and
*Events* are copied from the data file.

TIFF (``.tif``)
===============

PYME supports ``.tif`` as a format for saving individual images and stacks, but not
for spooling (see above). There is preliminary support for analysing data stored as
TIFF stacks.

PSF Files (``.psf``)
====================

``.psf`` files are the result of extracting a psf from bead data and are used both
for 3D fitting and deconvolution. They consist of a python pickle object containing
the PSF data as a numpy array and a voxelsize definition.

Shiftfield files (``.sf``)
==========================
``.sf`` files are saved vector shift fields used for correction of chromatic shift
in multi-colour imaging. Again, a python pickle.

Metadata (``.json``, ``.md``, ``.xml``)
=======================================

PYME supports metadata in a number of formats, for more details see :ref:`metadata`.

PYME Recipes (``.yaml``)
========================

These are used to store the details of processing pipelines used for either 
standard (e.g. confocal) data analysis or for postprocessing super-resolution
reconstructions.

PYME Compressed Images (``.pzf``)
=================================

These are a very minimal container for images compressed with our experimental high performance lossy compression
protocol. They consist of a minimal header followed by the compressed data and are mostly designed as 'wire' protocol
for data transfer to and within our cluster. It is also our on disk storage format within the cluster, and can be embedded
within HDF5 files (at the expense of loosing portability). For further documentation see :mod:`PYME.IO.PZFFormat`.

