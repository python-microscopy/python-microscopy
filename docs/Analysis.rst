**PYMEImage (dh5view)** - Image Data Analysis and Viewing
*********************************************************

DSView is a general purpose data viewer that supports all the :ref:`DataFormats`
and additionally the ``QUEUE`` objects used in real-time data analysis. DSView may
either be invoked with the command::

   dh5view [options] filename

or launched programatically by passing a *numpy* array or PYME *datasource* to
``PYME.DSView.View3D(...)``.

Modules
=======

DSView has a number of modules for performing various analysis tasks,
including :ref:`Localisation Analysis <localisationanalysis>`, PSF Extraction, Deconvolution, and Tiling.

Modes
=====

Modes represent personalities suitable for use with certain types of data
and determine which modules are loaded on startup. The mode used is determined
by filename, although modules which are not loaded by default can be inserted
later from the **Modules** menu.

The modes are:

=======  ======================  =================================
Name     File types              Description
=======  ======================  =================================
lite     programatic invocation  no modules loaded
LM       ``.h5``, ``QUEUE``      localisation analysis
blob     ``.tif``                blob analysis and PSF extraction
default  aything else            PSF extraction
=======  ======================  =================================

A particular mode can be forced by calling ``dh5view`` with the ``-m`` option, eg::

    dh5view -m LM filename



