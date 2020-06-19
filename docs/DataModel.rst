.. _datamodel:

PYME Data Model
***************

PYME fundamentally supports two types of data. `Image Data`_, and `Tabular Data`_. These are outlined below:

Image Data
==========

.. _imagestack:

The ImageStack object
---------------------

Our base image class is :class:`PYME.IO.image.ImageStack`. This wraps both image data and image metadata, and provides functions
for loading and saving images in a number of formats. The ``ImageStack`` object has two key attributes. ``ImageStack.data``
which provides access to the image data, and ``ImageStack.metadata`` [#mdh]_ which holds the metadata.

DataSources
-----------

The ``ImageStack.data`` attribute is an instance of a *DataSource*. These datasources implement lazy loading logic which
permits us to rapidly load data without needing to pull the entire contents of a file into memory. This is accomplished
by addressing the file on a frame by frame basis. Low level data source modules (which can be thought of as input drivers
or adapters) can be found in :mod:`PYME.IO.DataSources`. Each of these data sources implements 3 core methods:

``getSlice(ind)``, ``getNumSlices()``, and ``getSliceShape()``

End user code should not generally use these methods directly. Instead DataSources present a custom ``__getitem__`` [#getitem]_
which allows data sources to be sliced as though they were numpy arrays.  For example, returning the *nth*
slice from a 3D image could be accomplished as follows. ::

    image.data[:,:,n]

To extract a profile along z (or t) at a given (x,y) position ::

    image.data[x,y,:]

Or to extract a 3D sub-image ::

    image.data[100:120,230:250,10:30]

Slicing a ``DataSource`` returns a new numpy array built on the fly by concatenating elements obtained from repeated
calls to ``getSlice``. Because 2D slicing is performed before concatenation, this allows axial line profiles or ROIs to
be extracted without ever having the full image file in memory. *DataSources* also present a ``.shape`` attribute which
is very similar to the ``.shape`` attribute of numpy arrays, with the major difference that a number of empty dimensions
are appended to the end of the shape. It is perfectly OK to index a data source outside it's
true dimensionality. e.g. for a 3D data source ::

    image.data[100:120,230:250,10:30]

is the same as ::

    image.data[100:120,230:250,10:30, 0]

This extra-dimensional indexing is there to allow consistent handling of data regardless of the number of colour channels.
i.e. it will always be possible to access the 0th colour channel, even if the data is only a single channel. Similarly,
it is always possible to access the 0th slice along the 3rd dimension, even if the underlying data is only 2D.

.. warning::

    With the way colour channels were implemented in older versions of the code, it was not possible to slice the
    4th dimension (indexing was OK). You can now slice along the 4th dimension, but it is possible that some parts of
    the code have not caught up.

We currently use a 4D model, where the first and second dimensions are *x* and *y*, the third dimension is either *z* or
*t* and the 4th dimension is the colour channel.

.. note::

  The 4D data model means that there is currently no support for 3D time series, and that no distinction is made between
  time series and 3D stacks when processing. Up until this point, this has not been a major limitation, but it would be
  nicer if we had a consistent 5D data model. Transitioning to a 5D model is on the roadmap, but I have not yet decided
  if this will be a backwards compatible change.

.. _tabular_data:

Tabular Data
============

The second principle data type in PYME is tabular data. In it's loosest form this consists of columns which are accessible
by name (behaving a little like a dictionary). The most canonical form of tabular data is a class derived from
:class:`PYME.IO.tabular.TabularBase` [#inpFilt]_. In some parts of the code, however, you will find numpy record arrays,
pandas data frames, or even dictionaries standing in as tabular data.

The 4 key requirements for tabular data are:

* It should be indexable by column name like a dictionary
* Each column should be returned as a one-dimensional numpy array [#pandasviolation]_
* Each column should have the same length
* It should implement a ``.keys()`` function which returns a list of the column names [#recarrayviolation]_

If these requirements are met, tabular data can be processed by *filters* and *mappings* (defined in :py:mod:`PYME.IO.tabular`)
and a processing pipeline built up by cascading filters. One example of a tabular processing pipleine is the VisGUI
pipeline, :class:`PYME.LMVis.pipeline.Pipeline`.

.. note::

    At this point in time saving support is not baked into ``TabularBase``, and the most consistent / easiest way of
    saving tabular data is probably to call the ``.toDataFrame()`` method and then use pandas io functions. e.g. ::

        table.toDataFrame().to_csv('filename.csv')


.. rubric:: Footnotes

.. [#mdh] There is also accessible through a shortcut, ``ImageStack.mdh``, which is used in most existing code. New code
    should use the more descriptive ``ImageStack.metadata``.

.. [#getitem] Inherited from a common base class.

.. [#pandasviolation] This is not strictly true if using pandas data frames (indexing by column returns another data
    frame). In most cases these are sufficiently similar to numpy arrays that you can get away with it, but caution is
    advised. TODO: write a ``TabularBase`` derived wrapper for data frames.

.. [#inpFilt] This was previously ``PYME.LMVis.inpFilt``

.. [#recarrayviolation] numpy recarrays do not implement a ``keys()`` method and should normally be wrapped in an instance
   :class:`PYME.IO.tabular.recArrayInput`
