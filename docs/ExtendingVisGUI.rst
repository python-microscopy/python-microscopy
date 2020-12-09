.. _extendingvisgui:

Extending PYMEVisualise (VisGUI)
********************************


The VisGUI pipeline
===================

VisGUI utilises a pipeline to process incoming point position data prior to visualization.  It is responsible for filtering to reject bad fits as well as performing other
operations such as re-mapping data, munging column names into our desired format, clumping grouped localizations, and
applying various data corrections and calibrations. The pipeline consists of 3 key parts: A file import adapter, a variable
section, and a fixed section. Each section consists of a number of cascaded classes which each implement the
:ref:`tabular_data` model (see also `Tabular filters`_ and :mod:`PYME.IO.tabular`). As a general rule, these classes are written
to lazily access the data and do not cache results. They are typically 'look through' for any variables which aren't
altered. The :class:`pipeline object <PYME.LMVis.pipeline.Pipeline>` itself also implements the tabular interface
and exposes the output of the fixed section.

.. figure:: images/pipeline_new.png

    The VisGUI pipeline showing a hypothetical configuration of the recipe section which might be used for clumping
    repeated observations of molecules.


The input adapter
-----------------

This is specific to the file format being loaded, and typically consists of in input filter which loads data into tabular
form, and a mapping filter which adapts column names and scaling to fit VisGUIs standard requirements
(see `VisGUI column names`_).

The variable section
--------------------

This is implemented as a :ref:`PYME recipe <recipes>` (see also :ref:`writingrecipemodules`). It takes the output of the
input adapter, and performs any additional manipulation
that might be sample, microscope, or dataset specific. This should include tasks such as event clumping and any
calibrations or corrections. The recipes *namespace* replaces the previous ``.dataSources`` attribute of
the pipeline.

.. warning::

    The variable section is **very new**, and as such there will still be bugs. Notably a lot of stuff which should be
    in the variable section is currently accomplished by hard to follow circular code paths within the fixed section.

The fixed section
-----------------

This is responsible for filtering events prior to display or rendering, and for selecting which colour channels to
display. It can use the output of any recipe block in the variable section as it's input, but will usually point to the
tail (or last output) of the variable section. What the fixed section uses for input can be set by calling
:meth:`PYME.LMVis.pipeline.Pipeline.selectDataSource`. [#recipenames]_

.. note::

    Those who are familiar with the old configuration of the pipeline will recognize this as being the bulk of the old
    pipeline. At present, the fixed section still contains a :class:`~PYME.IO.tabular.mappingFilter`, this is still
    available as ``pipeline.mapping`` and much of the functionality still revolves around manipulating this, on
    inserting new data sources [#datasourceinsertion]_, and on a circular flow [#circularflow]_. Moving forward, much of
    this logic should be moved into the variable section, and the flow linearized. **The use of the ``.mapping`` attribute
    of the pipeline is deprecated.**

Tabular filters
===============

Tabular filters are classes which take tabular data as an input and themselves expose the tabular interface. They are
used extensively for the manipulation of data within the pipeline, and are generally look-through and lazily evaluated
on column access (This reduces the memory footprint and latency of maintaining a large pipeline at the expense of some
computation when results need to be recomputed. Computationally intensive tabular filters will often cache results, but
usually on a column by column basis with lazy computation on first access.)

The two archetypal tabular filters are :class:`PYME.IO.tabular.resultsFilter` and :class:`PYME.IO.tabular.mappingFilter`
which implement filtering and re-mapping or data respectively. The mapping filter in particular permits new columns to
be derived from a functional manipulation of existing columns.

Writing VisGUI Plugins
======================

Writing VisGUI plugins is very similar to :ref:`writing plugins for dsviewer <extendingdsviewer>`. A plugin is a python
file which implements a ``Plug(visfr)`` method. In contrast to dsviewer plugins, the ``Plug()`` usually takes a
:class:`PYME.LMVis.VisGUI.VisGUIFrame` instance as it's argument [#visfrindsviewer]_. Like dsviewer plugins, the
``visfr`` object exposes a :meth:`.AddMenuItem() <PYME.ui.AUIFrame.AUIFrame.AddMenuItem>` method. Unlike dsviewer
plugins, ``visfr`` will have a ``.pipeline`` attribute which is an instance of the current
:class:`pipeline object <PYME.LMVis.pipeline.Pipeline>`.

The other main difference to dsviewer plugins is the location where plugins will be discovered. VisGUI plugins will be
automatically found in ``PYME.LMVis.Extras`` or ``PYMEnf.LMVis.Extras`` [#pymenf]_.

.. note::

    A more flexible method for discovering VisGUI plugins is on the TODO list.


Plugins which use the output of the pipeline
--------------------------------------------

These are plugins which use the output of the pipeline, but don't modify the pipeline itself. Examples are:
:mod:`PYME.LMVis.Extras.photophysics`, :mod:`PYME.LMVis.Extras.vibration`, and :mod:`PYME.LMVis.Extras.shiftmapGenerator`

These are relatively trivial to write - just use the output of the pipeline by accessing the visfr.pipeline object as
though it were a dictionary. e.g. ::

    visfr.pipeline['x']

See also `VisGUI column names` for details on what column names can be used.

Occaisionally you might also want to use the :class:`colour filter <PYME.IO.tabular.colourFilter>` to switch between
colour channels. :mod:`PYME.LMVis.Extras.photophysics` has a good example of this.

Plugins which modify the pipeline
---------------------------------

These are a little harder. The general procedure (alpha) is as follows:

#. Find or write recipe module(s) which perform the desired task
#. For each of the modules
    #. Create an instance of each recipe module, using ``pipeline.recipe`` as the parent, and the current selected datasource
       key or the ``outputName`` of the previous module as the ``inputName``.
    #. Add the module instance to ``pipeline.recipe.modules``
#. Execute the recipe
#. Update the selected data source to point to the output of the last module.

An example below: ::

    def OnDBSCANCluster(visfr):
        from PYME.recipes.tablefilters import DBSCANClustering
        clumper = DBSCANClustering(visfr.pipeline.recipe, inputName=visfr.pipeline.selectedDataSourceKey, outputName='dbscanClumps')

        if clumper.configure_traits(kind='modal'):
            visfr.pipeline.recipe.modules.append(clumper)
            visfr.pipeline.recipe.execute()
            visfr.pipeline.selectDataSource('dbscanClumps')

    def Plug(visfr):
        visfr.addMenuItem('Extras', 'Find DBSCAN clusters', lambda e : OnDBSCANCluster(visfr))

.. warning::

    This is exceptionally new and might not currently work as expected. There are several things yet to be done:

    * Make the recipe re-execute when parameters etc ... change.
    * Add convenience functions for adding recipe modules to reduce the boiler plate.
    * Refactor existing code to use the new scheme.


VisGUI column names
===================

The core column names that should be defined in VisGUI and you can rely on in the pipeline output are as follows:

+----------------+---------+--------------------------------------------+
+ Name           + Units   + Description                                +
+================+=========+============================================+
+ x              + nm      + x position of points in nm                 +
+----------------+---------+--------------------------------------------+
+ y              + nm      + y position of points in nm                 +
+----------------+---------+--------------------------------------------+
+ z              + nm      + z position (focus and offset combined)     +
+----------------+---------+--------------------------------------------+
+ t              + frames  + frame num at which a point was observed    +
+----------------+---------+--------------------------------------------+



.. New Rendering Modules
.. =====================

I've got a neat algorithm in another language, can I use it in PYME?
====================================================================

The answer to this question is almost certainly yes, with the best solution depending on what language the original
algorithm is written in. Algorithms written in low level languages such as c are comparatively easy to interface, 
using tools such as cython and ctypes. Interfaces from python to high level languages, such as R and MATLAB are also 
available (e.g. the `rpy2 package <https://rpy2.github.io/>`_  and `MATLAB's engine 
library <https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html>`_ ) but these
typically require the installation of large additional software packages on the users computer with potential dependency
and licensing issues. As a result, whilst the r2py and matlab engine interfaces are appropriate for testing and
internal use, it is generally advisable to translate the code to either python, c, or a standalone DLL before trying to
share the code with others. Translating MATLAB code to python is quite easy (see, e.g., `the numpy 
documentation for MATLAB users <https://numpy.org/doc/stable/user/numpy-for-matlab-users.html>`_).


.. rubric:: Footnotes

.. [#recipenames] :meth:`~PYME.LMVis.pipeline.Pipeline.selectDataSource` effectively allows you to 'walk' the recipe
    namespace.

.. [#datasourceinsertion] You can still technically inject a new data source using ``pipeline.addDataSource``, but it
    is now injected into the recipe namespace. New code should avoid this and use the variable section instead.

.. [#circularflow] The classic example of this is/was event clumping. You took the output of the pipeline, used this to
    determine and extract clumped positions, and then injected these upstream in the data sources and ran them through
    the pipeline again.

.. [#visfrindsviewer] The exception to this is when a VisGUI plugin is loaded from within dsviewer, by way of the
    ``visgui`` plugin. In either case, the argument to ``Plug(...)`` is guaranteed to have ``.pipeline`` and
    ``.AddMenuItem(...)`` attributes.

.. [#pymenf] PYMEnf is a module which is used internally within the Baddeley and Soeller groups and contains code that we
    cannot distribute due to licensing restrictions, contains sensitive information, or for some other reason is not
    ready for public release.
