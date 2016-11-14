.. _extendingdsviewer:

Extending the image data viewer (dh5view/View3D)
************************************************


The PYME image data viewer is implemented in :class:`PYME.DSView.dsviewer.DSViewFrame` and can be invoked either by running ``dh5view``
from the command line, or by importing and calling :func:`PYME.DSView.View3D` or :func:`PYME.DSView.ViewIm3D` from within
python code [#needwx]_. Most of the viewers image processing functionality is accomplished by plugins, and this guide
will focus on writing new plugins.

dsviewer plugins
================

``dsviewer`` plugins are python files which are located in the ``PYME.DSView.modules`` directory. Any file that
is located in within that directory will be automatically detected and treated as a plugin.

.. note::

    ``PYME.DSView.modules`` and ``PYMEnf.DSView.modules`` [#pymenf]_ are currently the only locations where modules will be detected.
    A more flexible mechanism of module discovery is high on the TODO list.

Plugins **must** implement a function called ``Plug(dsviewer)`` which takes an instance of the current
:class:`PYME.DSView.dsviewer.DSViewFrame`, and can implement any additional python logic. It is good practice not to put
too much processing logic in the plugin itself [#pluginmvc]_. A typical ``Plug()`` method instantiates a class which stores
plugin state, keeps a reference to the ``DSViewFrame`` object, and either adds menu items for it's functions, or registers
callbacks for overlays or GUI panels.


A :class:`PYME.DSView.dsviewer.DSViewFrame` instance exposes three important
attributes:

#. ``dsviewer.image`` : A reference to the currently displayed :ref:`_imagestack` object.
#. ``dsviewer.do`` : A reference to a :class:`PYME.DSView.displayOptions.DisplayOpts` instance which stores the display
   settings for the current image. This is useful for determining the current position in the stack, for extracting
   manual threshold levels, and for setting overlays.
#. ``dsviewer.view`` : A reference to the current view class (not commonly used).



.. rubric:: Footnotes

.. [#needwx] The program should be running a wxpython event loop. This will always be the case if called within one of
    the PYME GUI programs (dh5view, VisGUI, PYMEAcquire). If you want to call ``View3D`` or ``ViewIm3D`` from an ipython
    session you will need to run ``ipython --gui=wx`` to make sure the wx event loop is running. In an ipython/jupyter
    notebook you will need to use the ``%gui wx`` magic before running View3D.

    Running from an ipython notebook with anaconda on OSX requires some additional fiddling - you have to change
    the shebang of ``/PATH/TO/anaconda/bin/ipython`` to point to the framework copy of python (usually
    ``PATH/TO/anaconda/python.app/Contents/MacOS\python``) so that ipython notebooks can access the display without dying.


.. [#pymenf] PYMEnf is a module which is used internally within the Baddeley and Soeller groups and contains code that we
    cannot distribute due to licensing restrictions, contains sensitive information, or for some other reason is not
    ready for public release.

.. [#pluginmvc] Although the model-view-controller pattern is poorly followed in the majority of PYME code, it is
    useful to think of plugins existing at the controller level - providing the interface between image processing
    routines and libraries and the view code. That said, a lot of existing plugin code includes both GUI and program logic.