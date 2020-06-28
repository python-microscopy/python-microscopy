.. _writingrecipemodules:

Writing a Recipe Module
***********************

PYME recipes are a way of developing automated analysis pipelines by chaining together building blocks, or *recipe modules*.
Each recipe module reads its input from, and saves it's output into a common namespace (that looks like a python dictionary).
They generally take PYME image or tabular data types (see :ref:`datamodel`) as input, and output and will propagate any
relevant metadata.


Anatomy of a recipe module
==========================

Recipe modules are classes which derive from :class:`PYME.recipes.base.ModuleBase`. They **must** override the
:meth:`~PYME.recipes.base.ModuleBase.execute` method, which takes the recipe namespace (a python :class:`dict` like
object). Recipes are expected to define string parameters defining the names / keys of their input and output data
and to draw their input(s) from the recipe namespace using the values of the input parameters as key(s), an store
their output(s) back in the namespace using the output name(s) as key(s). A trivial example of a recipe module (which
simply echos it's output) is given below: ::

    from PYME.recipes.base import register_module, ModuleBase, Input, Output

    @register_module('Echo')
    class Echo(ModuleBase):
        inputName = Input('input')
        outputName = Output('output')

        def execute(self, namespace):
            namespace[self.outputName] = namespace[self.inputName]




Module parameters
-----------------

Module parameters are defined using `Traits <http://code.enthought.com/projects/traits/documentation.php>`_. Traits are
a form of typed parameter for python which know how to perform validation and, importantly, generate their own GUI. This
means that we can simply specify the parameter types and default values, and not have to worry about writing any of the
usual 'glue' code. A couple of good resources are `http://docs.enthought.com/traits/traits_user_manual/intro.html#what-are-traits`_
and `http://docs.enthought.com/traitsui/tutorials/traits_ui_scientific_app.html`_.

The code below shows an example of a slightly more sophisticated module which uses Traits for it's parameters. Note that
we import the traits from ``PYME.recipes.traits`` rather than ``traits.api`` [#traitsimport]_

::

    from PYME.recipes.base import register_module, ModuleBase
    from PYME.recipes.traits import Input, Output, Float, Int, Bool

    @register_module('OffsetAndScale')
    class OffsetAndScale(ModuleBase):
        inputName = Input('input')
        outputName = Output('output')

        offset = Float(0)
        scale = Float(1.0)

        def execute(self, namespace):
            from PYME.IO.image import ImageStack

            _in = namespace[self.inputName]
            _out = ImageStack(self.scale*(_in.data[:,:,:] - self.offset), mdh=_in.mdh)
            namespace[self.outputName] = _out


.. figure:: images/traits_generated_gui.png
    :scale: 50 %
    :align: center

    The GUI automatically generated from the above code.


Special parameters - input and output
-------------------------------------

In addition to general parameters which affect how processing should occur, modules should define input and output
variables which specify the **name** of the input and output data within the recipe namespace. These variables **must**:

* start with either 'input' or 'output'
* use the ``Input`` or ``Output`` trait types, as defined in ``recipes.base``

.. note::
    Old code (including much of the existing code base) uses ``CStr`` for both input and output names, and relies on the
    names starting with either 'input' or 'output to determine what is an input, what is an output, and what is a standard
    parameter. ``Input`` and ``Output`` are very thin wrappers of ``CStr`` which permit a more semantic declaration. In the
    future, we plan on porting all existing code to using the ``Input`` and ``Output`` traits, and will probably relax the
    restrictions on naming. For now, modules should conform to both conventions.

Making modules visible
----------------------

Making recipe modules visible in the recipes GUI is a two step process. Firstly the module should register itself by
using the ``@register_module()`` decorator, as illustrated above, which takes the display name as a string. Secondly, we
have to ensure that the python file  code containing the recipe module gets imported. This is currently achieved by adding
an appropriate import line to ``PYME.recipes.modules``. In the future a more flexible module discovery and import system
is planned.

Customizing Views
=================

If a module has a large number of parameters, it might be appropriate to customize how they are displayed. This can be
achieved by using `traitsui <http://docs.enthought.com/traitsui/index.html>`_  *Views*. PYME recipes build on top
of ``traitsui`` to support two types of view - a *default* view used when building and configuring a flexible recipe,
and a *pipeline* view which is a simpler view to be used when the parameters of a recipe might want to be configured
without effecting connectivity. For this reason, *pipeline* views hide the ``Input`` and ``Output`` parameters. To
customize either view one should override the :meth:`PYME.recipes.base.ModuleBase.default_view` or
:meth:`PYME.recipes.base.ModuleBase.pipeline_view` properties.

.. warning::

    In contrast to most of the *traitsui* example code which shows statically defined views, views within recipe modules
    **must** be created dynamically when the view property is accessed. This is to allow recipe modules to be used in
    cases where a GUI is not present and defining a View would otherwise crash the code (generally generating a fatal
    error). This is a **major** limitation with the current implementation of *traitsui*, but one we have to work around.

    This restriction on dynamic creation extends to importing the ``traitsui`` module, and one cannot even import the
    module without crashing in the absence of a GUI. This means that the import statements should be within the dynamic
    GUI generation function.



An example where a view has been over-ridden (in this case to use a custom editor for one of the parameters) is given below: ::

    @register_module('FilterTable')
    class FilterTable(ModuleBase):
        """Create a new mapping object which derives mapped keys from original ones"""
        inputName = Input('measurements')
        filters = DictStrList()
        outputName = Output('filtered')

        def execute(self, namespace):
            inp = namespace[self.inputName]

            map = tabular.resultsFilter(inp, **self.filters)

            if 'mdh' in dir(inp):
                map.mdh = inp.mdh

            namespace[self.outputName] = map

        @property
        def _ds(self):
            try:
                return self._parent.namespace[self.inputName]
            except:
                return None

        @property
        def pipeline_view(self):
            from traitsui.api import View, Group, Item
            from PYME.ui.custom_traits_editors import FilterEditor

            modname = ','.join(self.inputs) + ' -> ' + self.__class__.__name__ + ' -> ' + ','.join(self.outputs)

            return View(Group(Item('filters', editor=FilterEditor(datasource=self._ds)), label=modname))

        @property
        def default_view(self):
            from traitsui.api import View, Group, Item
            from PYME.ui.custom_traits_editors import CBEditor, FilterEditor

            return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                        Item('_'),
                        Item('filters', editor=FilterEditor(datasource=self._ds)),
                        Item('_'),
                        Item('outputName'), buttons=['OK'])

.. rubric:: Footnotes

.. [#traitsimport] ``PYME.recipes.traits`` is a very thin wrapper of ``traits.api``. This wrapper exists for two reasons:

    1. To allow us to add new traits such as ``Input`` and ``Output`` and to subclass individual
    Traits or even replace the Traits module completely at some point in the future without changing module code.

    2. To work hide the fact that the traits module can be found in one of two different locations - either ``traits.api``
    or ``enthought.traits.api``.