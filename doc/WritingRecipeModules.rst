.. _writingrecipemodules:

Writing a Recipe Module
***********************

PYME recipes are a way of developing automated analysis pipelines by chaining together building blocks, or *recipe modules*.
Each recipe module reads its input from, and saves it's output into a common namespace (that looks like a python dictionary).


Anatomy of a recipe module
==========================

Recipe modules are classes which derive from :class:`PYME.recipes.base.ModuleBase`. They **must** override the
:meth:`~PYME.recipes.base.ModuleBase.execute` method, which takes the recipe namespace (a python :class:`dict` like
object). Recipes are expected to define string parameters defining the names / keys of their input and output data
and to draw their input(s) from the recipe namespace using the values of the input parameters as key(s), an store
their output(s) back in the namespace using the output name(s) as key(s). A trivial example of a recipe module (which
simply echos it's output) is given below: ::

    from PYME.recipes.base import ModuleBase, CStr

    class Echo(ModuleBase):
        inputName = CStr('input')
        outputName = CStr('output')

        def execute(self, namespace):
            namespace[self.outputName] = namespace[self.inputName]


Module parameters
-----------------

