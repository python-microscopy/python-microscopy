Using PYME's cluster rules
***************************

**Rules** are essentially a json template which defines how tasks may be 
generated (by substitution into the template on the client side). Generation of
individual tasks on the client has the dual benefits of a) reducing the CPU 
load and memory requirements on the server and b) dramatically decreasing the 
network bandwidth used for task distribution.

The rules API, defined in ``PYME.cluster.rules.py``, can be used to distribute
tasks programatically on the PYME cluster from third-party software.

Recipe Rules
============

Recipe rules can be used apply a single :ref:`recipe <recipes>` to multiple 
files in parallel. A ``RecipeRule`` expects recipe text, an output directory
path that is relative to the PYME cluster's root data directory and at least
one input. The recipe text can alternatively be passed via a cluster URI that
points to recipe text.

For example, we can create a recipe that will simulate random points and save
them in an output directory of our choice.

::
    recipe_text= '''
    - simulation.RandomPoints:
        output: points
    - output.HDFOutput:
        filePattern: '{output_dir}/test.hdf'
        inputVariables:
        points: points
        scheme: pyme-cluster://
    '''

.. note::

    ``scheme: pyme-cluster://`` puts the output into the cluster file system so 
    that it is accessible to subsequent recipe runs etc ... This is typically 
    ``~/PYMEData`` in a :ref:`cluster-of-one <localisationanalysis>` scenario 
    unless otherwise specified through `PYME.config <api/PYME.config>` It also 
    routes all IO through the dataserver, avoiding IO conflicts.

    To concatenate the results from multiple operations into a single table, 
    use ``scheme: pyme-cluster:// - aggregate``.

Recipe tasks are generated for each input in the rule - for simulation we use 
the ``__sim`` proxy input. For ``__sim`` "inputs" the filename can be an 
arbitrary string, and is propagated to the ``sim_tag`` context variable which 
can be used in the output `filePattern`.

::
    from PYME.cluster import rules

    r = rules.RecipeRule(recipe=recipe_text, output_dir='test', inputs={'__sim':['1']})
    r.push()

Inputs are specified as a dictionary, where the key is passed in as the name of 
the input datasource, for direct use within the recipe, and the value must be a 
*list* of data sources.  ``__sim`` is a special case where no input is used in 
the recipe. If, instead, we wanted to create a recipe that read in a data set, 
created a surface, and then saved this surface to file, we would execute the 
following.

::
    recipe_text = '''
    - pointcloud.Octree:
        input_localizations: shape_shape
        output_octree: octree
    - surface_fitting.DualMarchingCubes:
        input: octree
        output: mesh
        remesh: true
    - output.STLOutput:
        filePattern: '{{output_dir}}/my_surface.stl'
        inputName: membrane
        scheme: pyme-cluster://
    '''

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, 
                      inputs={'shape': [f'pyme-cluster:///{output_dir}/shape.hdf']})

    rule.push()

Note that in this case, an HDF file is passed as input. This is opened in the
pipeline as a data source with name ``<input_name>_<table_name>``. In this 
case, the table name is also named ``shape`` and so ``input_localizations`` in 
the recipe is named ``shape_shape``.
