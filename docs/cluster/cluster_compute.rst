Distributed computation on the PYME Cluster
*******************************************

The PYME cluster architecture contains tools for distributed analysis of 
imaging data in a "data-local" manner where computation is performed on
the nodes housing the data, avoiding IO overhead which would otherwise limit
performance.

Compute tasks
=============

We currently support two types of computational tasks (below), although the
framework is designed to allow additional task types to easily be added in 
the future. 

Localisation tasks
------------------

These are a localisation-microscopy specific task type optimised for
performing single-molecule localisation on a single frame of an image series.
A wide range of different localisation algorithms are supported (for more details see :ref:`localisationanalysis`)

"Recipe" tasks
--------------

These are a more generic task type which supports a range of different image
and data-processing functions specified by combining analysis building blocks
into :ref:`recipes <recipes>`. Recipes are specified using a .yaml text format,
and can either be created manually or using the recipe editor.


Rule-based task scheduler
=========================

PYME takes a different approach to task scheduling than many 
common cluster schedulers. Rather than centrally definining individual tasks 
and distribututing these across cluster nodes from a central server, 
we define so-called "**Rules**" which decribe how to generate tasks.
Each cluster node contacts the central server for a copy of the currently
active rules, generates candidate tasks based on the rules, scores these
based on node capability and whether the required data is available
locally on the that node, and then submits a bid to the central server 
specifying which tasks it want's to perform and at what cost. The central 
server then compares the bids and replies with a set of tasks indices for
which the node has won the bid.

This architecture solves a number of problems faced when trying to distribute
a very large number (thousands per second) of small tasks. It has 3 main 
advantages: 

- It dramatically reduces the network traffic required for task distribution, 
  with only the rule and integer task IDs transmitted between nodes.
- It distributes the work of assigning the tasks across the entire cluster. 
  Most notably, it allows data locality checking to be performed locally. 
  *"Do I have this file?"* turns out out to be a **much, much** cheaper question than 
  *"Which node has this file?"*. 
- It reduces memory usage in storing task descriptions, with task-descriptions
  only generated for enough tasks to full each nodes input queue. This is 
  critically important when, e.g. re-analysing streamed data, where a 
  conventional task allocation strategy would lead to the generation of 200,000,000 
  tasks for a day's worth of streamed image frames. 


High-level rule API
-------------------

Recipe Rules
''''''''''''

Recipe rules can be used apply a single :ref:`recipe <recipes>` to multiple 
files in parallel. A ``RecipeRule`` expects recipe text, an output directory
path that is relative to the PYME cluster's root data directory and at least
one input. The recipe text can alternatively be passed via a cluster URI that
points to recipe text.

For example, we can create a recipe that will simulate random points and save
them in an output directory of our choice.

.. code-block:: python

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

.. code-block:: python

    from PYME.cluster import rules

    r = rules.RecipeRule(recipe=recipe_text, output_dir='test', inputs={'__sim':['1']})
    r.push()

Inputs are specified as a dictionary, where the key is passed in as the name of 
the input datasource, for direct use within the recipe, and the value must be a 
*list* of data sources.  ``__sim`` is a special case where no input is used in 
the recipe. If, instead, we wanted to create a recipe that read in a data set, 
created a surface, and then saved this surface to file, we would execute the 
following.

.. code-block:: python

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



Low level API
-------------

