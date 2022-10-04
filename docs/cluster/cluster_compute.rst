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

If interfacing the task-distribution architecture from another language, the low level API must be
used. 

Ruleserver REST API
''''''''''''''''''''

This is a HTTP REST interface on the ruleserver. Similarly to the
cluster filesystem servers, this is discoverable using mDNS protocol. using the ``_pyme-taskdist.tcp``
service type.

The following HTTP endpoints are available for submiting rules and checking rule status:

============================================================================================== ================
Endpoint                                                                                       Description
============================================================================================== ================
:py:meth:`/add_integer_id_rule <PYME.cluster.ruleserver.RuleServer.add_integer_id_rule>`       Submit a new rule (takes a task template in the request body)
:py:meth:`/release_rule_tasks <PYME.cluster.ruleserver.RuleServer.release_rule_tasks>`         Extend the range of frames for a rule (used when streaming)
:py:meth:`/mark_release_complete <PYME.cluster.ruleserver.RuleServer.mark_release_complete>`   Indicate that there are no more frames coming (used when streaming)
:py:meth:`/queue_info_longpoll <PYME.cluster.ruleserver.RuleServer.queue_info_longpoll>`       JSON formatted information on the progress of current rules
:py:meth:`/inactivate_rule <PYME.cluster.ruleserver.RuleServer.inactivate_rule>`               Inactivate (cancel) a rule.
============================================================================================== ================

There are also a number of endpoints used internally within the cluster during scheduling and 
task execution. These are detailed in the docs for :py:class:`PYME.cluster.ruleserver.RuleServer`


Task templates
''''''''''''''

At the heart of each rule is a JSON-formatted *task template*. This template is used on worker nodes
to generate individual tasks. The *task template* differs slightly between rule types, but always
has ``id`` and ``type`` keys. The template formats for localisation and recipe tasks are detailed
below. In all cases, ``{{ruleID}}``, ``{{taskID}}`` and any other escaped parameters (e.g. ``{{taskInputs}}``)
are replaced using string substitution on the worker nodes prior to parsing the json. ``{{taskID}}``
is the magic parameter which permits multiple tasks to be generated from a single rule. It will 
always be an integer, within a range which has been released. 

**Localisation:**

.. code-block:: text

    {
      "id": "{{ruleID}}~{{taskID}}",
      "type": "localization",
      "taskdef": {"frameIndex": "{{taskID}}", "metadata": "PYME-CLUSTER:///path/to/series/analysis/metadata.json"},
      "inputs": {"frames": "PYME-CLUSTER:///path/to/series.pcs"},
      "outputs": {"fitResults": "HTTP://a.cluster.ip.address:port/__aggregate_h5r/path/to/analysis/results.h5r/FitResults",
                  "driftResults": "HTTP://a.cluster.ip.address:port/__aggregate_h5r/path/to/analysis/results.h5r/DriftResults"}
    }

where localisation fit type and settings are specified in the analysis metadata. For localisation tasks,
``{{taskID}}`` maps to the index of a frame within the image series (one rule generates tasks for every
frame). In this example we have short-circuited the cluster load distribution for the output files to 
specify a specific data server (this can be useful when using the `__aggregate` endpoints to avoid race conditions when different nodes try to create the 
file at the same time).

**Recipe:**

.. code-block:: text

    {
      "id": "{{ruleID}}~{{taskID}}",
      "type": "recipe",
      "taskdef": {"recipe": "<RECIPE_TEXT>"},
      "inputs": {{taskInputs}},
      "output_dir": "PYME-CLUSTER:///path/to/output/directory",
      "optimal-chunk-size": 1
    }

For recipe tasks, ``{{taskID}}`` maps to an index into a dictionary of inputs provided to 
``/add_integer_id_rule``.


Or alternatively, where the recipe is specified as a path to a recipe file on the cluster, and the
input is directly specified (only suitable when generating a single task from this rule).

.. code-block:: text

    {
      "id": "{{ruleID}}~{{taskID}}",
      "type": "recipe",
      "taskdefRef": "PYME-CLUSTER:///path/to/recipe.yaml",
      "inputs": {"input": "PYME-CLUSTER:///path/to/somefile.tif"},
      "output_dir": "PYME-CLUSTER:///path/to/output/directory",
      "optimal-chunk-size": 1
    }



Example
'''''''

This creates a recipe rule which performs a Gaussian filter on a set of files. 

**Recipe (filter.yaml):**

.. code-block:: yaml

    - filters.GaussianFilter:
        inputName: input
        outputName: filtered
        sigmaX: 5.0
        sigmaY: 5.0
    - output.ImageOutput:
        filePattern: '{output_dir}/{file_stub}.tif'
        inputName: filtered
        scheme: pyme-cluster://

**REST request**

.. code-block:: python

    import requests

    payload = '''

    {
      "template":  {
        "id": "{{ruleID}}~{{taskID}}",
        "type": "recipe",
        "taskdefRef": "PYME-CLUSTER:///path/to/filter.yaml",
        "inputs": {{taskInputs}},
        "output_dir": "PYME-CLUSTER:///path/to/output/directory",
        "optimal-chunk-size": 1
      },
      "inputsByTask": {
        0: {"input": "PYME-CLUSTER:///path/to/file0.tif"},
        1: {"input": "PYME-CLUSTER:///path/to/file1.tif"},
        2: {"input": "PYME-CLUSTER:///path/to/file2.tif"}
        }
    }
    ''''

    # NOTE - as we know the number of tasks in advance, we can provide the optional max_tasks, 
    # release_start, and release_end parameters to the REST call and avoid the need to call 
    # /release_rule_tasks and /mark_release_complete

    requests.post('HTTP://ruleserver.ip:port/add_integer_id_rule?max_tasks=3&release_start=0&release_end=3', 
                  data=payload, headers={'Content-Type': 'application/json'})






