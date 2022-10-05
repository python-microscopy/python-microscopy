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

Localisation Rules
''''''''''''''''''

Localisation rules which generate a task for each frame in an image series can be invoked 
as follows ...

.. code-block:: python

    from PYME.IO import MetaDataHandler
    from PYME.cluster import rules

    # Specify analysis settings in metadata
    analysis_metadata = MetaDataHandler.DictMDHandler({
      'Analysis.FitModule':'LatGaussFitFR', # 2D Gaussian fit (CPU)
      'Analysis.DetectionThreshold': 1.0, # SNR based candidate detection threshold
      'Analysis.StartAt': 30, # start at frame 30
      'Analysis.BGRange': [-30,0], # Use an average of the previous 30 frames for background correction
      'Analysis.subtractBackground': True,
    })

    # create rule and push to the cluster
    rules.LocalisationRule(seriesName='PYME-CLUSTER:///path/to/series.pcs', analysisMetadata=analysis_metadata).push()

For more details on the different types of fits and their parameters see  :ref:`localisationanalysis`. 
For data sets which do not already contain good metadata on image noise properties 
(read noise, AD offset, AD conversion factors, etc ...) or pixel size, these can be provided as
part of the analysis metadata above. An example is provided below.

.. code-block:: python

    from PYME.IO import MetaDataHandler
    from PYME.cluster import rules

    # Specify analysis settings in metadata
    analysis_metadata = MetaDataHandler.DictMDHandler({
      'Analysis.FitModule':'AstigGaussGPUFitFR', # Astigmatic Gaussian fit (GPU)
      'Analysis.DetectionThreshold': 1.0, # SNR based candidate detection threshold
      'Analysis.StartAt': 30, # start at frame 30
      'Analysis.BGRange': [-30,0], # Use an average of the previous 30 frames for background correction
      'Analysis.subtractBackground': True,
      'Camera.ADOffset': 100,  # Camera properties (these approximate what you'd expect for an sCMOS camera) 
      'Camera.ElectronsPerCount': 0.45,
      'Camera.ReadNoise': 0.5,
      'Camera.TrueEMGain': 1.0,
      'Camera.VarianceMapID': 'PYME-CLUSTER:///path/to/variance.tif', # per-pixel maps of sCMOS noise properties, in units of ADUs
      'Camera.DarkMapID': 'PYME-CLUSTER:///path/to/dark.tif',
      'Camera.FlatfieldMapID': 'PYME-CLUSTER:///path/to/flatfield.tif',
      'voxelsize.x': 0.105, # x pixel size in um
      'voxelsize.y': 0.105, # y pixel size in um
    })

    # create rule and push to the cluster
    rules.LocalisationRule(seriesName='PYME-CLUSTER:///path/to/series.tif', analysisMetadata=analysis_metadata).push()

.. warning::

    The used of a .tif formatted series in the above example is for illustration only. Starting 
    cluster-based analysis on series stored as .tif
    is likely to result in pathological performance as the whole .tif file will be copied to each of
    the nodes. It is strongly reccomended to convert to the sharded .pcs format first. Even on a single
    node cluster (``PYMEClusterOfOne``) a substantial performance boost will be obtained by converting 
    the format.



Recipe Rules
''''''''''''

Recipe rules can be used apply a single :ref:`recipe <recipes>` to multiple 
files in parallel. A ``RecipeRule`` expects recipe text, an output directory
path relative to the PYME cluster's root data directory and at least
one input. The recipe text can alternatively be passed via a cluster URI that
points to recipe text.

For example, we can create a recipe that will take a number of localisation 
data sets, filter them to exclude any localisations with an estimated 
localisation error > 30 nm, generate Gaussian-based density estimates,
and save the results to tif files.
 

.. code-block:: python

    recipe_text = '''
        - localisations.AddPipelineDerivedVars:
            inputEvents: ''
            inputFitResults: FitResults
            outputLocalizations: Localizations
        - localisations.ProcessColour:
            input: Localizations
            output: colour_mapped
        - tablefilters.FilterTable:
            filters:
            error_x:
                - 0
                - 30
            inputName: colour_mapped
            outputName: filtered_localizations
        - localisations.DensityMapping:
            inputLocalizations: filtered_localizations
            jitterVariable: error_x
            outputImage: image
            renderingModule: Gaussian
        - output.ImageOutput:
            filePattern: '{output_dir}/{file_stub}.tif'
            inputName: image
            scheme: pyme-cluster://
    '''

    rule = RecipeRule(recipe=recipe_text, output_dir=output_dir, 
                      inputs={'input': ['pyme-cluster:///path/to/Series_0000.h5r',
                                        'pyme-cluster:///path/to/Series_0001.h5r',
                                        'pyme-cluster:///path/to/Series_0002.h5r',
                                        'pyme-cluster:///path/to/Series_0003.h5r',
                                        'pyme-cluster:///path/to/Series_0004.h5r']})

    rule.push()

Inputs are specified as a dictionary, which maps recipe namespace keys to a list of filenames. 
The rule will generate tasks for each filename in the list, and the file data will be accessible 
to the recipe under the dictionary key. When a recipe only has one input it is (soft) convention to use 
the ``'input'`` key name as this preserves compatibility with recipe usage within the ``PYMEImage``
and ``Bakeshop`` utilities. Multiple inputs can be specified by adding additional keys - the length
of the file lists must be the same for each key.

.. note:: HDF input table name mangling

    As HDF files (.h5r, .hdf) can contain multiple independant tables, they are mapped into the
    recipe namespace differently to other data types (images, .csv, etc). Each table in the 
    HDF file is mapped as a separate datasource with a mangled input name ``<input_key>_<table_name>``. 

    There is a further special case if ``<input_key>`` is the default ``"input"``, in which case
    HDF tables appear under ``<table_name>`` with no leading ``<input_key>_``.

.. note:: Output schemes

    ``scheme: pyme-cluster://`` puts the output into the cluster file system so 
    that it is accessible to subsequent recipe runs etc ... This is typically 
    ``~/PYMEData`` in a :ref:`cluster-of-one <localisationanalysis>` scenario 
    unless otherwise specified through :py:mod:`PYME.config`. It also 
    routes all IO through the dataserver, avoiding IO conflicts.

    To concatenate the results from multiple tasks into a single table, 
    use ``scheme: pyme-cluster:// - aggregate``. As the tasks run in parallel, 
    the ``aggregate`` scheme makes no gaurantees about ordering - if ordering is important
    recipes should add a column to the output table to allow ordering/reassignment in 
    postprocessing.



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






Recipe rules for simulation
===========================

The cluster task distribution schema can also be used to distribute simulation
tasks encoded by recipes. Because tasks are normally generated on a per input file basis 
we need to *trick* the framework into running inputless simulation tasks. This is achieved 
by using the ``__sim`` proxy input. For ``__sim`` *"inputs"* the filename can be an 
arbitrary string, and is propagated to the ``sim_tag`` context variable which 
can be used in the output ``filePattern``.

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

    from PYME.cluster import rules

    r = rules.RecipeRule(recipe=recipe_text, output_dir='test', inputs={'__sim':['1']})
    r.push()