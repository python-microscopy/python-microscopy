.. _recipes::

Using recipes for data processing and quantification
****************************************************

Recipes are an experimental feature of PYME for automating image post-processing and anaylsis tasks, for both
super-resolution and conventional microscopy data. They allow you do define a processing workflow consisting of multiple
different tasks such as filtering or quantification and to run each of these tasks automatically over a set of images.
The concept is broadly similar to the 'block' or 'pipeline' processing models used in e.g. OpenInventor, Simulink,
Khorus/Cantata, or CellProfiler. My motivation behind developing recipes was to allow image processing scripts to be
written quickly without having to worry about file I/O, and keeping track of e.g. metadata, and for these scripts to be
able to be run in a variety of environments, with or without a GUI. The principle conceptual differences are thus:

  - PYME recipes have been designed around an easily edited textual representation (although a GUI is also available). 
  - Image metadata (e.g. pixel sizes) is automatically propagated with the images.
  - many modules are very thin wrappers around the corresponding functions in either scipy or scikits-image. 

An example of a recipe (as displayed in the GUI) is shown below.

.. image:: /images/recipe.png

Connections between modules, execution order, and dependencies
==============================================================

Recipes differ from many of the existing block or pipeline architectures in that connections between modules and the
order in which modules are executed are both defined implicitly. Inputs and outputs to each module are assigned names,
or keys, which are then used to find locate each modules input data in a common namespace (implemented as a python
dictionary) and to store the modules output(s) in the same namespace. The connections between modules are then inferred
by matching input and output names, and the execution order determined using a dependency solver such that modules do
not execute before their inputs have been generated. This allows for a compact and quickly adaptable representation of
the recipe, which can easily be altered without having to manually re-connect a complex net. To link two modules, you
simply have to set the input name on the second module to the output name of the first.

Module names can be pretty much anything you choose, and should be descriptive. Avoiding spaces and non-alphanumeric
characters however is advised, particularly for input and output names. When used for batch processing, all names
beginning with **in** are considered to be inputs, and all names beginning with **out** are marked as outputs, and will
be saved to disk in the output directory. When operating within *dh5view*, recipes currently only support one input
(called **input**) and one output (**output**). In this case the **input** image is the currently open image, and the
**output** is displayed in a new window (or, in the case of measurement data as annotations to the current image).

Creating and running recipes
============================

Recipes can be visually constructed using the PYME bakeshop. The bakeshop is opened by running

.. code-block:: bash

    bakeshop

Recipe modules can then be added using the "Add Module" button, which will bring up a menu of module choices. Clicking
any of these modules displays documentation about the module and its parameters. After selecting the desired module,
clicking "Add" will open a dialog to enter input/output parameters for the recipe module. Once these parameters are
entered, the module can be added by clicking "OK".

Modules can be edited once they have already been added either visually by clicking on the recipe module tile or
textually using the yaml-formatted text on the right-hand side of the bakeshop GUI. If you edit the recipe textually,
please press "Apply" before running or saving the recipe or your changes will be ignored.

Once the recipe is built, it can be saved or run directly in the bakeshop GUI. To run the recipe, enter the file pattern
in order to select the input files (e.g. ``~/*.h5r`` to select all h5r files in the top of the user's home directory) before
clicking "Get Matches". Enter the desired output directory where indicated before pressing "Bake".

Batch Processing
================

Batch processing allows you to automatically run a pre-defined recipe over a series of input files. 

Command line interface
----------------------

The command line interface to batch processing is the script located at ``PYME/Analysis/Modules/batchProcess.py``, and
can be called, for example as follows:

.. code-block:: bash
	
	python /path/to/batchProcess.py -input1=/path/to/input/channel1*.tif -input2=/path/to/input/channel2*.tif recipe.yaml /path/to/output/dir


