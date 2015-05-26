Using recipes for data processing and quantification
****************************************************

Recipes are an experimental feature of PYME for automating image post-processing and anaylsis tasks, for both super-resolution and conventional microscopy data. They allow you do define a processing workflow consisting of multiple different tasks such as filtering or quantification and to run each of these tasks automatically over a set of images. The concept is broadly similar to the 'block' or 'pipeline' processing models used in e.g. OpenInventor, Simulink, Khorus/Cantata, or CellProfiler. My motivation behind developing recipes was to allow image processing scripts to be written quickly without having to worry about file I/O, and keeping track of e.g. metadata, and for these scripts to be able to be run in a variety of environments, with or without a GUI. The principle conceptual differences are thus:
  - PYME recipes have been designed around an easily edited textual representation (although a GUI is also available). 
  - Image metadata (e.g. pixel sizes) is automatically propagated with the images.
  - many modules are very thin wrappers around the corresponding functions in either scipy or scikits-image. 

.. image:: images/recipe.png