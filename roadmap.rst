============
PYME Roadmap
============

This is a combination wishlist and concrete roadmap with the aim of giving some idea as to where PYME might be going in
the medium term future, along with detailing some of the open questions about the project direction. This does not attempt
to be exhaustive.

.. contents:: Table of Contents
.. section-numbering::

Restructuring / Refactoring
===========================

UI / Logic separation
---------------------

Whilst we have already come a long way, there are still several places where application logic is included in the wx GUI
code, making alternative UI frontends / api-based access challenging. A long term project is to push for as much model/
view separation as possible.

Splitting into component packages
---------------------------------

PYME has become quite a behemoth, with a long list of dependencies which can be hard to satisfy, particularly
when combined with potential complimentary packages (e.g. keras, cell profiler, python-biofomats, etc ...). It would be
nice to break it down into smaller packages so that users only install those components they need, ideally reducing the
maintenance burden, and improving re-use.

The exact nature of this split is unclear, but could look something like core, core-ui (which would include both dh5view
and VisGUI), and PYMEAcquire. Relatively self-contained analysis features such as the meshing functions could also be
split out.


New features / Improvements
===========================


UI
--


UI alignment between VisGUI and dh5view
'''''''''''''''''''''''''''''''''''''''


Web UI
''''''


Better logging / error reporting in UI apps
'''''''''''''''''''''''''''''''''''''''''''


Recipe error handling
'''''''''''''''''''''

Cleanup of VisGUI shader related code
'''''''''''''''''''''''''''''''''''''

Standardised UI components for tabular data and "reports"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

OpenGL based replacement for dh5view image display
''''''''''''''''''''''''''''''''''''''''''''''''''

With support for large, tiled, datasets.


IO / Analysis
-------------



Planned Deprecations
====================

Library support
---------------

Ultimately we would like to drop support for both python <3.6 and wxpython <4. There are a number of things we need to
address first. Several bits of the GUI are still broken on wx4 (most notably anything which uses TraitsUI, but also some
of the less used bits of our GUI code). We also rely on python 2.7 libraries for spooling and localisation analysis on
windows - we are pretty close to having an alternative, but are not quite there yet. There also needs to be a lot more
testing on Py3.

A tentative timeline would see us shifting the default install to py3 around 1 Sept 2020 and ceasing python 2 & wx3
support around 1 Feb 2021. Note that these dates are targets, not deadlines, and will be extended if things are not fully
functional by that time.

Documentation
=============


Packaging / Distribution
========================

Continuous Integration & Testing
--------------------------------

Conda packages & dependencies
-----------------------------

Py3
'''


Streamlined bioformats installation / packaging
'''''''''''''''''''''''''''''''''''''''''''''''

Shapely?
''''''''


Pip-installable packages (wheels)
---------------------------------

Because conda dependency resolution can be a bit of a nightmare, there have been calls for a pip-installable version.
Some level of pip-installability would certainly be desirable, but it's a hard call whether to aim to make this the
default, or to leave it as an experts-only approach. As it stands, most people with the development expertise and tooling
to get a functional pip-installed version running would likely be better served with a development install. This might
change as we break PYME up into smaller component packages, and it seems reasonable to aim to give pip equal weight to
conda for the spun-out components, starting with pymecompress - the one package which has already been spun out.

When contemplating pip as a distribution means, we need to recognise that some of our users will not have a c compiler
installed, and may not have the technical prowess (or potentially even the access rights - some users here are on
centrally managed systems with no admin privileges) to install a c compiler. This means making binary wheels and
ensuring that binary wheels are available for all our dependencies (a quick test of the pip installation route on
linux confirmed that there are currently a fair number of packages which need to be compiled when pip-installing).

The other arguments for conda over pip are:

- conda links numpy against MKL rather than ATLAS, which can lead to a substantial performance improvement
- conda installs tend to be more self-contained (pip installs often assume the presence of various OS libraries - e.g. HDF5 and FFTW which may or may not be present in the right version)
- conda has a nice way of installing menu items/ shortcuts. If you 'conda install python-microscopy' on windows you now get links to the component programs in your start menu. This is not possible using pip.
- conda constructor offers a reasonably simple way of creating an executable installer for windows and OSX.

Stacked against this is the pain that is conda dependency management. My gut feeling is to stick with conda as the default install route, but to offer pip as an option for people with a little move technical expertise.


