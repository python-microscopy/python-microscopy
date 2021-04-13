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

Define what is API and what is internal
---------------------------------------

Decide what is likely to be used from external code / plugins / contributions. Annotate and document what constitutes the
API. Make this API more consistent and more clearly structured.


New features / Improvements
===========================


UI
--


UI alignment between VisGUI and dh5view
'''''''''''''''''''''''''''''''''''''''

With the goal of promoting a more seamless user-experience and reducing the amount of stuff to be learnt by a new user.
Goal is that if you can use PYMEImage you have a head-start on PYMEVisualise and vice versa. Should also help code re-use
by allowing us to remove quite a lot of duplicated code (e.g. pixel -> nm translations etc ...) Includes:

- Making display settings / layers appear in the same place in PYMEImage and PYMEVisualise (maybe right hand sidebar)
- Making overlays and scaling work the same in both
- Allowing easier overlaying of image and localisation data.

And lays the framework for having things like annotation modules which can work on both image and point data.

OpenGL based replacement for dh5view image display
''''''''''''''''''''''''''''''''''''''''''''''''''

With support for large, tiled, datasets and tile pyramids. Should improve viewer performance. Potentially a part of UI alignment above.


Standardised UI components for tabular data and "reports"
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This would take the form of a spreadsheet type view for tabular results in recipes, and a viewer for HTML reports generated
by recipes in dh5view/visgui. Longer term it would involve porting many of the current analysis tasks that display matplotlib
windows to producing html reports.

Better logging / error reporting in UI apps
'''''''''''''''''''''''''''''''''''''''''''

Largely addressed with error context manager on menu items, but doesn't capture everything.

Web UI
''''''

Would allow:
- the use of a hosted version of PYME without installation
- quick exploration of cluster data
- more platforms (e.g. phones & tablets)
- reduced dependence on traitsui

General housekeeping
''''''''''''''''''''

- tidy up shader code (which currently has at least one too many layers of abstraction)
- tidy/sort menus
- tidy/sort plugins
- unified splashscreens when loading (and update institutional icons etc ...)
- better support for associating files with UI components
- single UI entry point??
- make windows use non-generic icons in taskbar
- improve PYMEImage opening speed (currently limited by clusterIO name resolution)


IO
--

- more support for tiled/chunked image formats (e.g. zarr)
- move to a true 5D data model
- better OME interop
  - get PYME formats into bioformats
  - access files from OMERO
  - push stuff to OMERO

Localisation Analysis
---------------------

- Make it easy to plug custom localisation routines
- 3D multi-emitter fitting
- Refresh / fix fitInfo localisation inspection
- Other sample quality stuff?

Acquisition
-----------

- add support for using micromanager hardware drivers
- expand and better document hardware base classes
- clearly document how new hardware types (e.g. Adaptive optics, FPGAs etc) should interface with PYMEAcquire
- write an initialisation script wizard to lower the barrier to setting up PYMEAcquire on new microscopes

Recipes
-------

- add support for parallelism on a per-chunk rather than per image basis
- deprecate the `processFramesIndividually` option in favour of separate minumum chunk size and
- re-organise modules to make them easier to find. Potentially push some of the more esoteric stuff out into plugins

Planned Deprecations
====================

Library support
---------------

Ultimately we would like to drop support for both python <3.6 and wxpython <4. There are a number of things we need to
address first. Several bits of the GUI are still broken on wx4 (most notably anything which uses TraitsUI, but also some
of the less used bits of our GUI code).

We are currently around 98% done with the transition, with [3.6 <= python <= 3.7] and wx=4.0.x recommended for new installs,
but maintaining code compatibility with existing python 2.7 and wx3 installs for another few months
(targeting end of March 2021, extended from Feb 1 2021). There are a bunch of deprecation warnings on wx4.0.x which
become errors on 4.1.x which need addressing once we drop wx3 support.

Documentation
=============

Both the user facing and api documentation need a **lot** of work. An incomplete list of items

- Much of the existing end user documentation is out of date. Refresh this.
- Write more user documentation where needed. This should be a combination of general overviews (what are the components
and what are they good for) as well as task-specific walk-throughs, HOWTOs, tutorials etc ...
- Document recipes better
- Make sure functions which are likely to be called from plugin code (the API) all have docstrings (ideally all functions
should have docstrings, but this is a realistic starting point).


Packaging / Distribution
========================

Continuous Integration & Testing
--------------------------------

We currently have some CI based testing, but this is pretty limited. Packaging etc is done manually.

Testing:
''''''''

- fix failing tests
- improve test coverage
- run coverage checks on newly submitted PRs (and get this summarized nicely / with a bot etc ... so we can see if a given PR improves coverage)

Packaging:
''''''''''

- set up automatic package builds (conda, pip)
- set up automatic builds of executable installers


Conda packages & dependencies
-----------------------------

There are still a few holes in our default conda based packaging:

- start building conda packages for python-microscopy for py3
- ensure all dependencies are being built for py3.6 and 3.7
- make it easy to install bioformats (this might mean maintaining conda packages for both bioformats and a JVM)
- consider packaging shapely (not available across platforms from the core conda channels)

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

Stacked against this is the pain that is conda dependency management. My gut feeling is to stick with conda as the default
install route, but to offer pip as an option for people with a little move technical expertise.


