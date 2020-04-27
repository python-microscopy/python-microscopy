============
PYME Roadmap
============

This is a combination wishlist and concrete roadmap with the aim of giving some idea as to where PYME might be going in
the medium term future, along with detailing some of the open questions about the project direction.

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



Planned Deprecations
====================

Library support
---------------

Ultimately we would like to drop support for both python <3.6 and wxpython <4. There are a number of things we need to
address first. Several bits of the GUI are still broken on wx4 (most notably anything which uses TraitsUI, but also some
of the less used bits of our GUI code). We also rely on python 2.7 libraries for spooling and localisation analysis on
windows - we are pretty close to having an alternative, but are not quite there yet. We also need to do a lot more testing
on Py3.

Documentation
=============


Packaging / Distribution
========================

Continuous Integration & Testing
--------------------------------

Conda packages for Py3
----------------------

Pip-installable packages (wheels)
---------------------------------
