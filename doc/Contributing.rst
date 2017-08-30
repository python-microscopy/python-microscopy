Roadmap
=======

PYME has grown to be a rather large project with a huge number of dependencies. This is starting to make maintenance
time consuming and makes it hard to quickly merge new functionality. We intend to split the project into a GUI
independent core, and have the various GUIs interface with this. We also envisage splitting off some of the functionally
distinct modules. The exact restructuring has yet to be decided, but at this point we encourage separation of GUI and
logic to the extent possible. To facilitate this separation, we strongly suggest that new processing functionality for
either `dh5view` or `VisGUI` be implemented in the form of 'recipe' modules using `traits` to define parameters.

Python 3 Migration
==================

It is our intent to migrate at least the core functionality to Python3. It is however unlikely that all the 3rd party
dependencies we use will ever be supported. For this reason we will also continue to support Python 2.7. As several
compatibility features have been added to python3 since the initial version we will require at least python 3.6 and wil
not support earlier versions of python 3.

To date we have spent roughly a full week on migration, and addressed most of the errors which come up through static analysis,
but much remains to be done. At this point, PYME will build under python 3 is currently passing 42 of 46 tests although
the test coverage is currently abysmal.

The following tasks need to be addressed (in order of importance) before further progress can be made:

- find (or create) a reliable anaconda package for wxpython on python3 and add it to our channel. *NEW: An early wx build for python 3 suitable for testing
  is available by running `conda install -c newville wxpython-phoenix`*
- resolve backwards incompatibility issues in wxpython
- write more unit tests (current coverage is really bad)
- **write all new code so that it is compatible with both python 2.7 and python >= 3.6** using `six`, `future` and other
  compatibility modules as needed. This means using the functional form of `print`, using new style exception handling,
  as well as some care around strings. There are quite a few documents online about how to write compatible code. I'd
  also recommend setting up your editor to check for both python 2 and python 3 syntax.
- Go through and check string / byte string usage, and fix to make compatible with both python 2 and 3. I've already
  invested some time in this, but know that there are remaining issues.
- Port our Pyro code to a newer version of Pyro (the version we currently use does not support Python3 and the new
  version has backwards incompatible changes)
- Check for and correct relative module imports
- Build a conda module for dispatch on python3
- Fix all c coded modules to use new init methods

We will probably discover more as we progress.

Coding style
============

The coding style in PYME as it stands is fairly inconsistent. I'd like to improve on this. New code should:
- follow PEP8 where possible
- use Numpy style docstrings
- have strong model-view separation
- be written so as to be compatible with both python 2.7 and python > 3.6.

Pull requests
=============

Pull requests are always welcomed, but to increase the chances of speedy review and incorporation should:

- Address a single, fairly narrowly defined issue.
- Clearly identify if they are a bug fix or new functionality.
- Provide some context about what part(s) of the code are affected.
- Describe what the problem was, and why this is the best solution (I might ultimately disagree, but knowing the
  intent is really useful).

When to include code in PYME and when to write an extension module?
===================================================================

PYME now has a reasonably usable plugin system by which modules for VisGUI, dh5view, and recipes can live outside the
main repository and yet still be discovered and used by the core components. Developing plugins outside of the core
repository has the advantage that you don't need to wait on me, and also helps keep the overall codebase more
comprehensible. My suggestions are thus:

- If the new functionality can stand alone and only uses established plugin interfaces it is likely to be easier to
  develop outside the core codebase, at least initially. Should this be something of wider interest moving it to the
  core can be done at a later stage.
- If the new functionality requires modification to any of the core components, it should either be developed within
  the PYME codebase, or split into the development of a new interface within the core code, and a standalone component.
- I want to discourage long lasting 'forks' - i.e. if it affects the core, we should hopefully be able to merge
  relatively quickly so everyone is using the same core.
- One potentially attractive option would be to have an official 'PYME-plugins' repository which put plugins in a place
  where they can get easily distributed, but which could have substantially laxer stringency on what we accept.

Regardless of the approach taken, I'm keen to be involved as early in the process as possible.
