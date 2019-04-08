Contributing
************

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
but much remains to be done. At this point, PYME will build under python 3 and is currently passing 42 of 46 tests although
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
- Find an alternative to / port PyFFTW3 to python3

We will probably discover more as we progress.

Coding style
============

The coding style in PYME as it stands is fairly inconsistent. I'd like to improve on this. New code should:

- follow PEP8 where possible
- use Numpy style docstrings
- have strong model-view separation
- be written so as to be compatible with both python 2.7 and python > 3.6.

This **DOES NOT** mean that changes to existing variable names to follow PEP8 are welcome. Changes to existing variable
names incurs a significant proof-reading cost for very little benefit, adding significant "noise" to pull requests. A
personal aesthetic preference for a particular style does not justify changes to otherwise functioning code. The
exception here is classes and functions which form part of our "API" (i.e. might be called from user code / new code)
where a good case can be made for conversion to PEP8 in order to present a consistent interface. The bar for such API
changes however is high in order not to break existing code. Such changes:

- Should be discussed in advance (both to identify areas - potentially in projects other than the core PYME - where the API is
  used and also to decide if further refactoring / changes to the API should occur at the same time. There are multiple
  places in PYME where the API has evolved and might no longer be the most logical way of calling things. If we are
  putting the effort into changing all our calls, we might as well get the API right at the same time.)
- Should be in a separate PR which only deals with re-naming
- Must consider how the code might be used (e.g. several people use bits of PYME from ipython notebooks) and provide
  backwards compatible fallbacks (along with deprecation warnings) where appropriate

Pull requests
=============

Pull requests are always welcomed, but to increase the chances of speedy review and incorporation should:

- Address a single, fairly narrowly defined issue.
- Clearly identify if they are a bug fix or new functionality.
- Provide some context about what part(s) of the code are affected.
- Describe what the problem was, and why this is the best solution (I might ultimately disagree, but knowing the
  intent is really useful).

Pull requests which change existing variable names for the hell of it (see above) are likely to be rejected even if they
include useful new functionality or bugfixes as the maintenance burden of such changes is high.

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
