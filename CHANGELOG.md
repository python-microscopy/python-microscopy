# Changelog

This document records changes to PYME in the absence of a formal release cycle. It's goal is to
allow people to see what the minimum version of PYME is for a given feature / bugfix and also
to aid decisions on whether to upgrade.
It should be less granular than individual commits, but should include a line for each of:
    
- any major bugfixes
- any major new functionality
- api changes or deprecations
- tagged github releases (generally coincident with the generation of new pip and conda packages)

Entries should take the following format:

**version** - description of change

Where **version** is the PYME version number (i.e. the date in YY.MM.DD NZT) when the change was
merged into the main branch. YY.MM.(DD+1) is also acceptable to avoid timezone ambiguities. Tagged
releases for which both conda and pip packages have been generated should be formatted as section
headings. New entries should be added to the top of the changes section below.


## Changes

**20.07.08** - created CHANGELOG.md (changes before here are incomplete)
**20.07.07** - improved dsviewer plugin API - see PYME.DSView.modules.loadModule() docs

### Release 20.04.25 (note - conda only)