# Changelog

This document records changes to PYME in the absence of a formal release cycle. It's goal is to allow people to see what
the minimum version of PYME is for a given feature / bugfix and also to aid decisions on whether to upgrade.
It should be less granular than individual commits, but should include a line for each of:
    
- any major bugfixes
- any major new functionality
- api changes or deprecations
- tagged github releases (generally coincident with the generation of new pip and conda packages)

Entries should take the following format:

<pre>
**version** - description of change
</pre>

Where **version** is the PYME version number (i.e. the date in YY.MM.DD NZT) when the change was merged into the main
branch. YY.MM.(DD+1) is also acceptable to avoid timezone ambiguities. The important thing here is that 
`conda install python-microcopy >= version` (similarly for pip) should get the change (or fail if packages have not yet 
been built containing the change). Tagged releases for which both conda and pip packages have been generated should be
formatted as section headings. New entries should be added to the top of the changes section below.

Entries for new functionality should be added when that functionality is (mostly) ready to use and not likely to 
disappear again or have it's API change massively. It is the nature of our continuous release process that early 
iterations of some features might make it into the main branch before they are strictly ready for primetime. 

For PRs / features added in branches it is important that the version reported here is the version at which that PR/branch
is merged into **main**. When submitting a PR, the required changes to this file can either be submitted as part of the 
PR, but with a <pre>**FIXVERSION**</pre> placeholder for the version, or as a separate PR after the PR has been merged. 
The intention of this file is to be easier to parse than the PR / commit history and not all PRs will require an update here.

## Changes

**20.07.08** - created CHANGELOG.md (changes before here are incomplete)

**20.07.07** - improved dsviewer plugin API - see PYME.DSView.modules.loadModule() docs

### Release 20.04.25 (conda only)
