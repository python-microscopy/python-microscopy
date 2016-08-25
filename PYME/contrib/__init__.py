"""
The contrib package consists of several 3rd party packages which I have chosen to include within the codebase rather than
as external dependencies. The rational for inclusion here could be 3-fold. a) simplifying packaging / code not available
as a stand alone package (e.g. `cpmath`, which is a small part of cell-profiler. the whole cell profiler package would
be quite a large dependency), b) have been modified (e.g. :py:mod:`PYME.contrib.gohlke.tiffile` or
:py:mod:`PYME.contrib.bcl`) or c) were snippets - e.g. `wxPlotPanel`.
"""