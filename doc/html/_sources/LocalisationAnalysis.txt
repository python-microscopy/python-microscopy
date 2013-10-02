.. _localisationanalysis:

Analysing Localisation Microscopy data
**************************************

Distributed analysis and queues
===============================

PYME has a distributed analysis model whereby a server process manages *Task Queues*
and distributes groups of frames to
multiple different worker processes. These can either all be on the same machine, or be distributed
across a network/cluster. These server and worker processes need to be running before starting the analysis. The ``launchWorkers.py``
script simplifies this by starting the server and the same number of workers as there
are cores on the current machine. To communicate with each other the server and worker
processes use a package called Pyro, and a pyro nameserver needs to be running so workers and 
servers can find each other. If there is no name server running on the local network, launchWorkers
starts one, but this is probably only suitable/robust if only one machine is being used
for analysis. 

If you're running the analysis on the data acquisition computer, and have a decent number of
cores available (we have 8),  better performance can be achieved by reserving a core for
each of the Acqusition and Server processes (ie limiting the number of workers to 6 in our
8 core case). This can be done by explicitly specifying the number of workers to launch as
an argument eg: ``launchWorkers 6``.

Distributing over multiple computers
++++++++++++++++++++++++++++++++++++

Distributing the analysis over multiple computers (a small ad-hoc cluster) is now easy:

* Make sure a pyro nameserver is running somewhere on your network and that it is 
  bound to the external interface rather than localhost (see the `Pyro  documentation <http://packages.python.org/Pyro/5-nameserver.html>`_). If you don't explicity run a nameserver, the first copy of ``launchWorkers`` you start will run one for you. The caveat with this approach is that you shouldn't close this copy while you (or anyone else on your network segment) is doing analyisis, even on other computers.
* Run ``launchWorkers`` on each machine you want to use.

Loading data
============

Once the server and worker processes are running, the data should be opened
using :ref:`dh5view <dh5view>`. For data spooled to a ``.h5 file`` this can be
performed as one would expect, by either specifying the filename on the command
line or by ascociating ``dh5view`` with ``.h5`` files. For data saved directly to
a queue, the easiest way is probably to click the **Analyse** button on the
Spooling panel in :ref:`PYMEAcquire <PYMEAcquire>`. Many protocols will do this
automatically after a the intial pre-bleaching phase has been performed.

For data not originating from *PYMEAcquire* the process is a little more complex
(see :ref:`analysingforeigndata`).

Analysis settings
=================

With the data loaded in dh5view, one should see something like:

.. image:: /images/dh5view_lm.*

The **Analysis** and **Point Finding** panes in the left hand panel control the
analysis settings. The settings are:

====================  ============================================================================
Setting               Description
====================  ============================================================================
Threshold             The threshold for event detection. This is not a strict threshold, but
                      rather a scaling factor applied to a local threhold computed from the
                      estimated pixel SNR. Will generally not need to be altered (except for
                      some of the interpolation based fits)
Type                  The type of fit to perform
Interp                The type of interpolation to perform (only methods with *Interp* in
                      their name)
Start at              The frame # should to start our analysis at
Background            What range of frames should be used for background estimation. Uses
                      python slice notation.
Debounce r            The radius (in pixels) within which 2 events cannot be reliably
                      distinguished
Shifts                The shift field to use for chromatic shift correction (only methods with
                      *Splitter* in their name.
PSF                   The PSF measurement to use (only *Interp* methods
Estimate Drift        Whether to estimate drift using fiduciaries **[BROKEN]**
Subtract background   Should we subtract the background before fitting (rather than
                      just event detection)
====================  ============================================================================

Fit types
+++++++++

PYME offers a number of different fit types, and is easily extensible to support more.
The current ones are, with the ones you'd usually want to use in bold:

======================  ==============================================================
Type                    Description
======================  ==============================================================
ConfocCOIR              Determines a 3D COI from confocal/widefield data
Gauss3DFitR             Fits a 3D gaussian to confocal/widefield data
**InterpFitR**          Fits an interpolated PSF to localisation Data (**3D**)
LatFitCOIR              Determines the position of events by taking their centroid.
                        Fast but not as good as a proper fit.
**LatGaussFitFR**       Simple 2D gaussian fitting.
LatObjFindFR            Just perform the object finding part of the fitting process
LatPSFFitR              Fits a symplified model of a widefield PSF (**3D**). Use
                        InterpFitR instead
SplitterFitCOIR         Determines centroids when two channels are split onto
                        separate halves of the CCD
SplitterFitFR           Like LatGaussFitFR but for split data
**SplitterFitInterpR**  Like InterpFitR but for split data (**3D**)
**SplitterFitQR**       Faster version of SplitterFitFR (ommits background parameters)
SplitterObjFindR        Like LatObjFindFR, but for split data
SplitterShiftEstFR      Used for estimating shift fields
======================  ==============================================================

Starting the fitting
====================

Testing the object detection
++++++++++++++++++++++++++++

Whilst the threhold factor is fairly robust, it is generally worth testing the
detection by clicking the **Test** button. This performs the object finding step
on a selection of frames spaced throughout the sequence. If this fails one should
check the ``Camera.ADOffset`` setting in the metadata (accessible through the **Metadata** tab)
to see if this is reasonable before attempting to tweak the detection threshold. (The *ADOffset* is estimated by
taking a number of dark frames before the acquisition starts, and can be fooled if
the room lights are on and/or the laser shutters are misbehaving). Metadata parameters
can be edited by right clicking the appropriate field in the Metadata tab.

Launching the analysis tasks
++++++++++++++++++++++++++++

Once satisfied with the event detection, the analysis proper can be started by
clicking the **Go** button. The results will automatically be saved, either under the
``PYMEDATADIR`` directory (if the environment variable was set earlier), or in a directory
called ``PYMEData`` in the users home directory (``c:\\Users\\<username>\\`` under windows).

