.. _localisationanalysis:

Analysing Localisation Microscopy data
**************************************

.. toctree::
   :maxdepth: 1

   AnalysingForeignData
   SplitterMulticolour

Starting distributed analysis infrastructure
============================================

To improve performance, PYME distributes localization analysis over multiple worker processes, with a server process
used for communication. These can either all be on the same machine, or be distributed across a network/cluster. The
server and worker processes need to be running before starting the analysis. This is achieved with the

.. code-block:: bash

   launchWorkers

command. On OSX or linux, enter this in a terminal window. On Windows either enter it in an *"Anaconda prompt"* (found
under the "Anaconda" group on the start menu, or navigate to the directory you installed Anaconda to (most likely
``C:\Users\<username>\Anaconda2`` or ``C:\Users\<username>\Miniconda2``) and make a shortcut to ``Scripts\launchWorkers.exe``
somewhere convenient for you.

This ``launchWorkers`` script starts the server and the same number
of workers as there are cores on the current machine [#numworkers]_. The server and worker processes should then find
each other automatically [#mdns]_ and communicate using a package called Pyro. On Windows you may be prompted to allow
python through the firewall, which you should accept.

.. note::

   We are currently in the process of migrating all our analysis to use the new, high-throughput infrastructure which is
   python3 compatible, faster, and more robust. This will mean that you need to run ``PYMEClusterOfOne`` (or similar),
   rather than ``launchWorkers`` but will leave the rest of the process largely unchanged.

Loading data
============

Once the server and worker processes are running, the data should be opened
using :ref:`dh5view <dh5view>`.

Data acquired using PYMEAcquire
+++++++++++++++++++++++++++++++

Data spooled to a ``.h5 file`` can be opened by running ``dh5view filename.h5``, by double clicking ``dh5view.exe`` in
the ``Anaconda\Scripts`` directory (or a shortcut) and using the file open dialog, or by ascociating ``dh5view`` with ``.h5``
files. For data saved directly to a queue, the easiest way is probably to click the **Analyse** button on the
Spooling panel in :ref:`PYMEAcquire <PYMEAcquire>`. Many protocols will do this
automatically after a the intial pre-bleaching phase has been performed.

Data not acquired using PYMEAcquire
+++++++++++++++++++++++++++++++++++
For data not originating from *PYMEAcquire* the process is a little more complex as ``dh5view`` will not detect that it
should launch in *"Localization Mode"* and some metadata will probably be missing.
(see :ref:`analysingforeigndata` for details). In short, either use ``dh5view -m LM filename`` or use the file open dialog,
complete any missing metadata entries and then choose ``LMAnalysis`` from the ``Modules`` menu.

Special attention is also needed for analysing simultaeneous multi-colour data (see :ref:`imagesplitter`).

With the data loaded in dh5view, one should see something like:

.. image:: /images/dh5view_lm.*


Analysis settings
=================

The **Analysis** and **Point Finding** panes in the left hand panel control the
analysis settings.

Fit types
+++++++++

PYME offers a large number of different fit types, and is easily extensible to support more.
The most useful ones are given below. If in doubt, you usually want **LatGaussFitFR**.

======================  ==============================================================
Type                    Description
======================  ==============================================================
**LatGaussFitFR**       Simple 2D gaussian fitting.
InterpFitR              Fits an interpolated PSF to localisation Data (**3D**)
SplitterFitFNR          Like LatGaussFitFR but for ratiometric data
SplitterFitInterpNR     Like InterpFitR but for ratiometric data (**3D**)
SplitterFitInterpBNR    Like SplitterFitInterpNR but for biplane ratiometric data (**3D**)
GaussMultiFitSR       2D Multi-emitter fitting
SplitterShiftEstFR      Used for estimating shift fields between channels
======================  ==============================================================


Settings
++++++++

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
Subtract background   Should we subtract the background before fitting (rather than just for event detection)
Debounce r            The radius (in pixels) within which 2 events cannot be reliably
                      distinguished
Shifts                The shift field to use for chromatic shift correction (only methods with
                      *Splitter* in their name.
PSF                   The PSF measurement to use (only *Interp* methods)
Track Fiducials       Whether to estimate drift using fiduciaries
Variance Map          An image containing a pixel by pixel map of camera variance (read noise) for
                      sCMOS correction
Dark Map              An image containing a pixel by pixel map of dark values for sCMOS correction
Flatfield Map:        An image used for flatfielding (gain correction) with sCMOS cameras
====================  ============================================================================

Some fit modules will also display custom settings.


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
clicking the **Go** button. The results will automatically be saved as ``.h5r`` files, either under the
``PYMEDATADIR`` directory (if the environment variable was set earlier), or in a directory
called ``PYMEData`` in the users home directory (``c:\\Users\\<username>\\`` under windows).

The resulting ``.h5r`` files can be opened in :ref:`VisGUI <VisGUI>`.


.. note::

   Distributing analysis over multiple computers
   =============================================

   A crude form of distributing the analysis over multiple computers (a small ad-hoc cluster) can be achieved by:

   * Make sure that the same version of PYME is installed on all machines
   * Run ``launchWorkers`` on each machine you want to use.

   A much better approach, however, is to use the 'new-style' distributed analysis which is both significantly faster and
   more robust. TODO - write docs.

.. rubric:: Footnotes

.. [#numworkers] If you're running the analysis on the data acquisition computer, and have a decent number of
   cores available (e.g. 8),  better performance can be achieved by reserving a core for
   each of the Acqusition and Server processes (ie limiting the number of workers to 6 in our
   8 core case). This can be done by explicitly specifying the number of workers to launch as
   an argument eg: ``launchWorkers 6``.

.. [#mdns] Using the zeroconf/MDNS protocol.
