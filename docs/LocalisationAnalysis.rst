.. _localisationanalysis:

Analysing Localisation Microscopy data
**************************************

.. toctree::
   :maxdepth: 1

   AnalysingForeignData
   SplitterMulticolour

Starting analysis infrastructure
================================

To improve performance, PYME distributes localization analysis over multiple worker processes, with a server process
used for communication. These can either all be on the same machine, or be distributed across a network/cluster. The
server and worker processes need to be running before starting the analysis. To launch all components on a single machine
, launch ``PYME>PYMEClusterOfOne`` from the start menu (Windows), or run

.. code-block:: bash

   PYMEClusterOfOne

in a terminal window (OSX, Linux) or *"Anaconda prompt"* (Windows) [#anacondaprompt]_ .On Windows you may be prompted to allow
python through the firewall, which you should accept.

.. note::

   We previously used a slightly different analysis architecture, launched with the ``launchWorkers`` command. If you are familiar with
   ``launchWorkers``, ``PYMEClusterOfOne`` should be a drop in replacement. The most noticeable differences will be a different
   task monitoring window, and that analysis results now go in an ``analysis`` subdirectory of the image directory
   rather than a higher level analysis directory. We've done a reasonable ammount of testing, but if something doesn't
   work ``launchWorkers`` is still available (for now, python 2.7 only). Please also let us know so we can fix it.

.. note::

   To distribute analysis over a computer cluster, see :ref:`cluster setup <cluster_install>`.

Loading data
============

Once the the cluster (of one) is running, open raw blinking series with :ref:`dh5view <dh5view>`.

If the data was acquired with :ref:`PYMEAcquire <PYMEAcquire>` and saved as `.h5` the localization analysis plugin should
load automatically. Otherwise, select ``LMAnalysis`` from the ``Modules`` drop-down menu to activate *"Localisation Mode"* [#lmmode]_.

With the data loaded in dh5view, one should see something like:

.. image:: /images/dh5view_lm.*


Data not acquired using PYMEAcquire
-----------------------------------
In addition to requiring manual activation of *"Localisation Mode"*, data not originating from *PYMEAcquire* will require
values for various camera parameters to be entered in the metadata (see :ref:`analysingforeigndata` for details).

Ratiometric multicolor data
---------------------------
Special attention is also needed for analysing simultaeneous multi-colour data (see :ref:`imagesplitter`).



Analysis settings
=================

The **Analysis** and **Point Finding** panes in the left hand panel control the
analysis settings.

Fit types
---------

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
GaussMultiFitSR         2D Multi-emitter fitting
SplitterShiftEstFR      Used for estimating shift fields between channels
AstigGaussGPUFitFR      Fast astigmatic fitting (**3D**) Requires `pyme-warp-drive <https://github.com/python-microscopy/pyme-warp-drive>`_.
======================  ==============================================================


Settings
--------

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
Start at              The frame # should to start our analysis at (zero-indexed)
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

Testing object detection
------------------------

We use a signal-to-noise dependent threshold, which makes detection fairly robust with a threshold factor near 1 (if
that the metadata is correct). It is nevertheless worth testing the detection by clicking the **Test** button, especially
when looking at data from a new microscope or type of sample. This performs the object finding step
on the current frame. If this performs poorly, one should check the ``Camera.ADOffset`` setting in the metadata (accessible
through the **Metadata** tab) to see if this is reasonable before attempting to tweak the detection threshold. *ADOffset*
is defined as the average dark value on the camera. We recommend imaging protocols take a number of dark frames before the
turning the laser on, and will use these frames, if present, to estimate *ADOffset*. This estimation can be fooled if
the room lights are on and/or the laser shutters are misbehaving). Incorrect camera maps (sCMOS), or read noise calibrations
can also lead to poor detection. Metadata parameters can be edited by right clicking the appropriate field in the Metadata tab.

Launching the analysis tasks
----------------------------

Once satisfied with the event detection, the analysis proper can be started by
clicking the **Go** button. The results will automatically be saved as ``.h5r`` files, either under the
``PYMEDATADIR`` directory (if the environment variable was set earlier), or in a directory
called ``PYMEData`` in the users home directory (``c:\\Users\\<username>\\`` under windows).

The resulting ``.h5r`` files can be opened in :ref:`PYMEVisualize <visgui>`.


Localizing directly from PYMEAcquire
====================================

Localization can be initiated directly from PYMEAcquire. In the 'Time/ Blinking Series' panel on the right of PYMEAcquire,
expand the 'Real-time analysis' section and click the 'Save analysis settings to metadata' checkbox. During acquisition,
you can either click the 'Analyse' button in spooling progress section, or your acquisition protocol can do this automatically.


Batching analysis
=================

To batch-run analysis of multiple series, launch the full ``clusterUI`` webapp by running ``PYMEClusterOfOne --clusterUI=True``.
This will require the optional dependency django (See also :ref:`cluster setup <cluster_install>`).


.. rubric:: Footnotes

.. [#anacondaprompt] Found under the "PYME" group on the start menu, if you used the installer, otherwise under the "Miniconda"
   or "Anaconda" group.

.. [#numworkers] If you're running the analysis on the data acquisition computer, and have a decent number of
   cores available (e.g. 8),  better performance can be achieved by reserving a core for
   each of the Acquisition and Server processes (ie limiting the number of workers to 6 in our
   8 core case). This can be done by explicitly specifying the number of workers to launch as
   an argument eg: ``launchWorkers 6``.

.. [#lmmode] Instead of manually loading the LMAnalysis module every time you launch dh5view, you can force it to start
   in localisation mode with the ``-m LM`` command line switch - i.e. ``dh5view -m LM filename``

.. [#mdns] Using the zeroconf/MDNS protocol.
