.. _localization_analysis:

# Analyzing Localization Microscopy data


.. toctree::
   :maxdepth: 1

   AnalysingForeignData
   SplitterMulticolour

## Starting analysis infrastructure

To improve performance, PYME distributes localization analysis over multiple worker processes, with server processes used for communication. These can all be on the same machine, or distributed across a network/cluster. They need to be started before running analysis. To run analysis
on a single machine (with all networking done internally on your computer), run

.. code-block:: bash

   PYMEClusterOfOne

in a terminal window or *"Anaconda prompt"*. The worker processes will start, and your web browser should open a tab from which you can view the status of any running analyses.

You may also wish to use our browser-based cluster user interface. It may decrease performance for cluster of one users, hower it offers more detailed status information and functionality to batch-run both localization analysis and analysis recipes. To start clusterUI, you currently need a :ref:`from-source <installationFromSource>` installation.
From a terminal or *"Anaconda prompt"* run

.. code-block:: bash

   python <path-to>/python-microscopy/PYME/cluster/clusterUI/manage.py runserver 0.0.0.0:9000

and point your web browser to http://127.0.0.1:9000/.

.. note::

   We have deprecated the `launchWorkers`-style localization infrastructure, as
   the new architecture is higher performance and compatible with both Python 2
   and Python 3. The `PYMEClusterOfOne` is for the most part a drop-in replacement. Still, legacy documentation for localization analysis can be found `here <_localisationanalysis>`.

### Distributing analysis over multiple computers
To distribute analysis over a computer cluster, see :ref:`cluster setup <cluster_install>`.

## Loading data

Once the the cluster (of one) is running, open raw blinking series with :ref:`dh5view <dh5view>`. If the data was acquired with :ref:`PYMEAcquire <PYMEAcquire>` and saved as `.h5` the localization analysis plugin should load automatically. Otherwise, click ``LMAnalysis`` from the ``Modules`` drop-down menu.

With the data loaded in dh5view, one should see something like:

.. image:: /images/dh5view_lm.*

### Localizing directly from PYMEAcquire

Localization can be initiated directly from PYMEAcquire. In the 'Time/ Blinking Series' panel on the right of PYMEAcquire, expand the 'Real-time analysis' section and click the 'Save analysis settings to metadata' checkbox. During acquisition, you can either click the 'Analyse' button in spooling progress section, or your acquisition protocol can do this automatically.

### Data not acquired using PYMEAcquire
Data which does not originate from PYMEAcquire will need to have appropriate metadata (see :ref:`analysingforeigndata` for details). In short, either use ``dh5view -m LM filename`` or use the file open dialog,
complete any missing metadata entries and then choose ``LMAnalysis`` from the ``Modules`` menu.

### Ratiometric multicolor data
Special attention is also needed for analysing simultaeneous multi-colour data (see :ref:`imagesplitter`).


# Analysis settings

The ``Analysis`` and ``Point Finding`` panes in the left-hand panel control the
analysis settings.

## Fit types

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
AstigGaussGPUFitFR      Fast astigmatic fitting for (**3D**) Requires [pyme-warp-drive](https://github.com/python-microscopy/pyme-warp-drive)
======================  ==============================================================


## Settings

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

# Starting the fitting

## Testing object detection

We use a singal-to-noise dependent threshold, which makes detection fairly robust
with a threshold factor near 1, but it is generally worth testing this by clicking the ``Test`` button. This performs the object finding step on the current frame. If this performs poorly, one should
check the ``Camera.ADOffset`` setting in the metadata (accessible through the ``Metadata`` tab), and the camera maps, if in use, and confirm they are correct before attempting to tweak the detection threshold. For some series, the *ADOffset* is estimated by taking a number of dark frames before the acquisition starts, which can be fooled if the room lights are on and/or the laser shutters are misbehaving.
Metadata parameters can be edited by right clicking the appropriate field in the Metadata tab.

## Launching the analysis tasks

Once satisfied with the event detection, the analysis proper can be started by
clicking the ``Go`` button. The results will automatically be saved as ``.h5r`` files, either under the ``PYMEDATADIR`` directory (if the environment variable was set earlier), or in a directory
called ``PYMEData`` in the users home directory (``c:\\Users\\<username>\\`` under windows).

The resulting ``.h5r`` files can be opened in :ref:`PYMEVisualize <VisGUI>`.
