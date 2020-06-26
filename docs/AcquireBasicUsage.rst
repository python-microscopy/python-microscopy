Using PYME Acquire
******************

The camera & preview
====================

PYMEAquire will start with the camera running in a live preview mode. The mapping
of the (high bit depth) camera data to the display is controlled by the display widget.
By default this auto-scales the available dynamic range  (max value - min value)
in the image to 8 bits for display. This auto-scaling behaviour can be disabled by
unchecking the **Auto** option. When auto-scaling is disabled, the scaling can be
adjusted by dragging the red reference lines on the histogram, or by clicking
**Optimise** to achieve a once off min-max scaling. For historical reasons this
display assumes a maximum camera dynamic range of 12 bits and with cameras possesing
a higher dynamic range it is possible to saturate the display before the camera
saturates. The display widget also allows the display magnification to be changed.

Integration time
++++++++++++++++++++

The camera integration time can be set using the integration time slider or the 
accompanying combobox.

Region of interest
++++++++++++++++++

A region of interest can be selected by using the mouse to select an area in the
preview window and then selecting **Controls > Camera > Set ROI** from the menu,
or by using the ascociated keyboard shortcut (**F8**). The ROI can then be cleared
through the menu, or by pressing **F8** again.

Advanced camera properties
++++++++++++++++++++++++++

Some cameras (eg the Andor EMCCDs) will cause an additional pane of advanced
settings to be shown which allow, for example, the EM gain end TE cooling temperature
to be set. A not completely obvious aspect of the acquisition software is that it
will run the camera in *continuous* mode (ie letting the camera acquire frames one
after the other and stream them into a buffer). This has the advantage of allowing
the higest possible frame rates, but means that the camera frames and any hardware
actions are essentially asynchronous. If syncronised motion is required, the camera
can be placed into *single shot* mode whereby control returns to the software after
each frame. For the Andor camera, this is achieved in the **Advanced** section of the
EMCCD controls. This is probably advisable when doing, for example, conventional
z-stacks. Z-stacks in protocols avoid the syncronisation problem by recording a
timestamp of when the movement occured and calculating, post-hoc, which frame it
corresponds to.

Multiple camera support
+++++++++++++++++++++++

PYMEAcquire has some basic support for multiple cameras. If multiple cameras are
defined in the initialisation file, a control will be shown permitting switching
between the cameras.

Acquiring data
==============

Acquiring single images
+++++++++++++++++++++++

Single images (snapshots) may be acquired by either selecting **Acquire > 1 pic**
or by pressing **F5**. These can be saved as ``.tif``, ``.h5``, or ``.npy`` (see :ref:`DataFormats`)

Z-Stacks
++++++++

3D image stacks such as used for conventional widefield microscopy and, for example
PSF measurements can be obtained using the control at top right. Again, the options are ``.tif``, ``.h5``, or ``.npy``

Streaming/Spooling
++++++++++++++++++

Streaming functionality (the Spooling panel) is what is typically used for localisation
microscopy, and permits data to be streamed either to disk in ``.h5`` format or
to a server process (default) for real time analysis. If you want to save directly to
file and analyse later, the **Save to Queue** option needs to be unchecked. Filenames
are automatically generated from the current date and user name (although can be
changed manually). To access some of the controls (eg 'Spool Directory' and 'Save
to Queue' you might need to either drag the 'Hardware' controls down, or unpin the
Z-Stack controls.

Before starting streaming you can choose a :ref:`protocol <Protocols>`, which defines
actions to be performed during the sequence. You can then choose to acquire a simple
sequence (by clicking on **Series**), or one in which the focus is stepped at regular
intervals (**Z-Series**). The Z-Series uses the z-stepping regime defined using the
Z-stack control.

