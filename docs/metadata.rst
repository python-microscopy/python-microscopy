.. _metadata:

PYME Metadata
*************

PYME metadata is stored as a series of key-value pairs. By convention, these pairs are hierarchical, with a high level
category separated from a lower level parameter by a dot.

.. note::

    The metatdata was formally stored within a hierarchical structure (and still is within HDF files), and can be accessed
    in a hierarchical fashion using some metadata handlers. We are slowly moving to a flatter internal representation,
    with the hierarchical nature being by convention only. You are nevertheless encouraged to follow this convention when
    defining new keys.

Programmatic interaction with metadata is accomplished using the classes in :mod:`PYME.IO.MetaDataHandler` module. A
number of on disk storage formats are supported, as outlined below. The `.md` format has comments describing some of the
key fields.

JSON metadata
-------------

This is the format recommended for any new usage. A `.json` file with the same base file name (the bit excluding the
extension) as an image file will be automatically parsed and used as metadata. This is the most restrictive form of
metadata (it only supports metadata values which can be easily converted to json, either by consisting of simple python
types, or by implementing a to_JSON method, whereas older metadata formats support embedding arbitrary python objects as
pickles).

An example .json metadata file is shown below:

.. code-block:: json

    {
    "AcquiringUser": "David_Baddeley",
    "Analysis.BGRange": [-50, 0],
    "Analysis.DataFileID": 624268635,
    "Analysis.DebounceRadius": 4,
    "Analysis.DetectionThreshold": 0.7,
    "Analysis.FitModule": "SplitterFitFNR",
    "Analysis.PCTBackground": 0.0,
    "Analysis.subtractBackground": true,
    "Camera.ADOffset": 251.0,
    "Camera.CycleTime": 0.051750000566244125,
    "Camera.EMGain": 90,
    "Camera.ElectronsPerCount": 4.99,
    "Camera.IntegrationTime": 0.05000000074505806,
    "Camera.Model": "DU897_BV",
    "Camera.Name": "Andor IXon DV97",
    "Camera.NoiseFactor": 1.41,
    "Camera.ROIHeight": 511,
    "Camera.ROIPosX": 1,
    "Camera.ROIPosY": 1,
    "Camera.ROIWidth": 511,
    "Camera.ReadNoise": 88.1,
    "Camera.SerialNumber": 7863,
    "Camera.StartCCDTemp": -70,
    "Camera.TrueEMGain": 35.50129002714877,
    "EndTime": 1415229790.515,
    "EstimatedLaserOnFrameNo": 201,
    "NikonTi.FilterCube": "647LP",
    "NikonTi.LightPath": "L100",
    "Positioning.PIFoc": 166.1875,
    "Positioning.Stage_X": 12.527,
    "Positioning.Stage_Y": 13.0805,
    "Protocol.BleachFrames": [61,200],
    "Protocol.DarkFrameRange": [0,20],
    "Protocol.DataStartsAt": 201,
    "Protocol.Filename": "prebleach642.py",
    "Protocol.PrebleachFrames": [21,58],
    "Sample.Creator": "Kenny",
    "Sample.Labelling": [["TOM20", "A647"],["alpha-tubulin","Cy3B"]],
    "Sample.SlideRef": "5_11_2014_A",
    "Splitter.Channel0ROI": [0,0,256,512],
    "Splitter.Channel1ROI": [256,0,256,512],
    "Splitter.Dichroic": "FF700-Di01",
    "Splitter.Flip": false,
    "Splitter.TransmittedPathPosition": "Left",
    "StackSettings.EndPos": 176.0875,
    "StackSettings.NumSlices": 100,
    "StackSettings.ScanMode": "Middle and Number",
    "StackSettings.ScanPiezo": "PIFoc",
    "StackSettings.StartPos": 156.2875,
    "StackSettings.StepSize": 0.2,
    "StartTime": 1415229470.461,
    "chroma.ShiftFilename": "G:\\Data\\David_Baddeley\\5_11_series_A.sf",
    "chroma.dx": null,
    "chroma.dy": null,
    "imageID": 624268635,
    "voxelsize.units": "um",
    "voxelsize.x": 0.07,
    "voxelsize.y": 0.07,
    "voxelsize.z": 0.2
    }


.md Metadata files
------------------

This is a slightly older metadata format, and is essentially a python script  with a `.md` extension which is interpreted
to fill a :class:`MetaDataHandler <PYME.IO.MetaDataHandler.MDHandlerBase>`  object. You can do a lot with it, but, as it
is a python script, the normal security implications apply. Like the `.json` metadata files, PYME will search for a `.md`
file with the same base name as an image file and use this as metadata.

an example `.md` file is shown below:

.. code-block:: python

    #PYME Simple Metadata v1
    md['AcquiringUser'] = 'David_Baddeley'               # who sat at the microscope to take the image

    # Protocol fields are written by the acquisition protocol and generally describe changes in laser powers etc ...
    # They are pretty specialized to our acquisition software.
    md['Protocol.DataStartsAt'] = 201   # frame where we expect good quality data to start. IE where to start localizing molecules
    md['Protocol.PrebleachFrames'] = (21, 58)            # a range of frames taken at low laser power prior to bleaching (for generation of a widefield image)
    md['Protocol.Filename'] = u'prebleach642.py'
    md['Protocol.DarkFrameRange'] = (0, 20)
    md['Protocol.BleachFrames'] = (61, 200)

    # Parameters of an image splitting device (e.g. optosplit or custom built) used for ratiometric multi-colour or biplane imaging
    md['Splitter.Channel0ROI'] = [0, 0, 256, 512]
    md['Splitter.Dichroic'] = 'FF700-Di01'
    md['Splitter.Channel1ROI'] = [256, 0, 256, 512]
    md['Splitter.TransmittedPathPosition'] = 'Left'
    md['Splitter.Flip'] = False

    md['NikonTi.LightPath'] = 'L100'
    md['NikonTi.FilterCube'] = u'647LP'

    # information about a measured vectorial shift field used to correct for chromatic aberrations between channels
    # note that specifying the filename should be sufficient (the other fields will then be automatically completed)
    md['chroma.dx'] = pickle.loads('''ccopy_reg\n_reconstructor\np1\n(cscipy.interpolate.fitpack2\nSmoothBivariateSpline\np2\nc__builtin__\nobject\np3\nNtRp4\n(dp5\nS'fp'\np6\nF1746.7117511335171\nsS'degrees'\np7\n(I3\nI3\ntp8\nsS'tck'\np9\n(cnumpy.core.multiarray\n_reconstruct\np10\n(cnumpy\nndarray\np11\n(I0\ntS'b'\ntRp12\n(I1\n(I8\ntcnumpy\ndtype\np13\n(S'f8'\nI0\nI1\ntRp14\n(I3\nS'<'\nNNNI-1\nI-1\nI0\ntbI00\nS'\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@'\ntbg10\n(g11\n(I0\ntS'b'\ntRp15\n(I1\n(I8\ntg14\nI00\nS'%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@'\ntbg10\n(g11\n(I0\ntS'b'\ntRp16\n(I1\n(I16\ntg14\nI00\nS"\xa9\xafZ\xc2\xb75^@Z\xf5\x99]\x91*<@N\x12\xca|\xe1`\xff\xbfs\x15\x87a\xc2\x9bZ@\xa7K\x7f\xff(\xdaM@ \xa0\xe7\xa3\x1e\xe0C\xc0U\x89\xc0\x85.\x1dB\xc0\xb8:\x95\xb7\xd0\xc14@v\xca\x19\x06\xe1\x11J@\xb8';\xea\x1e\xd6G\xc0P\xa8e\x9d\xf6\xfaF\xc0\x93\x89\xa2\xd5\xe5.;@\xbf\xe6\xf2\xc2\x8f\xe0_@\x95\x054\x1a\xca\x9f=@\xf9(\x86\xba\xc93'@`\xcc0\xa6t\xafR@"\ntbtp17\nsb.''')
    md['chroma.ShiftFilename'] = u'G:\\Data\\David_Baddeley\\5_11_series_A.sf'
    md['chroma.dy'] = pickle.loads('''ccopy_reg\n_reconstructor\np1\n(cscipy.interpolate.fitpack2\nSmoothBivariateSpline\np2\nc__builtin__\nobject\np3\nNtRp4\n(dp5\nS'fp'\np6\nF661.02649656893755\nsS'degrees'\np7\n(I3\nI3\ntp8\nsS'tck'\np9\n(cnumpy.core.multiarray\n_reconstruct\np10\n(cnumpy\nndarray\np11\n(I0\ntS'b'\ntRp12\n(I1\n(I8\ntcnumpy\ndtype\np13\n(S'f8'\nI0\nI1\ntRp14\n(I3\nS'<'\nNNNI-1\nI-1\nI0\ntbI00\nS'\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@\x98k\xb0o\xac\xc80@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@!\x06\xbf\xe1\x13\xb7\xd0@'\ntbg10\n(g11\n(I0\ntS'b'\ntRp15\n(I1\n(I8\ntg14\nI00\nS'%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@%\xcc?\xd3\x8e\xdfs@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@\xea\x88\xaeu\xd4\xb8\xdf@'\ntbg10\n(g11\n(I0\ntS'b'\ntRp16\n(I1\n(I16\ntg14\nI00\nS'gU\x00\xe6?\xa5\xf0?\xe7~\xe4rn\xa76\xc0IH.d*\xa8A\xc03/\x9ap\xe5\x8aT\xc0\xa4\x82\'\x8e\x1bpA\xc0_{\xd7\xc8\xbdsF\xc0)+M\xe2\xfd\x012\xc0\xa8\x858\n\xcb\x07M\xc0\xb4L\xbd\x1dB\x80U\xc0>p\x12\xeb<\xedE\xc0\xae\x81\xf73\x0c\xd5A\xc0\x8b\xaf|\x84m\x1c&\xc0\xe0M\x1b\x98\xb0\xc4^\xc0R\xba\x9f"-\x9cO\xc0\xffD\x11\x952\x8a7\xc0\x11\xa6\xc5U\x8eL!@'\ntbtp17\nsb.''')

    # these are the settings which would have been used if we were taking a stack. They are always recorded, whether or not
    # a stack is acquired (i.e. this data will still be here for 2D images, but is not meaningful. Stack acquisitions
    # will have ProtocolFocus events in their events data.
    md['StackSettings.ScanMode'] = 'Middle and Number'
    md['StackSettings.StepSize'] = 0.20000000000000001
    md['StackSettings.EndPos'] = 176.08750000000001
    md['StackSettings.NumSlices'] = 100
    md['StackSettings.StartPos'] = 156.28749999999999
    md['StackSettings.ScanPiezo'] = 'PIFoc'

    # Where the various stages are, in their native units: TODO these should really all be in um
    md['Positioning.Stage_X'] = 12.526999999999999
    md['Positioning.Stage_Y'] = 13.080500000000001
    md['Positioning.PIFoc'] = 166.1875

    # settings for the analysis. Added to the raw data if doing real time streaming analysis, otherwise only present in
    # the analysed data
    md['Analysis.DetectionThreshold'] = 0.69999999999999996
    md['Analysis.PCTBackground'] = 0.0
    md['Analysis.DebounceRadius'] = 4
    md['Analysis.DataFileID'] = 624268635
    md['Analysis.BGRange'] = [-50, 0]
    md['Analysis.FitModule'] = 'SplitterFitFNR'
    md['Analysis.subtractBackground'] = True

    # a unique ID ascociated with the RAW image suitable for use as e.g. a public key. This is generated either by taking
    # a hash of the part of the first frame data, or by computing a starting timestamp along with a random hash.
    md['imageID'] = 624268635

    # metadata about the sample. This is written into an SQL database on acquisition. The database schema is significantly
    # more structured (especially for the labelling) than the summary data written here, and also includes e.g. species
    # and strain info.
    md['Sample.Labelling'] = [(u'TOM20', u'A647'), (u'alpha-tubulin', u'Cy3B')]
    md['Sample.SlideRef'] = u'5_11_2014_A'
    md['Sample.Creator'] = u'Kenny'

    # Information about the camera
    md['Camera.ROIHeight'] = 511
    md['Camera.ROIPosY'] = 1
    md['Camera.ROIPosX'] = 1
    md['Camera.CycleTime'] = 0.051750000566244125         # time between consecutive frames, in seconds
    md['Camera.EMGain'] = 90                              # the EM Gain setting (gain register 8 bit voltage value)
    md['Camera.ADOffset'] = 251.0                         # the AD offset (zero point) in counts / ADUs
    md['Camera.TrueEMGain'] = 35.501290027148769          # the calibrated EM gain (this is the important one)
    md['Camera.StartCCDTemp'] = -70
    md['Camera.SerialNumber'] = 7863
    md['Camera.ROIWidth'] = 511
    md['Camera.ElectronsPerCount'] = 4.9900000000000002
    md['Camera.Model'] = 'DU897_BV'
    md['Camera.ReadNoise'] = 88.099999999999994           # the read noise in photoelectrons (e-)
    md['Camera.NoiseFactor'] = 1.4099999999999999         # the excess noise factor for the multiplcation step 1.4 for EMCCD, 1.0 for sCMOS
    md['Camera.IntegrationTime'] = 0.05000000074505806
    md['Camera.Name'] = 'Andor IXon DV97'

    md['StartTime'] = 1415229470.461          # the start time as a unix style timestamp

    md['EstimatedLaserOnFrameNo'] = 201       # synonymous with and redundant to 'Protocol.DataStartsAt'. Added automatically during analysis.

    md['EndTime'] = 1415229790.5150001        # the end time as a unix style timestamp

    md['voxelsize.units'] = 'um'           # NB voxelsizes should ALWAYS be in um. The units specified here are ignored
    md['voxelsize.x'] = 0.070000000000000007
    md['voxelsize.z'] = 0.20000000000000001
    md['voxelsize.y'] = 0.070000000000000007


XML metadata
------------

We also support an XML version of the metadata. This is principally used when embedding PYME metadata (as a structured
annotation) into OME metadata when producing OME TIFFs. It can also be used in a standalone fashion like the `.json` and
`.md` formats, but this usage is discouraged.


HDF metadata
------------

This is the original hierarchical on disk format and uses HDF groups for the category labels, and pytables attributes
to those groups as the individual keys. This is likely to be poorly supported when using any library other than
pytables to access the HDF file, although does seem to be readable using the standard `h5ls` and `h5dump` tools (see
example below) with the exception of a few fields which are stored as python pickles:

.. code-block:: bash

    DB3:~ david$ h5ls -rv /Users/david/PYMEData/david/2016_11_30/30_11_series_A.h5/MetaData
    Opened "/Users/david/PYMEData/david/2016_11_30/30_11_series_A.h5" with sec2 driver.
    /Camera                  Group
        Attribute: ADOffset scalar
            Type:      native double
            Data:  967
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: CycleTime scalar
            Type:      native double
            Data:  0.1
        Attribute: EMGain scalar
            Type:      native long
            Data:  0
        Attribute: ElectronsPerCount scalar
            Type:      native double
            Data:  27.32
        Attribute: IntegrationTime scalar
            Type:      native double
            Data:  0.1
        Attribute: Name scalar
            Type:      23-byte null-terminated ASCII string
            Data:  "Simulated EM CCD Camera"
        Attribute: NoiseFactor scalar
            Type:      native double
            Data:  1.41
        Attribute: ROI scalar
            Type:      23-byte null-terminated ASCII string
            Data:  "(I0\nI0\nI1024\nI256\ntp1\n."
        Attribute: ROIHeight scalar
            Type:      native long
            Data:  256
        Attribute: ROIPosX scalar
            Type:      native long
            Data:  0
        Attribute: ROIPosY scalar
            Type:      native long
            Data:  0
        Attribute: ROIWidth scalar
            Type:      native long
            Data:  1024
        Attribute: ReadNoise scalar
            Type:      native double
            Data:  109.8
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:6528
        Links:     1
    /Lasers                  Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:8264
        Links:     1
    /Lasers/l405             Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: MaxPower scalar
            Type:      native double
            Data:  1000
        Attribute: On scalar
            Type:      native 8-bit field
            Data:  0x00
        Attribute: Power scalar
            Type:      native long
            Data:  10
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:9136
        Links:     1
    /Lasers/l488             Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: MaxPower scalar
            Type:      native double
            Data:  1000
        Attribute: On scalar
            Type:      native 8-bit field
            Data:  0x00
        Attribute: Power scalar
            Type:      native long
            Data:  10
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:10520
        Links:     1
    /Positioning             Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: Fake_x_piezo scalar
            Type:      native double
            Data:  5
        Attribute: Fake_y_piezo scalar
            Type:      native double
            Data:  5
        Attribute: Fake_z_piezo scalar
            Type:      native double
            Data:  50
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Attribute: x scalar
            Type:      native double
            Data:  5
        Attribute: y scalar
            Type:      native double
            Data:  5
        Attribute: z scalar
            Type:      native double
            Data:  50
        Location:  1:5248
        Links:     1
    /Protocol                Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: DarkFrameRange scalar
            Type:      13-byte null-terminated ASCII string
            Data:  "(I0\nI20\ntp1\n."
        Attribute: DataStartsAt scalar
            Type:      native long
            Data:  21
        Attribute: Filename scalar
            Type:      11-byte null-terminated UTF-8 string
            Data:  "simul488.py"
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:80288
        Links:     1
    /StackSettings           Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: EndPos scalar
            Type:      native double
            Data:  59.9
        Attribute: NumSlices scalar
            Type:      native long
            Data:  100
        Attribute: ScanMode scalar
            Type:      17-byte null-terminated ASCII string
            Data:  "Middle and Number"
        Attribute: ScanPiezo scalar
            Type:      1-byte null-terminated ASCII string
            Data:  "z"
        Attribute: StartPos scalar
            Type:      native double
            Data:  40.1
        Attribute: StepSize scalar
            Type:      native double
            Data:  0.2
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Location:  1:3648
        Links:     1
    /voxelsize               Group
        Attribute: CLASS scalar
            Type:      5-byte null-terminated ASCII string
            Data:  "GROUP"
        Attribute: TITLE scalar
            Type:      1-byte null-terminated ASCII string
            Data:  ""
        Attribute: VERSION scalar
            Type:      3-byte null-terminated ASCII string
            Data:  "1.0"
        Attribute: units scalar
            Type:      2-byte null-terminated ASCII string
            Data:  "um"
        Attribute: x scalar
            Type:      native double
            Data:  0.07
        Attribute: y scalar
            Type:      native double
            Data:  0.07
        Attribute: z scalar
            Type:      native double
            Data:  0.2
        Location:  1:11576
        Links:     1

We are considering replacing this with a somewhat more portable and simple implementation to make it easier to read in
3rd party clients.

Read-only support of 3rd party metadata
---------------------------------------

We can read limited metadata (e.g. voxelsize) from OME TIFFS, Zeiss .lsm, and anything we open using bioformats