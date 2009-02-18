import numpy
from PYME.Acquire.MetaDataHandler import NestedClassMDHandler, HDFMDHandler

#class VoxelSize:
#	def __init__(self, x, y, z, unit='um'):
#		self.x = x
#		self.y = y
#		self.z = z
#		self.unit = unit
#
#class CCDMetaDataIXonDefault:
#	#all at 10Mhz, e.m. amplifier
#	name = 'Andor IXon DV97'
#	ADOffset = 1100 #counts (@ T ~ -70, em gain ~ 150)
#	ReadNoise = 109.8 #electrons
#	noiseFactor = 1.41
#	electronsPerCount = 27.32
#
#
#class MetaData:
#	def __init__(self, voxSize, CCDMD = None):
#		self.voxelsize = voxSize
#		self.CCD = CCDMD
#
##FIXME - THIS SHOULD ALL BE EXTRACTED FROM LOG FILES OR THE LIKE
#TIRFDefault = MetaData(VoxelSize(0.07, 0.07, 0.2), CCDMetaDataIXonDefault())

TIRFDefault = NestedClassMDHandler()

#voxelsize
TIRFDefault.setEntry('voxelsize.x',0.07)
TIRFDefault.setEntry('voxelsize.y',0.07)
TIRFDefault.setEntry('voxelsize.z',0.2)
TIRFDefault.setEntry('voxelsize.units','um')

#camera properties - for Andor camera - see above
#TIRFDefault.setEntry('Camera.ADOffset',1100)
TIRFDefault.setEntry('Camera.ReadNoise',109.8)
TIRFDefault.setEntry('Camera.NoiseFactor',1.41)
TIRFDefault.setEntry('Camera.ElectronsPerCount',27.32)
TIRFDefault.setEntry('Camera.TrueEMGain',20) #mostly use gain register setting of 150 - this will hopefully be overwitten

#def genMetaDataFromHDF(h5File):
#    md = TIRFDefault
#
#    if 'MetaData' in h5File.root: #should be true the whole time
#        mdh5 = HDFMDHandler(h5File)
#        md.copyEntriesFrom(mdh5)
#
#    #once we start putting everything into data file properly,
#    #grab the relavant bits here
#
#    #Guestimate when the laser was turned on
#
#    #Laser turn on will result in large spike - look at derivative
#    dI = numpy.diff(h5File.root.ImageData[:200, :,:].mean(2).mean(1))
#
#    tLon = numpy.argmax(dI)
#    print tLon
#    dIm = dI[tLon]
#
#    #Now do some sanity checks ...
#    if tLon > 100: #shouldn't bee this late
#        tLon = 0
#        #expect magnitude of differential to be much larger at turn on than later ...
#    elif not dIm > 10*dI[(tLon + 50):(tLon + 100)].max():
#        tLon = 0 # assume laser was already on
#    #also expect the intensity to be decreasing after the initial peak faster than later on in the piece
#    elif not dI[(tLon + 5):(tLon + 15)].mean() < -abs(dI[(tLon + 50):(tLon + 100)].mean()):
#        tLon = 0
#
#    md.setEntry('EstimatedLaserOnFrameNo', tLon)
#
#        #Estimate the offset during the dark time before laser was turned on
#        #N.B. this will not work if other lights (e.g. room lights, arc lamp etc... are on
#    if not tLon == 0:
#        md.setEntry('Camera.ADOffset', numpy.median(h5File.root.ImageData[:max(tLon - 1, 1), :,:].ravel()))
#    else: #if laser was on to start with our best estimate is at maximal bleaching where the few molecules that are still on will hopefully have little influence on the median
#        md.setEntry('Camera.ADOffset',numpy.median(h5File.root.ImageData[-5:, :,:].ravel()))
#        print '''WARNING: No clear laser turn on signature found - assuming laser was already on
#                 and fudging ADOffset Estimation'''
#
#    if not 'MetaData' in h5File.root:
#        #something went wrong - should only happen for very early (mid 2008) .h5 files or for files taken
#        #in last couple of weeks of Jan 2009
#
#        import wx
#        if not None == wx.GetApp():
#            wx.MessageBox('Guessing that the EM Gain setting was 150 - if this is wrong fix in console', 'ERROR: No metadata fond in file ...', wx.ERROR|wx.OK)
#
#        print 'ERROR: No metadata fond in file ... Guessing that the EM Gain setting was 150'
#
#        md.setEntry('Camera.TrueEMGain', 20)
#
#        return md
#
#        #Quick hack for approximate EMGain for gain register settings of 150 & 200
#
#        #FIXME to use a proper calibration
#	if 'Camera.EMGain' in md.getEntryNames():
#		if md.Camera.EMGain == 200: #gain register setting
#			md.setEntry('Camera.TrueEMGain', 100) #real gain @ -50C - from curve in performance book - need proper calibration
#		elif md.Camera.EMGain == 150: #gain register setting
#			md.setEntry('Camera.TrueEMGain', 20) #real gain @ -50C - need proper calibration
#	else: #early file, or from camera without EMGain - assume early file - gain was usually 150
#		md.setEntry('Camera.TrueEMGain', 20)
#
#
#	if 'Camera.Name' in md.getEntryNames(): #new improved metadata
#		if md.Camera.Name == 'Simulated Standard CCD Camera': #em gain for simulated camera is _real_ em gain rather than gain register setting
#			md.Camera.TrueEMGain = md.Camera.EMGain
#
#	return md


def genMetaDataFromSourceAndMDH(dataSource, mdh=None):
    md = TIRFDefault

    if not mdh == None: #should be true the whole time
        md.copyEntriesFrom(mdh)

    #guestimate whatever wasn't in the metadata from remaining metadata
    #and image data
    fillInBlanks(md, dataSource)    
    return md


def fillInBlanks(md, dataSource):
    if not 'EstimatedLaserOnFrameNo' in md.getEntryNames():
        #Guestimate when the laser was turned on

        if dataSource.getNumSlices() < 200: #not long enough to bother
            md.setEntry('EstimatedLaserOnFrameNo', 0)
        else:
            #mean intensity in first 200 frames
            I_t = numpy.array([dataSource.getSlice(i).mean(1).mean(0) for i in range(200)])

            #Laser turn on will result in large spike - look at derivative
            dI = numpy.diff(I_t)

            tLon = numpy.argmax(dI)
            print tLon
            dIm = dI[tLon]

            #Now do some sanity checks ...
            if tLon > 100: #shouldn't bee this late
                tLon = 0
                #expect magnitude of differential to be much larger at turn on than later ...
            elif not dIm > 10*dI[(tLon + 50):(tLon + 100)].max():
                tLon = 0 # assume laser was already on
            #also expect the intensity to be decreasing after the initial peak faster than later on in the piece
            elif not dI[(tLon + 5):(tLon + 15)].mean() < -abs(dI[(tLon + 50):(tLon + 100)].mean()):
                tLon = 0

            md.setEntry('EstimatedLaserOnFrameNo', tLon)

    if not 'Camera.ADOffset' in md.getEntryNames():
        tLon = md.EstimatedLaserOnFrameNo
        #Estimate the offset during the dark time before laser was turned on
        #N.B. this will not work if other lights (e.g. room lights, arc lamp etc... are on
        if not tLon == 0:
            md.setEntry('Camera.ADOffset', numpy.median(numpy.array([dataSource.getSlice(i) for i in range(0, max(tLon - 1, 1))]).ravel()))
        else: #if laser was on to start with our best estimate is at maximal bleaching where the few molecules that are still on will hopefully have little influence on the median
            md.setEntry('Camera.ADOffset',numpy.median(numpy.array([dataSource.getSlice(i) for i in range(dataSource.getNumSlices()-5, dataSource.getNumSlices())]).ravel()))
            print '''WARNING: No clear laser turn on signature found - assuming laser was already on
                     and fudging ADOffset Estimation'''



    #Quick hack for approximate EMGain for gain register settings of 150 & 200

    #FIXME to use a proper calibration
	if 'Camera.EMGain' in md.getEntryNames():
		if md.Camera.EMGain == 200: #gain register setting
			md.setEntry('Camera.TrueEMGain', 100) #real gain @ -50C - from curve in performance book - need proper calibration
		elif md.Camera.EMGain == 150: #gain register setting
			md.setEntry('Camera.TrueEMGain', 20) #real gain @ -50C - need proper calibration


	if 'Camera.Name' in md.getEntryNames(): #new improved metadata
		if md.Camera.Name == 'Simulated Standard CCD Camera': #em gain for simulated camera is _real_ em gain rather than gain register setting
			md.Camera.TrueEMGain = md.Camera.EMGain