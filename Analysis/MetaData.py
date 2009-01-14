import numpy
class VoxelSize:
	def __init__(self, x, y, z, unit='um'):
		self.x = x
		self.y = y
		self.z = z
		self.unit = unit

class CCDMetaDataIXonDefault:
	#all at 10Mhz, e.m. amplifier
	name = 'Andor IXon DV97'
	ADOffset = 1100 #counts (@ T ~ -70, em gain ~ 150)
	ReadNoise = 109.8 #electrons
	noiseFactor = 1.41
	electronsPerCount = 27.32
	

class MetaData:
	def __init__(self, voxSize, CCDMD = None):
		self.voxelsize = voxSize
		self.CCD = CCDMD

#FIXME - THIS SHOULD ALL BE EXTRACTED FROM LOG FILES OR THE LIKE
TIRFDefault = MetaData(VoxelSize(0.07, 0.07, 0.2), CCDMetaDataIXonDefault())

def genMetaDataFromHDF(h5File):
	md = TIRFDefault

	#once we start putting everything into data file properly,
	#grab the relavant bits here

	#Guestimate when the laser was turned on
	
	#Laser turn on will result in large spike - look at derivative
	dI = numpy.diff(h5File.root.ImageData[:200, :,:].mean(2).mean(1))
	
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

        md.EstimatedLaserOnFrameNo = tLon

        #Estimate the offset during the dark time before laser was turned on
        #N.B. this will not work if other lights (e.g. room lights, arc lamp etc... are on
	if not tLon == 0: 
		md.CCD.ADOffset = numpy.median(h5File.root.ImageData[:max(tLon - 1, 1), :,:].ravel())
	else: #if laser was on to start with our best estimate is at maximal bleaching where the few molecules that are still on will hopefully have little influence on the median
		md.CCD.ADOffset = numpy.median(h5File.root.ImageData[-5:, :,:].ravel())
		print '''WARNING: No clear laser turn on signature found - assuming laser was already on
		         and fudging ADOffset Estimation'''

        #Quick hack for approximate EMGain for gain register settings of 150 & 200
        #FIXME to use a proper calibration
	if 'EMGain' in dir(h5File.root.MetaData.Camera._v_attrs):
		if h5File.root.MetaData.Camera._v_attrs.EMGain == 200: #gain register setting
			md.CCD.EMGain = 100 #real gain @ -50C - from curve in performance book - need proper calibration
		elif h5File.root.MetaData.Camera._v_attrs.EMGain == 150: #gain register setting
			md.CCD.EMGain = 20 #real gain @ -50C - need proper calibration
	else: #early file, or from camera without EMGain - assume early file - gain was usually 150
		md.CCD.EMGain = 20


	return md
