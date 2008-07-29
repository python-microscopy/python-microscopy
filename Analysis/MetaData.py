class VoxelSize:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

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
