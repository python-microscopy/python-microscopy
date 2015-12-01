#PYME Simple Metadata v1
import os
import numpy as np

#print __file__
md['EndTime'] = 1300676178.4949999
md['EstimatedLaserOnFrameNo'] = 0

### What PSF to use for simulation (and fitting if using an *Interp* fit
#md['PSFFile'] = os.path.join(os.path.split(__file__)[0], 'wf_theory2.psf')
#md['PSFFile'] = os.path.join('/home/david/Desktop/2014_11_13_psf_dec.psf')
#md['PSFFile'] = os.path.join('/home/david/Desktop/sp_psf_dec.psf')
#md['PSFFile'] = "ZMODEL:{'nSample': 1.51, 'zmodes':{}}"
#md['PSFFile'] = "ZMODEL:{'nSample': 1.4, 'zmodes':{4:.137, 5:-.185, 6:.035, 7:.093, 8:-.097, 9:.313, 10:.0415, 14:.157,15:.0857, 17:-.243}}"
md['PSFFile'] = "ZMODEL:{'nSample': 1.4, 'zmodes':{8:1}}"

md['StartTime'] = 1300676151.901
md['tIndex'] = 0
md['Analysis.BGRange'] = [0, 0]
md['Analysis.DataFileID'] = 1571469165
md['Analysis.DebounceRadius'] = 14
md['Analysis.DetectionThreshold'] = 7.0

md['Analysis.PoissonML'] = False

### The fit module to use
#md['Analysis.FitModule'] = u'SplitterFitFNR'		#2D splitter fit
md['Analysis.FitModule'] = u'SplitterFitInterpNR'	#3D splitter fit
#md['Analysis.FitModule'] = u'SplitterFitInterpRR'	#3D splitter fit
#md['Analysis.FitModule'] = u'SplitterFitInterpBNR'	#3D 'biplane' splitter fit in which ratio can only take on discrete values

md['Analysis.InterpModule'] = 'CSInterpolator'
md['Analysis.AxialShift'] = -250			#The focal shift between the two image planes
md['Analysis.EstimatorModule'] = 'biplaneEstimator'
md['Analysis.ColourRatio'] = 0.5
md['Analysis.FitShifts'] = False

### How to handle background
md['Analysis.subtractBackground'] = True		#Do we attempt to perform background subtraction before fitting (default=True)
md['Analysis.FitBackground'] = False			#Do we include a background term in the fit (shouldn't be necessary if background subtracted properly)

### Camera parameters - These should be OK as is
md['Camera.ADOffset'] = 1159.0
md['Camera.CycleTime'] = 0.25178998708724976
md['Camera.EMGain'] = 150
md['Camera.ElectronsPerCount'] = 27.32
md['Camera.IntegrationTime'] = 0.25
md['Camera.Model'] = 'DV897_BV'
md['Camera.Name'] = 'Andor IXon DV97'
md['Camera.NoiseFactor'] = 1.4099999999999999
md['Camera.ROIHeight'] = 227
md['Camera.ROIPosX'] = 16
md['Camera.ROIPosY'] = 275
md['Camera.ROIWidth'] = 420
md['Camera.ReadNoise'] = 109.8
md['Camera.SerialNumber'] = 1823
md['Camera.StartCCDTemp'] = -68
md['Camera.TrueEMGain'] = 33.415239495686144
md['Positioning.PIFoc'] = 172.5455
md['Positioning.Stage_X'] = 9.020599365234375
md['Positioning.Stage_Y'] = 9.8524932861328125
md['Splitter.Dichroic'] = 'FF741-Di01'
md['Splitter.TransmittedPathPosition'] = 'Top'
md['StackSettings.EndPos'] = 175.0829
md['StackSettings.NumSlices'] = 100
md['StackSettings.ScanMode'] = 'Middle and Number'
md['StackSettings.ScanPiezo'] = 'PIFoc'
md['StackSettings.StartPos'] = 170.12819999999999
md['StackSettings.StepSize'] = 0.050000000000000003
md['voxelsize.units'] = 'um'
md['voxelsize.x'] = 0.073999999999999996
md['voxelsize.y'] = 0.071999999999999995
md['voxelsize.z'] = 0.050000000000000003

### Fake a shift field
class sffake:
    def __init__(self, val):
        self.val = val

    def ev(self, x, y):
        return self.val

md['chroma.dx'] = sffake(0.)
#md['chroma.dy'] = sffake(100.)
md['chroma.dy'] = sffake(-0.)

## The splitting ratio to use for simulation
md['chroma.ChannelRatio'] = .7

## The discrete splitting ratios to try when using SplitterFitInterpBNR
md['chroma.ChannelRatios'] = [.7]

### Parameters and jitter magnitudes for testing
#md['Test.DefaultParams'] = [20000, 20000, 0, 0, 0, 0, 10]
#md['Test.ParamJitter'] = [15, 15, 90, 90, 250, 10, 10]
#md['Test.DefaultParams'] = [2000, 1000, 0, 0, 0, 0, 0]
#md['Test.ParamJitter'] = [0, 0, 90, 90, 350, 0, 0]

md['Test.DefaultParams'] = [4000, 0, 0, md['Analysis.AxialShift']*.5, 0, 0, 0, 0] #note the the reference to axial shift is to center the z-range
md['Test.ParamJitter'] = [00, 90, 90, 400, 0, 0, 0, 0]

### Module to use for generating the images
md['Test.SimModule'] = u'SplitterFitInterpBNR'

### Size of the ROI to generate and fit
md['Test.ROISize']=9

### Magnitude of background (in 'photons') to add to the generated molecule image
md['Test.Background'] = 10

#md['Test.PSFFile'] = os.path.join(os.path.split(__file__)[0], 'wf_theory2.psf')
#md['Test.PSFFile'] = os.path.join('/home/david/Desktop/2014_11_13_psf_dec.psf')
#md['Test.PSFFile'] = os.path.join('/home/david/Desktop/sp_psf_dec.psf')
#md['Test.PSFFile'] = "ZMODEL:{'nSample': 1.4, 'zmodes':{8: 0.44, 10: -0.2, 4: 0.7, 6: -0.3, 7: 0.3}}"
md['Test.PSFFile'] = "ZMODEL:{'nSample': 1.45, 'zmodes':{8:0}}"
