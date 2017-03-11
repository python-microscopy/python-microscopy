#PYME Simple Metadata v1
import os
import numpy as np

#print __file__
md['EndTime'] = 1300676178.4949999
md['EstimatedLaserOnFrameNo'] = 0
md['PSFFile'] = os.path.join(os.path.split(__file__)[0], 'astig_theory.tif')
md['StartTime'] = 1300676151.901
md['tIndex'] = 0
md['Analysis.BGRange'] = [0, 0]
md['Analysis.DataFileID'] = 1571469165
md['Analysis.DebounceRadius'] = 14
md['Analysis.DetectionThreshold'] = 7.0
md['Analysis.FitModule'] = u'SplitterFitInterpNR'
md['Analysis.InterpModule'] = 'CSInterpolator'
md['Analysis.AxialShift'] = 500
#md['Analysis.EstimatorModule'] = 'priEstimator'
#md['Analysis.subtractBackground'] = False
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
#load shift field
#dx, dy = np.load(os.path.join(os.path.split(__file__)[0], '30_9_series_A_1.sf'))
#md['chroma.dx'] = dx
#md['chroma.dy'] = dy
class sffake:
    def __init__(self, val):
        self.val = val

    def ev(self, x, y):
        return self.val

md['chroma.dx'] = sffake(50.)
md['chroma.dy'] = sffake(100.)
md['Test.DefaultParams'] = [2000, 2000, 0, 0, 0, 10, 10, 0, 0]
md['Test.ParamJitter'] = [15, 15, 90, 90, 250, 10, 10, 0, 0]
md['Test.Background'] = 10.
