"""
This file contains unit tests for the localization algorithms. The goal of these tests is to pick up on backwards
incompatible changes (regressions) which will break things majorly, rather than small changes in localization precision.
For this reason, the tests only look at the IQR of the localization error (i.e. ignore tails) and set fairly permissive
bounds. Because we are using random data for the testing however, and the errors follow a distribution there is a
finite chance that we will get the occasional test failure.  If the test passes on a re-run, this is not a
huge cause for concern.
"""


import os
from PYME.localization import Test
TESTPATH = os.path.dirname(Test.__file__)

from PYME.Analysis import _fithelpers
_fithelpers.EPS_FCN = 1e-4

def test_AstigGaussFitFR():
    """Test the Astigmatic Gaussian fit by fitting some randomly generated events. The pass condition here is fairly
    loose, but should be sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig

    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'astig_gauss.md'))
    tj.runTests(nTests=100)

    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 2.5


def test_LatGaussFitFR():
    from PYME.localization.Test import fitTestJigWC as fitTestJig
    
    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'gauss.md'))
    tj.runTests(nTests=100)
    
    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 2.5

def test_Dumbell3DFitR():
    from PYME.localization.Test import fitTestJigWC as fitTestJig
    
    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'dumbell_3d.md'))
    tj.runTests(nTests=100)

    def error(tj, varName0, varName1):
        xv = tj.ps[varName0].ravel()
        yv = tj.res['fitResults'][varName1]
        if hasattr(tj, varName0):
            yv = yv + tj.__getattribute__(varName0)
            
        return yv - xv
    
    errors_over_pred_IQR0 = fitTestJig.IQR((error(tj, 'x0', 'x0') / tj.res['fitError']['x0']))
    errors_over_pred_IQR1 = fitTestJig.IQR((error(tj, 'x0', 'x1') / tj.res['fitError']['x1']))
    errors_over_pred_IQR = min(errors_over_pred_IQR0, errors_over_pred_IQR1)
    print(errors_over_pred_IQR0, errors_over_pred_IQR1)

    # errors_over_pred_IQR = 3

    # sim_x0 = tj.ps['x0'] 
    # sim_x1 = tj.ps['x1'] 
    # sim_y0 = tj.ps['y0']
    # sim_y1 = tj.ps['y1'] 
    # sim_z0 = tj.ps['z0']
    # sim_z1 = tj.ps['z1']
    # fit_x0 = tj.res['fitResults']['x0']+tj.x0
    # fit_x1 = tj.res['fitResults']['x1']+tj.x0
    # fit_y0 = tj.res['fitResults']['y0']+tj.y0
    # fit_y1 = tj.res['fitResults']['y1']+tj.y0
    # fit_z0 = tj.res['fitResults']['z0']+tj.z0
    # fit_z1 = tj.res['fitResults']['z1']+tj.z0
    # lengths = tj.res['length']

    # for _i in range(100):
    #     print(sim_x0[_i], fit_x0[_i], sim_x1[_i], fit_x1[_i], sim_y0[_i], fit_y0[_i], sim_y1[_i], fit_y1[_i], sim_z0[_i], fit_z0[_i], sim_z1[_i], fit_z1[_i], lengths[_i])
    
    assert errors_over_pred_IQR < 2.5

def test_InterpFitR_astigmatism():
    """Test the 3D interpolated fit by fitting some randomly generated events. The pass condition here is fairly
    loose, but should be sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig
    
    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'astig.md'))
    tj.runTests(nTests=100)
    
    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 2.5


def test_InterpFitR_PRI():
    """Test the 3D interpolated fit by fitting some randomly generated events. The pass condition here is fairly
    loose, but should be sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig
    
    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'pri_theory.md'))
    tj.runTests(nTests=100)
    
    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 2.5


def test_SplitterFitInterpNR_astigmatism():
    """Test the 3D interpolated fit by fitting some randomly generated events. The pass condition here is fairly
    loose, but should be sufficient to detect when the code has been broken"""
    from PYME.localization.Test import fitTestJigWC as fitTestJig
    
    tj = fitTestJig.fitTestJig.fromMDFile(os.path.join(TESTPATH, 'astig_splitter.md'))
    tj.runTests(nTests=100)
    
    errors_over_pred_IQR = fitTestJig.IQR((tj.error('x0') / tj.res['fitError']['x0']))
    print(errors_over_pred_IQR)
    
    assert errors_over_pred_IQR < 2.5
    
def test_numpy_records_view_bug():
    """
    Some versions of numpy have problems with the .view on complex dtypes. Check for this.
    (Known to affect SplitterFitFNR)

    """

    import numpy
    print(numpy.version.full_version)
    
    from PYME.localization.FitFactories import SplitterFitFNR

    frb = numpy.zeros(1, dtype=SplitterFitFNR.fresultdtype)
    
    #this will raise a ValueError for broken versions of numpy
    rf = frb['fitResults'].view('7f4')
