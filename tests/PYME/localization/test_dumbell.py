import os
from PYME.localization import Test
TESTPATH = os.path.dirname(Test.__file__)

from PYME.Analysis import _fithelpers
_fithelpers.EPS_FCN = 1e-4

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
    
    # Overly permissive, but should indicate if test is totally failing
    assert errors_over_pred_IQR < 10.0