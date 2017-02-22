"""
This file contains unit tests for the localization algorithms. The goal of these tests is to pick up on backwards
incompatible changes (regressions) which will break things majorly, rather than small changes in localization precision.
For this reason, the tests only look at the IQR of the localization error (i.e. ignore tails) and set fairly permissive
bounds. Because we are using random data for the testing however, and the errors follow a distribution there is a
finite chance that we will get the occasional test failure.  If the test passes on a re-run, this is not a
huge cause for concern.
"""


import os
TESTPATH = os.path.dirname(__file__)

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
    