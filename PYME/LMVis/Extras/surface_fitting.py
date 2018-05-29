import numpy as np
from traits.api import HasPrivateTraits, Float, Bool
from traitsui.api import View, Item, OKButton

class SurfaceFitter(HasPrivateTraits):
    fitInfluenceRadius = Float(100)
    reconstructionRadius = Float(50)
    constrainSurfaceToPoint = Bool(True)
    limitReconstructionToSupportHull = Bool(False)

    view = View(Item('fitInfluenceRadius'),
                Item('reconstructionRadius'),
                Item('constrainSurfaceToPoint'),
                Item('limitReconstructionToSupportHull'),
                buttons=[OKButton])
    
    def __init__(self, visFr):
        self._visFr = visFr

        visFr.AddMenuItem('Extras>Surface Fitting', 'Settings', lambda e: self.configure_traits(kind='modal'))
        visFr.AddMenuItem('Extras>Surface Fitting', "Fit Surface Model", self.OnFitSurfaces)
        
        
        
    
    def OnFitSurfaces(self, event):
        from PYME.Analysis.points import surfit
        from PYME.IO import tabular
        #return
        
        pipeline = self._visFr.pipeline

        pts = np.vstack([pipeline['x'].astype('f'), pipeline['y'].astype('f'), pipeline['z'].astype('f')])
        
        #print pts.shape

        f = surfit.fit_quad_surfaces_Pr(pts.T, self.fitInfluenceRadius, fitPos=(not self.constrainSurfaceToPoint))
        
        print(len(f)) #, f.dtype

        if self.limitReconstructionToSupportHull:
            xs, ys, zs, xn, yn, zn, N = surfit.reconstruct_quad_surfaces_Pr_region_cropped(f, self.reconstructionRadius, pts.T,
                                                                           fit_radius=self.fitInfluenceRadius)
        else:
            xs, ys, zs, xn, yn, zn, N = surfit.reconstruct_quad_surfaces_Pr(f, self.reconstructionRadius)
        
        ds = tabular.mappingFilter({'x': xs, 'y': ys, 'z' : zs,
                                    'xn': xn, 'yn' : yn, 'zn': zn,
                                    'probe' : np.zeros_like(xs), 'Npoints' : N})
        
        pipeline.addDataSource('surf', ds, False)
        pipeline.selectDataSource('surf')
        pipeline.Rebuild()
        
        self._visFr.Refresh()


def Plug(visFr):
    '''Plugs this module into the gui'''
    #pass
    SurfaceFitter(visFr)