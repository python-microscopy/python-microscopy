import numpy as np
from traits.api import HasPrivateTraits, Float, Bool
from traitsui.api import View, Item, OKButton

class SurfaceFitter(HasPrivateTraits):
    fitInfluenceRadius = Float(100, desc='The region around each localization to include in the surface fit [nm]. The fit is performed on all points falling within this radius of each control point')
    reconstructionRadius = Float(50, desc ='The size of the reconstructed surface patch. This should usually be <= fitInfluenceRadius')
    constrainSurfaceToPoint = Bool(True, desc='Whether the fit should be constrained to pass through the control point')
    limitReconstructionToSupportHull = Bool(False, desc='If enabled, this will clip each surface reconstruction to the convex hull of all the points used for the fit.\
     Useful for avoiding the generation of large surface patches from isolated antibodies, but also reduces the ability to paper over holes')
    normalAlignmentThreshold = Float(0.85)
    reconstructionPointSpacing = Float(10., desc='Spacing of points used to reconstruct the surface')

    view = View(Item('fitInfluenceRadius'),
                Item('constrainSurfaceToPoint'),
                Item('reconstructionRadius'),
                Item('reconstructionPointSpacing'),
                Item('limitReconstructionToSupportHull'),
                Item('normalAlignmentThreshold'),
                buttons=[OKButton])
    
    def __init__(self, visFr):
        self._visFr = visFr

        visFr.AddMenuItem('Analysis>Surface Fitting', 'Settings', lambda e: self.configure_traits(kind='modal'))
        visFr.AddMenuItem('Analysis>Surface Fitting', "Fit Surface Model", self.OnFitSurfaces)
        
        
        
    
    def OnFitSurfaces(self, event):
        from PYME.Analysis.points import surfit
        from PYME.IO import tabular
        
        pipeline = self._visFr.pipeline
        
        #arrange point data in the format we expect
        pts = np.vstack([pipeline['x'].astype('f'), pipeline['y'].astype('f'), pipeline['z'].astype('f')])
        

        #do the actual fitting - this fits one surface for every point in the dataset
        f = surfit.fit_quad_surfaces_Pr(pts.T, self.fitInfluenceRadius, fitPos=(not self.constrainSurfaceToPoint))
        
        #print(len(f)) #, f.dtype
        
        sfits = tabular.recArrayInput(f)
        
        pipeline.addDataSource('surf_fits', sfits, False)
        
        #filter surfaces and throw out those which don't point the same way as their neighbours
        f = surfit.filter_quad_results(f, pts.T, self.fitInfluenceRadius,self.normalAlignmentThreshold)

        #do the reconstruction by generating an augmented point data set for each surface
        #this adds virtual localizations spread evenly across each surface
        if self.limitReconstructionToSupportHull:
            xs, ys, zs, xn, yn, zn, N = surfit.reconstruct_quad_surfaces_Pr_region_cropped(f, self.reconstructionRadius, pts.T,
                                                                           fit_radius=self.fitInfluenceRadius, step=self.reconstructionPointSpacing)
        else:
            xs, ys, zs, xn, yn, zn, N = surfit.reconstruct_quad_surfaces_Pr(f, self.reconstructionRadius, step=self.reconstructionPointSpacing)
        
        #construct a new datasource with our augmented points
        ds = tabular.mappingFilter({'x': xs, 'y': ys, 'z' : zs,
                                    'xn': xn, 'yn' : yn, 'zn': zn,
                                    'probe' : np.zeros_like(xs), 'Npoints' : N})
        
        #add the datasource to the pipeline and set it to be the active data source
        pipeline.addDataSource('surf', ds, False)
        pipeline.selectDataSource('surf')
        pipeline.Rebuild()
        
        self._visFr.Refresh()


def Plug(visFr):
    '''Plugs this module into the gui'''
    #pass
    SurfaceFitter(visFr)