from . import layers
from PYME.IO import tabular
import dispatch

from PYME.recipes.traits import HasTraits, Enum, ListFloat, Float, Bool, Instance

from pylab import cm
import numpy as np

ENGINES = {
    'points' : layers.Point3DRenderLayer,
    'pointsprites' : layers.PointSpritesRenderLayer,
    'triangles' : layers.TriangleRenderLayer,
}

class LayerWrapper(HasTraits):
    cmap = Enum(*cm.cmapnames, default='gist_rainbow')
    clim = ListFloat([0, 1])
    alpha = Float(1.0)
    visible = Bool(True)
    method = Enum(*ENGINES.keys())
    engine = Instance(layers.BaseLayer)
    
    def __init__(self, pipeline, method='points', ds_name='', cmap='gist_rainbow', clim=[0,1], alpha=1.0, visible=True):
        self._pipeline = pipeline
        self._namespace=getattr(pipeline, 'namespace', {})
        self._dsname = None
        self.engine = None
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        self.visible = visible

        self.on_update = dispatch.Signal()
        
        self.on_trait_change(lambda : self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha')
        self.on_trait_change(self._set_method, 'method')

        self.set_datasource(ds_name)
        self.method = method
        
        self._pipeline.onRebuild.connect(self.update)
        
    @property
    def data_source_names(self):
        names = ['']
        for k, v in self._namespace.items():
            names.append(k)
            if isinstance(v, tabular.colourFilter):
                for c in v.getColourChans():
                    names.append('.'.join([k, c]))
                    
        return names
        
    @property
    def datasource(self):
        if self._dsname == '':
            return self._pipeline
        
        parts = self._dsname.split('.')
        if len(parts) == 2:
            # special case - permit access to channels using dot notation
            # NB: only works if our underlying datasource is a ColourFilter
            ds, channel = parts
            return self._namespace.get(ds, None).get_channel_ds(channel)
        else:
            return self._namespace.get(self._dsname, None)
        
    def _set_method(self):
        self.engine = ENGINES[self.method]()
        self.engine.on_trait_change(self.update)
        
        self.update()
        
    def set_datasource(self, ds_name):
        self._dsname = ds_name
        
        self.update()
    
    def update(self, *args, **kwargs):
        if not (self.engine is None or self.datasource is None):
            self.engine.update_from_datasource(self.datasource, getattr(cm, self.cmap), self.clim, self.alpha)
            self.on_update.send(self)
        
    def render(self, gl_canvas):
        if self.visible:
            self.engine.render(gl_canvas)

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor
        
        try:
            cdata = self.datasource[self.engine.color_key]
        except KeyError:
            cdata = np.array([0,1])
            
        return View([Group([Item('visible'),]),
                     Item('method'),
                     #Item('_'),
                     Group([
                        Item('engine', style='custom', show_label=False, editor=InstanceEditor(view=self.engine.view(self.datasource.keys()))),
                         ]),
                     #Item('engine.color_key', editor=CBEditor(choices=self.datasource.keys())),
                     Item('cmap', label='LUT'),
                     Item('clim', editor = HistLimitsEditor(data=cdata)),
                     Item('alpha')],)
                    #buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
        