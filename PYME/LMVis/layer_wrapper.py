from . import layers
from PYME.IO import tabular
from PYME.contrib import dispatch

from PYME.recipes.traits import HasTraits, Enum, ListFloat, Float, Bool, Instance, CStr

# from pylab import cm
from matplotlib import cm
import numpy as np

ENGINES = {
    'points' : layers.Point3DRenderLayer,
    'pointsprites' : layers.PointSpritesRenderLayer,
    'triangles' : layers.TetrahedraRenderLayer,
}

class LayerWrapper(HasTraits):
    cmap = Enum(*cm.cmapnames, default='gist_rainbow')
    clim = ListFloat([0, 1])
    alpha = Float(1.0)
    visible = Bool(True)
    method = Enum(*ENGINES.keys())
    engine = Instance(layers.BaseLayer)
    dsname = CStr('output')
    
    def __init__(self, pipeline, method='points', ds_name='', cmap='gist_rainbow', clim=[0,1], alpha=1.0, visible=True, method_args={}):
        self._pipeline = pipeline
        #self._namespace=getattr(pipeline, 'namespace', {})
        #self.dsname = None
        self.engine = None
        
        self.cmap = cmap
        self.clim = clim
        self.alpha = alpha
        
        self.visible = visible

        self.on_update = dispatch.Signal()
        
        self.on_trait_change(lambda : self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, alpha, dsname')
        self.on_trait_change(self._set_method, 'method')

        #self.set_datasource(ds_name)
        self.dsname = ds_name
        
        self._eng_params = dict(method_args)
        self.method = method
        
        self._pipeline.onRebuild.connect(self.update)
        
    @property
    def _namespace(self):
        return self._pipeline.layer_datasources
    
    @property
    def bbox(self):
        return  self.engine.bbox
    
    @property
    def colour_map(self):
        return self.engine.colour_map
        
    @property
    def data_source_names(self):
        names = []#'']
        for k, v in self._namespace.items():
            names.append(k)
            if isinstance(v, tabular.ColourFilter):
                for c in v.getColourChans():
                    names.append('.'.join([k, c]))
                    
        return names
        
    @property
    def datasource(self):
        if self.dsname == '':
            return self._pipeline
        
        parts = self.dsname.split('.')
        if len(parts) == 2:
            # special case - permit access to channels using dot notation
            # NB: only works if our underlying datasource is a ColourFilter
            ds, channel = parts
            return self._namespace.get(ds, None).get_channel_ds(channel)
        else:
            return self._namespace.get(self.dsname, None)
        
    def _set_method(self):
        if self.engine:
            self._eng_params = self.engine.get('point_size', 'vertexColour')
            #print(eng_params)
            
        self.engine = ENGINES[self.method](self._context)
        self.engine.set(**self._eng_params)
        self.engine.on_trait_change(self._update, 'vertexColour')
        self.engine.on_trait_change(self.update)
        
        self.update()
        
    # def set_datasource(self, ds_name):
    #     self._dsname = ds_name
    #
    #     self.update()
    
    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(cdata.min()), float(cdata.max())]
        #self.update(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        print('lw update')
        if not (self.engine is None or self.datasource is None):
            self.engine.update_from_datasource(self.datasource, getattr(cm, self.cmap), self.clim, self.alpha)
            self.on_update.send(self)
        
    def render(self, gl_canvas):
        if self.visible:
            self.engine.render(gl_canvas)

    def _get_cdata(self):
        try:
            cdata = self.datasource[self.engine.vertexColour]
        except KeyError:
            cdata = np.array([0,1])
            
        return cdata
    
    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor
        
        
            
        return View([Group([Item('dsname', label='Data', editor=EnumEditor(values=self.data_source_names)),]),
                     Item('method'),
                     #Item('_'),
                     Group([
                        Item('engine', style='custom', show_label=False, editor=InstanceEditor(view=self.engine.view(self.datasource.keys()))),
                         ]),
                     #Item('engine.color_key', editor=CBEditor(choices=self.datasource.keys())),
                     
                     Group([Item('clim', editor = HistLimitsEditor(data=self._get_cdata), show_label=False),]),
                     Group([Item('cmap', label='LUT') ,Item('alpha'), Item('visible')], orientation='horizontal', layout='flow')],)
                    #buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
        