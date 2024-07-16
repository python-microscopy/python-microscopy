"""
A layer which draws text labels at the positions of the points in a tabular datasource.
"""

from PYME.LMVis.layers.base import SimpleLayer

from PYME.recipes.traits import CStr, Float, Enum, ListFloat, List, Bool
# from pylab import cm
from PYME.misc.colormaps import cm
import numpy as np
import string
from PYME.contrib import dispatch
from .text import Text

import logging
logger = logging.getLogger(__name__)

class LabelLayer(SimpleLayer):
    """
    A layer which draws text labels at the positions of the points in a tabular datasource.
    """

    dsname = CStr('output', desc='Name of the datasource within the pipeline to use as a source of points')
    _datasource_keys = List()
    _datasource_choices = List()

    textColour = CStr('', desc='Name of variable used to colour our text')
    font_size = Float(10, desc='font size in points')
    cmap = Enum('SolidWhite', cm.cmapnames, desc='Name of colourmap used to colour text')
    clim = ListFloat([0, 1], desc='How our variable should be scaled prior to colour mapping')

    format_string = CStr('P{idx}', desc='Format string for the text labels. This should be a python format string which can take column names (and the special idx name which is the index in the table)')

    def __init__(self, pipeline, **kwargs):
        SimpleLayer.__init__(self, **kwargs)
        self._pipeline = pipeline
        
        self.x_key = 'x' #TODO - make these traits?
        self.y_key = 'y'
        self.z_key = 'z'

        self._bbox = None
        self._text_objects = []
    
        # define a signal so that people can be notified when we are updated (currently used to force a redraw when
        # parameters change)
        self.on_update = dispatch.Signal()

        # signal for when the data is updated (used to, e.g., refresh histograms)
        self.data_updated = dispatch.Signal()

        # define responses to changes in various traits
        self.on_trait_change(self._update, 'textColour')
        self.on_trait_change(lambda: self.on_update.send(self), 'visible')
        self.on_trait_change(self.update, 'cmap, clim, dsname, font_size, format_string')

        # update any of our traits which were passed as command line arguments
        self.set(**kwargs)

        self.update()
        
        # if we were given a pipeline, connect ourselves to the onRebuild signal so that we can automatically update
        # ourselves
        if not self._pipeline is None:
            self._pipeline.onRebuild.connect(self.update)

    @property
    def datasource(self):
        """
        Return the datasource we are connected to (through our dsname property).
        """
        return self._pipeline.get_layer_data(self.dsname)
    
    def _get_cdata(self):
        try:
            cdata = self.datasource[self.textColour]
        except KeyError:
            cdata = np.array([0, 1])
    
        return cdata
    
    def _update(self, *args, **kwargs):
        cdata = self._get_cdata()
        self.clim = [float(np.nanmin(cdata)), float(np.nanmax(cdata))+1e-9]
        #self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        #print('lw update')
        self._datasource_choices = self._pipeline.layer_data_source_names
        if not self.datasource is None:
            self._datasource_keys = sorted(self.datasource.keys())
            
            self.update_from_datasource(self.datasource)
            self.on_update.send(self)
            

    @property
    def bbox(self):
        return self._bbox
    
    @property
    def colour_map(self):
        return cm[self.cmap]
    
    def update_from_datasource(self, ds):
        print('labels.update_from_datasource() - dsname=%s' % self.dsname)
        x, y = ds[self.x_key], ds[self.y_key]
        try:
            z = ds[self.z_key]
        except (KeyError, ValueError):
            z = 0*x
        
        if not self.textColour == '':
            c = ds[self.textColour]
        else:
            c = 0*x

        # find out what fields we need to format
        format_field_names  = [f for _, f, _, _ in string.Formatter().parse(self.format_string) if f is not None]  
        
        data_columns = {k: ds[k] for k in format_field_names if not k == 'idx'}
        data_columns['idx'] = np.arange(len(x))

        labels = [self.format_string.format(**{k:v[i] for k, v in data_columns.items()}) for i in range(len(x))]

        self.update_data(x, y, z, c, cmap=cm[self.cmap], clim=self.clim, labels=labels)

        self.data_updated.send(self)
    
    
    def update_data(self, x=None, y=None, z=None, colors=None, cmap=None, clim=None, labels=None, alpha=1.0):
        self._text_objects = []
        self._bbox = None

        if clim is not None and colors is not None and clim is not None:
            cs_ = ((colors - clim[0]) / (clim[1] - clim[0]))
            cs = cmap(cs_)
            cs[:, 3] = alpha
            
            cs = cs.ravel().reshape(len(colors), 4)
        else:
            if x is not None:
                cs = np.ones((x.shape[0], 4), 'f')
            else:
                cs = None
        
        if x is not None and y is not None and z is not None and len(x) > 0:
            self._text_objects = [Text(pos=(x[i], y[i], z[i]), text=labels[i], color=cs[i, :], font_size=self.font_size) for i in range(len(x))]
            
            self._bbox = np.array([x.min(), y.min(), z.min(), x.max(), y.max(), z.max()])
        else:
            
            self._bbox = None

    def render(self, gl_canvas):
        if not self.visible:
            return
        
        for to in self._text_objects:
            to.render(gl_canvas)

    @property
    def default_view(self):
        from traitsui.api import View, Item, Group, InstanceEditor, EnumEditor, TextEditor
        from PYME.ui.custom_traits_editors import HistLimitsEditor, CBEditor

        vis_when = 'cmap not in %s' % cm.solid_cmaps
    
        return View([Group([Item('dsname', label='Data', editor=EnumEditor(name='_datasource_choices')), ]),
                     Item('textColour', editor=EnumEditor(name='_datasource_keys'), label='Colour', visible_when=vis_when),
                     Group([Item('clim', editor=HistLimitsEditor(data=self._get_cdata, update_signal=self.data_updated), show_label=False), ], visible_when=vis_when),
                     Group(Item('cmap', label='LUT'),
                           #Item('alpha', visible_when="method in ['pointsprites', 'transparent_points']", editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                           Item('font_size', label=u'Font\u00A0size', editor=TextEditor(auto_set=False, enter_set=True, evaluate=float)),
                           Item('format_string', label='Format string', editor=TextEditor(auto_set=False, enter_set=True)),
                           )])
        #buttons=['OK', 'Cancel'])

    def default_traits_view(self):
        return self.default_view
        
        

       
    

