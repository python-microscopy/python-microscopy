from . import recipeLayout
import wx
import numpy as np
from PYME.IO import tabular

import logging
logger = logging.getLogger(__name__)

# class RecipeDisplayPanel(wx.ScrolledWindow):
#     def __init__(self, *args, **kwargs):
#         wx.ScrolledWindow.__init__(self, *args, **kwargs)

class RecipeDisplayPanel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args,**kwargs)
        self.recipe = None

        self.output_positions = {}
        self.input_positions = {}
        self.data_positions = {}
        self.data_positions_in = {}

        self.input_target_panes = {}
        self.output_source_panes = {}
        self.data_panes = {}
        
        self.orderd = []
        self.x0s = {}
        
        self.cols = {}
        self._pens = {}
        
        self.fp = None

        self._traits_views = []

        self._layout_valid = False
        
        #self.SetMinSize((200, 500))
        
        #self.SetScrollRate(0, 20)
        #self.ShowScrollbars(wx.SHOW_SB_DEFAULT, wx.SHOW_SB_ALWAYS)
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_IDLE, self.OnIdle)

    def _col(self,node):
        import matplotlib.pyplot as plt
        if not node in self.cols.keys():
            self.cols[node] = 0.7 * np.array(plt.cm.hsv(np.random.rand()))
            #cols[node] = 0.7 * np.array(plt.cm.hsv((_col.n_col % len(data_nodes)) / float(len(data_nodes))))
        #return self.cols[node]

        c = self.cols[node]
        if self.recipe.failed and not node in self.recipe.namespace.keys():
            c = 0.2 * c + 0.8 * 0.5
            return np.array([.9,.6,.6,1.])
            
        return c
        
    def SetRecipe(self, recipe):
        self.recipe = recipe
        self.recipe.recipe_changed.connect(self.invalidate_layout)
        self.recipe.recipe_failed.connect(self.invalidate_layout)
        self.invalidate_layout()
        self.recipe.recipe_executed.connect(self._update_n_events)

    def _refr(self, **kwargs):
        #print 'p_'
        #wx.CallLater(10, self.Refresh)
        self.Layout()
        #print self.GetSizer().GetSize()
        self.SetMinSize(self.GetSizer().GetSize())
        self.GetParent().Layout()
        self.GetParent().GetParent().Layout()
        self.Refresh()

    def _set_n_events(self, node, st):
        node_col = tuple([int(v) for v in (255 * self._col(node)[:3])])

        evts = ''
        data = self.recipe.namespace.get(node, None)
        if isinstance(data, tabular.TabularBase):
            evts = ' [%d evts]' % len(data)
                
        st.SetLabelMarkup("<span foreground='#%02x%02x%02x'>%s%s</span>" % (node_col + (node,evts)))

    def _update_n_events(self, *args, **kwargs):
        for node, t in self.data_panes.items():
            self._set_n_events(node, t[1])

    def OnIdle(self, evt=None):
        if not self._layout_valid:
            self._layout_valid = True
            self._layout()

    def invalidate_layout(self, *args, **kwargs):
        # avoid re-doing the layout multiple times
        # flag as invalid and let OnIdle do the layout
        self._layout_valid = False

    def _layout(self, *args, **kwargs):
        #print('RecipeView._layout')
        #import traceback
        #traceback.print_stack()

        # dispose of any current traits views
        for tv in self._traits_views:
            tv.dispose()

        self._traits_views.clear()

        wx.CallAfter(self._layout2)
        #self._layout2()
        
    def _layout2(self):
        #print('RecipeView._layout2')
        
        if self.fp:
            _popEventHandlers(self.fp, handler_type=wx.EvtHandler)
            self.fp.elements = []
            # Wrap this in a try block to prevent an error when using OutputModules
            # TODO: Figure out why this is necessary.
                       
            try:
                self.fp.DestroyChildren()
            except:
                logger.exception('error destroying fold panel children')
                pass
        self.fp = None
        #print('destroyed fold panel children')
        
        self.DestroyChildren()
        #print('destroyed children')
        

        from matplotlib import pyplot as plt
        from .base import ModuleBase, OutputModule
        import textwrap
        
        #from PYME.ui import autoFoldPanel as afp
        import PYME.ui.manualFoldPanel as afp

        dg = self.recipe.dependancyGraph()
        rdg = self.recipe.reverseDependancyGraph()

        self.ordered, self.x0s = recipeLayout.layout_vertical(dg, rdg)

        x_vals = -.1 * (1 + np.arange(len(self.ordered)))


        self.input_target_panes = {}
        self.output_source_panes = {}
        self.data_panes = {}
        
        #what data do we already have in the namespace?
        self.input_datasources = list(self.recipe.namespace.keys())
        
        for node in self.ordered:
            if isinstance(node, ModuleBase):
                for op in node.outputs:
                    try:
                        #this is obviously not an input - remove from list of input data
                        self.input_datasources.remove(op)
                    except ValueError:
                        pass

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.AddSpacer(20)
        
        self.fp = afp.foldPanel(self)
        hsizer.Add(self.fp, 1, wx.EXPAND, 0)

        hsizer.AddSpacer(10)
        
        for N, node in enumerate(self.ordered):
            if isinstance(node, ModuleBase):
                #This is a module - plot a box
                s = node.__class__.__name__
                
                item = afp.foldingPane(self.fp, -1, caption=s, pinned = False)
                if getattr(node, '_has_buttons', False) or (getattr(node,'_get_actions', None) is not None):
                    kind = 'panel'
                else:
                    kind = 'subpanel'
                pan = node.edit_traits(parent=item, kind=kind, view='pipeline_view_min')
                self._traits_views.append(pan)
                pan.control.SetMinSize((150, -1))
                item.AddNewElement(pan.control)
                if getattr(node, '_last_error', None):
                    # error on this node, turn background red
                    # TODO - this is a rather hacky
                    item.stCaption.style.update({'BACKGROUND_COLOUR_1': (220, 198, 198), #default AUI caption colours
                                               'BACKGROUND_COLOUR_2': (255, 226, 226)})
                self.fp.AddPane(item)
                
                #p = pan.control
                #j += 1
        
                #draw input lines
                inputs = list(node.inputs)
                ip_xs = [self._data_output_pos(ip)[0] for ip in inputs]
                ip_ys = np.linspace(0, 1, 2 * len(inputs) + 1)[1::2]
                ip_ys = ip_ys[np.argsort(ip_xs)[::-1]]

                for ip_y, ip in zip(ip_ys, inputs):
                    self.input_target_panes[ip] = (item, ip_y)
                    #self.input_positions[ip] = lambda : (self.fp.panes[j].Position[0], self.fp.panes[j].Position[1] + ip_y*self.fp.panes[j].GetSize()[1])
                    #self.input_positions[ip] = lambda : _get_op(j)
                #
                # #draw the nodes outputs
                outputs = list(node.outputs)[::-1]
                if len(outputs) > 0:
                    op_y = np.linspace(0, 1, 2 * len(outputs) + 1)[1::2]
                    op_x = 5*np.arange(len(outputs))[::-1]

                    for yp, xp, op in zip(op_y, op_x, outputs):
                        self.output_source_panes[op] = (item, xp, yp)
                        
                        #self.output_positions[op] = lambda : (item.GetPosition()[0] + item.GetSize()[0] + 5 + xp, item.GetPosition()[1] + yp*item.GetSize()[1])
        
            else:
                # we must be an input - route back to LHS
                try:
                    #only map back if we are going to be used.
                    if True:#rdg.get(node, False):
                        xi, yi = self._output_position(node)
                        x_0 = x_vals[self.x0s[node]]
                        #ax.plot([2, xi, xi, x_0], [yi, yi, y, y], '-', color=_col(node), lw=2)

                        #print('Adding static text')
                        item = afp.foldingPane(self.fp, -1, caption=None, pinned=True, folded=False, padding=0, style=0)
                        st = wx.StaticText(item, -1, node)
                        
                        self._set_n_events(node, st)
                            
                        item.AddNewElement(st, foldable=False)
                        self.fp.AddPane(item)
                        
                        self.data_panes[node] = (item, st)

                        #self.data_positions[node] = lambda : (item.GetPosition()[0] + x_0, item.GetPosition()[1] + 0.5 * item.GetSize()[1])
                        #self.data_positions_in[node] = lambda : (item.GetPosition()[0] + st.GetSize()[0], item.GetPosition()[1] + 0.5 * item.GetSize()[1])
            
        
                except KeyError:
                    #dangling input
                    #print('dangling input')
                    
                    x_0 = x_vals[self.x0s[node]]

                    item = afp.foldingPane(self.fp, -1, caption=None, pinned=True, folded=False, padding=0, style=0)
                    st = wx.StaticText(item, -1, node)

                    node_col = tuple([int(v) for v in (255 * self._col(node)[:3])])
                    
                    
                    bg = ''
                    evts = ''
                    
                    if node in self.input_datasources:
                        bg = " background='#D0D0D0'"

                    data = self.recipe.namespace.get(node, None)
                    if isinstance(data, tabular.TabularBase):
                        evts = ' [%d evts]' % len(data)
                       
                    st.SetLabelMarkup("<u><span foreground='#%02x%02x%02x' weight='bold'%s>%s%s</span></u>" % (node_col + (bg, node,evts)))
                    item.AddNewElement(st, foldable=False)
                    self.fp.AddPane(item)

                    self.data_panes[node] = (item, st)
                    #self.data_positions[node] = lambda : (item.GetPosition()[0] + x_0, item.GetPosition()[1] + 0.5 * item.GetSize()[1])
                
        
        print('Input datasources: ',  self.input_datasources)
        
        self.fp.fold_signal.connect(self._refr)
        self.SetSizerAndFit(hsizer)
        self.Layout()
        
    def _input_position(self, key):
        item, ip_y = self.input_target_panes[key]
        xp, yp = item.Position
        sx, sy = item.Size
        
        return xp, yp + ip_y*sy

    def _output_position(self, key):
        item, ip_x, ip_y = self.output_source_panes[key]
        xp, yp = item.Position
        sx, sy = item.Size
    
        return xp + sx + 5 + ip_x, yp + ip_y * sy
    
    def _data_input_pos(self, key):
        item, st = self.data_panes[key]
        xp, yp = item.Position
        sx, sy = item.Size
        
        return xp + st.GetTextExtent(st.GetLabel())[0] + 3, yp + 0.5*sy

    def _data_output_pos(self, key):
        item, st = self.data_panes[key]
        xp, yp = item.Position
        sx, sy = item.Size
    
        return xp - 5*(1+self.x0s[key]), yp + 0.5 * sy
        
        
    def OnPaint(self, dc):
        #print 'p'
        #if not self.IsShownOnScreen():
        #    return
        if self.fp is None:
            return
        

        dc = wx.PaintDC(self)
        #self.PrepareDC(dc)
        #dc.BeginDrawing()

        x_0 = self.fp.Position[0]
        x_1 = x_0 + self.fp.Size[0]

        dc.SetPen(wx.BLACK_PEN)
        for nd in self.x0s.keys():
            dc.SetPen(wx.Pen(wx.Colour(*((255*self._col(nd))[:3]).astype('i').tolist()), 2))
            try:
                x0, y0 = self._output_position(nd)
                x1, y1 = self._data_input_pos(nd)

                dc.DrawLines([(x_1, y0), (x0+x_0, y0), (x0+x_0, y1), (x1+x_0, y1)])
            except KeyError:
                pass

            try:
                x0, y0 = self._input_position(nd)
                x1, y1 = self._data_output_pos(nd)

                dc.DrawLines([(x_0, y0), (x1+x_0, y0), (x1+x_0, y1), (x_0, y1)])
            except KeyError:
                pass
        
        #dc.EndDrawing()

def _popEventHandlers(ctrl, handler_type=wx.EvtHandler):
    """Pop any event handlers that have been pushed on to a window and its
    children.
    """
    # borrowed from traitsui.wx.toolkit

    handler = ctrl.GetEventHandler()
    while ctrl is not handler:
        next_handler = handler.GetNextHandler()
        if isinstance(handler, handler_type):
            ctrl.PopEventHandler(True)
        handler = next_handler
    for child in ctrl.GetChildren():
        _popEventHandlers(child, handler_type)