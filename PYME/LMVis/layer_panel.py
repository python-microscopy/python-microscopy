import wx
import wx.lib.newevent

#import PYME.ui.autoFoldPanel as afp
import PYME.ui.manualFoldPanel as afp
import PYME.ui.layerFoldPanel as lfp
import numpy as np

from PYME.ui import histLimits

import PYME.config

import logging
logger = logging.getLogger(__name__)

#DisplayInvalidEvent, EVT_DISPLAY_CHANGE = wx.lib.newevent.NewCommandEvent()

def CreateLayerPane(panel, visFr):
    pane = LayerPane(panel, visFr)
    panel.AddPane(pane)
    return pane

def CreateLayerPanel(visFr):
    import wx.lib.agw.aui as aui
    pane = LayerPane(visFr, visFr)
    pane.SetSize(pane.GetBestSize())
    pinfo = aui.AuiPaneInfo().Name("optionsPanel").Right().Caption('Layers').CloseButton(False).MinimizeButton(True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART|aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
    visFr._mgr.AddPane(pane, pinfo)

from PYME.ui import cascading_layout
class LayerPane(afp.foldingPane):
    def __init__(self, panel, visFr, caption="Layers", add_button=True):
        afp.foldingPane.__init__(self, panel, -1, caption=caption, pinned=True)
        self.visFr = visFr
        
        self._needs_update = True

        self.il = wx.ImageList(16,16)
        self.il.Add(wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_TOOLBAR, (16,16)))
        
        print('Image list size: %d' % self.il.GetImageCount())

        self.fp = afp.foldPanel(self, single_active_pane=True, bottom_spacer=False)

        self.AddNewElement(self.fp)
        
        self.pan = wx.Panel(self, -1)

        self.vsizer = wx.BoxSizer(wx.VERTICAL)
        
        #self.nb = wx.Notebook(self.pan, size=(200, -1))
        #self.nb.AssignImageList(self.il)
        
        self.pages = []
        
        self.update()

        #self.vsizer.Add(self.nb, 1, wx.ALL|wx.EXPAND, 0)
        if add_button:
            bAddLayer = wx.Button(self.pan, -1, 'New', style=wx.BU_EXACTFIT)
            bAddLayer.Bind(wx.EVT_BUTTON, self.add_layer)
    
            self.vsizer.Add(bAddLayer, 0, wx.ALIGN_CENTRE, 0)

        self.pan.SetSizerAndFit(self.vsizer)
        self.AddNewElement(self.pan, priority=0)
        
        #print('Creating layer panel')
        
        self.visFr.layer_added.connect(self.update)
        if hasattr(panel, '_layout'):
            self.fp.fold_signal.connect(panel._layout)
        
        self.Bind(wx.EVT_IDLE, self.on_idle)
        #self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_page_changed)

    def _layout(self, *args, **kwargs):
        print('layout')
        #self.cascading_layout()
        self.vsizer.Fit(self)
        self.Layout()
        self.GetParent().Layout()
        

    def on_idle(self, evt=None):
        if self._needs_update:
            self._needs_update = False
            self._update0()

    def update(self, *args, **kwargs):
        #logger.debug('LayerPanel.update()')
    
        #self.nb.DeleteAllPages()
        #for p in self.pages:
        #p.control.Destroy()
        #    p.dispose()
        #    pass
    
        #while (self.nb.GetPageCount() > 0):
        #    pg = self.nb.RemovePage(0)
    
        #wx.CallAfter(self._update0)
        self._needs_update = True
        
    def _update0(self):
        #logger.debug('LayerPanel._update0()')
        for p in self.pages:
            #p.control.Close()
            p.dispose()
            pass
        self.pages = []
        
        wx.CallAfter(self._update1)
        
    def _update1(self):
        #logger.debug('LayerPanel._update1()')
        self.fp.Clear()
        
        h = 0
        
        print('Creating layers GUI')
        for i, layer in enumerate(self.visFr.layers):
            #print(i, layer)
            item = lfp.LayerFoldingPane(self.fp, layer=layer, caption='Layer %d' % i, pinned=False, folded=True)
            page = layer.edit_traits(parent=item, kind='subpanel')
            item.AddNewElement(page.control)
            
            h = max(h, item.GetBestSize().height)
            self.fp.AddPane(item)
            self.pages.append(page)
            #self.fp.fold1(item)
            #print('Added layer: ', i)
            
        
        n_layers = len(self.pages)
        if  n_layers > 1:
            h += (n_layers -1)*(item.stCaption.GetBestSize().height+5)
        
        print('height: ', h)
        #self.fp.SetMinSize((200, h))
        
        #self.vsizer.Fit(self.pan)
        #self.pan.SetMinSize(self.pan.GetSize())
        
    
        self.sizer.Fit(self)
    
        #print self.pan.GetBestSize(), self.pan.GetSize(), self.GetBestSize(), self.GetSize()
        print('NB best size: ' + repr(self.fp.GetBestSize()))
    
        #self.cascading_layout()
        try:
            self.GetParent().GetParent().Layout()
        except AttributeError:
            pass

        if n_layers >= 1:
            item.Unfold()

        

        #logger.debug('Layer panel update done')

    def add_layer(self, evt):
        dlg = wx.SingleChoiceDialog(self, 'Choose type of layer to add:', 'Add Layer', ['points', 'mesh', 'image', 'tracks', 'quiver'])
        if dlg.ShowModal() == wx.ID_OK:
            type = dlg.GetStringSelection()
            if type == 'points':        
                self.visFr.add_pointcloud_layer()
            elif type == 'mesh':
                self.visFr.add_mesh_layer()
            elif type == 'quiver':
                self.visFr.add_quiver_layer()
            else:
                raise NotImplementedError('Layer type "%s" not supported yet' % type)
        
    def _update(self, *args, **kwargs):
        
        #self.nb.DeleteAllPages()
        #for p in self.pages:
            #p.control.Destroy()
        #    p.dispose()
        #    pass
        
        while (self.nb.GetPageCount() > 0):
            pg = self.nb.RemovePage(0)

        for p in self.pages:
          p.control.Close()
          #p.dispose()
          pass
        
        self.pages = []
        for i, layer in enumerate(self.visFr.layers):
            page = layer.edit_traits(parent=self.nb, kind='subpanel')
            self.pages.append(page)
            self.nb.AddPage(page.control, 'Layer %d' % i)

        self.nb.InvalidateBestSize()

        h = self.nb.GetBestSize().height
        self.nb.SetMinSize((200, h))
        
        self.vsizer.Fit(self.pan)
        self.pan.SetMinSize(self.pan.GetSize())
        
        self.sizer.Fit(self)

        #print self.pan.GetBestSize(), self.pan.GetSize(), self.GetBestSize(), self.GetSize()
        print('NB best size: ' +  repr(self.nb.GetBestSize()))
        
        try:
            self.GetParent().GetParent().Layout()
        except AttributeError:
            pass
         
         
            
        #self.nb.AddPage(wx.Panel(self.nb), 'New', imageId=0)
        
    # def on_page_changed(self, event):
    #     if event.GetSelection() == (self.nb.GetPageCount() -1):
    #         #We have selected the 'new' page
    #
    #         wx.CallAfter(self.visFr.add_layer)
    #     else:
    #         event.Skip()
        
        