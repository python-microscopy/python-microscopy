from PYME.recipes.traits import HasTraits, Float, CStr
import numpy as np

import matplotlib.pyplot as plt

import wx
import wx.lib.agw.aui as aui

from ._base import Plugin

class _Snake_Settings(HasTraits):
    length_weight = Float(0) #alpha
    smoothness = Float(0.1) #beta
    line_weight = Float(-1) #w_line - -ve values seek dark pixels
    edge_weight = Float(0)
    boundaries = CStr('fixed')
    prefilter_sigma = Float(2)


import wx.lib.mixins.listctrl as listmix

class myListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):#, listmix.TextEditMixin):
    def __init__(self, parent, ID, pos=wx.DefaultPosition, size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)
        #listmix.TextEditMixin.__init__(self)
        #self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginLabelEdit)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnLabelActivate)
    
    def OnBeginLabelEdit(self, event):
        if event.m_col == 0:
            event.Veto()
        else:
            event.Skip()
    
    def OnLabelActivate(self, event):
        newLabel = wx.GetTextFromUser("Enter new category name", "Rename")
        if not newLabel == '':
            self.SetStringItem(event.GetIndex(), 1, newLabel)


class LabelPanel(wx.Panel):
    def __init__(self, parent, labeler, **kwargs):
        kwargs['style'] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, parent, **kwargs)
        
        #self.parent = parent
        self.labeler = labeler
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        sbsizer = wx.StaticBoxSizer(wx.StaticBox(self, label='Lock curves to ...'), wx.HORIZONTAL)
        self.cSelectBehaviour = wx.Choice(self, choices=['None', 'ridges', 'valleys', 'edges', 'custom'])
        self.cSelectBehaviour.Bind(wx.EVT_CHOICE, self.on_change_lock_mode)
        
        sbsizer.Add(self.cSelectBehaviour, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        self.bAdjustSnake = wx.Button(self, label='Adj', style=wx.BU_EXACTFIT)
        self.bAdjustSnake.Bind(wx.EVT_BUTTON, self.on_adjust_snake)
        self.bAdjustSnake.SetToolTipString('Adjust the parameters of fo the "snake" (active contour) used for curve locking')
        
        sbsizer.Add(self.bAdjustSnake, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 2)
        
        vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 5)
        
        self.lLabels = myListCtrl(self, -1, style=wx.LC_REPORT)
        self.lLabels.InsertColumn(0, 'Label')
        self.lLabels.InsertColumn(1, 'Structure')
        
        for i in range(10):
            self.lLabels.InsertStringItem(i, '%d' % i)
            self.lLabels.SetStringItem(i, 1, 'Structure %d' % i)
        
        self.lLabels.SetStringItem(0, 1, 'No label')
        
        self.lLabels.SetItemState(1, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED)
        self.lLabels.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lLabels.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        
        self.lLabels.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnChangeStructure)
        
        vsizer.Add(self.lLabels, 1, wx.ALL | wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Line width:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.tLineWidth = wx.TextCtrl(self, -1, '1')
        self.tLineWidth.Bind(wx.EVT_TEXT, self.OnChangeLineWidth)
        hsizer.Add(self.tLineWidth, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.bAddLine = wx.Button(self, label='Add', style=wx.BU_EXACTFIT)
        hsizer.Add(self.bAddLine, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.bAddLine.Bind(wx.EVT_BUTTON, self.labeler.add_curved_line)
        self.bAddLine.SetToolTipString('Add a curve annotation (ctrl-L / cmd-L)')
        vsizer.Add(hsizer, 0, wx.ALL | wx.EXPAND, 5)
        
        self.SetSizer(vsizer)
    
    def OnChangeStructure(self, event):
        self.labeler.cur_label_index = event.GetIndex()
    
    def OnChangeLineWidth(self, event):
        self.labeler.line_width = float(self.tLineWidth.GetValue())
        
    def on_change_lock_mode(self, event=None):
        self.labeler.set_lock_mode(self.cSelectBehaviour.GetStringSelection())
        
    def on_adjust_snake(self, event):
        self.cSelectBehaviour.SetStringSelection('custom')
        self.on_change_lock_mode()
        self.labeler._snake_settings.edit_traits(kind='modal')
        
    def get_labels(self):
        return {n : self.lLabels.GetItemText(n, 1) for n in range(10)}

class Annotater(Plugin):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)

        self.cur_label_index = 1
        self.line_width = 1
        self.lock_mode = 'None'
    
        self._annotations = []
        self.show_annotations = True
        self._snake_settings = _Snake_Settings()
        
        self.selected_annotations = []

        self.penColsA = [wx.Colour(*plt.cm.hsv(v, alpha=0.5, bytes=True)) for v in np.linspace(0, 1, 16)]
        self.brushColsA = [wx.Colour(*plt.cm.hsv(v, alpha=0.2, bytes=True)) for v in np.linspace(0, 1, 16)]
        #self.trackPens = [wx.Pen(c, 4) for c in self.penColsA]

        dsviewer.AddMenuItem('Annotation', "Refine selection\tCtrl-R", self.snake_refine_trace)
        dsviewer.AddMenuItem('Annotation', "Draw line\tCtrl-L", self.add_curved_line)
        dsviewer.AddMenuItem('Annotation', "Draw filled polygon\tCtrl-F", self.add_filled_polygon)
        dsviewer.AddMenuItem('Annotation', 'Clear selected anotations\tCtrl-Back', self.clear_selected)
        dsviewer.AddMenuItem('Annotation', itemType='separator')
        dsviewer.AddMenuItem('Annotation', 'Generate label image', self.get_label_image)
        dsviewer.AddMenuItem('Annotation', 'Generate mask', lambda e : self.get_label_image(mask=True))
        dsviewer.AddMenuItem('Annotation', 'Apply mask to image', self.apply_mask)
        dsviewer.AddMenuItem('Annotation', itemType='separator')
        dsviewer.AddMenuItem('Annotation', "Train SVM Classifier", self.train_svm)
        dsviewer.AddMenuItem('Annotation', "Train Naive Bayes Classifier", self.train_naive_bayes)
        
        self._mi_save = dsviewer.AddMenuItem('Segmentation', 'Save Classifier', self.OnSaveClassifier)
        self._mi_save.Enable(False)
        dsviewer.AddMenuItem('Segmentation', 'Load Classifier', self.OnLoadClassifier)
        self._mi_run = dsviewer.AddMenuItem('Segmentation', "Run Classifier", self.svm_segment)
        self._mi_run.Enable(False)
        
        self.do.on_selection_end.connect(self.snake_refine_trace)
        self.view.add_overlay(self.DrawOverlays, 'Annotations')

        self.view.selectHandlers.append(self.select_annotation)

        self.labelPanel = LabelPanel(dsviewer, self)
        self.labelPanel.SetSize(self.labelPanel.GetBestSize())

        pinfo2 = aui.AuiPaneInfo().Name("labelPanel").Right().Caption('Annotation').CloseButton(False).MinimizeButton(
            True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        dsviewer._mgr.AddPane(self.labelPanel, pinfo2)
        
        
    
    def add_curved_line(self, event=None):
        if self.do.selectionMode == self.do.SELECTION_SQUIGGLE:
            l = self.do.selection_trace
            if len(l) < 1:
                print('Line must have at least 1 point')
                return
                
            if isinstance(l, np.ndarray):
                l = l.tolist()
            self._annotations.append({'type' : 'curve', 'points' : l,
                                      'labelID' : self.cur_label_index, 'z' : self.do.zp,
                                      'width' : self.line_width})
            self.do.selection_trace = []
            
        elif self.do.selectionMode == self.do.SELECTION_LINE:
            x0, y0, x1, y1 = self.do.GetSliceSelection()
            self._annotations.append({'type' : 'line', 'points' : [(x0, y0), (x1, y1)],
                                      'labelID' : self.cur_label_index, 'z':self.do.zp,
                                      'width' : self.line_width})
            
        self.dsviewer.Refresh()
        self.dsviewer.Update()

    def add_filled_polygon(self, event=None):
        if self.do.selectionMode == self.do.SELECTION_SQUIGGLE:
            l = self.do.selection_trace
            if isinstance(l, np.ndarray):
                l = l.tolist()
            self._annotations.append({'type': 'polygon', 'points': l,
                                      'labelID': self.cur_label_index, 'z': self.do.zp,
                                      'width': 1
                                      })
            self.do.selection_trace = []
    
        elif self.do.selectionMode == self.do.SELECTION_RECTANGLE:
            x0, y0, x1, y1 = self.do.GetSliceSelection()
            # TODO - make this a polygon instead?
            self._annotations.append({'type': 'rectangle', 'points': [(x0, y0), (x1, y1)],
                                      'labelID': self.cur_label_index, 'z': self.do.zp,
                                      'width':1
                                      })
    
        self.dsviewer.Refresh()
        self.dsviewer.Update()
        
    def get_json_annotations(self):
        import ujson as json
        
        annotations = {'annotations' : self._annotations,
                       'structures' : self.labelPanel.get_labels()}
        
        return json.dumps(annotations)
            
    def set_lock_mode(self, mode):
        self.lock_mode = mode
        
        if mode in ['ridges', 'valleys']:
            self._snake_settings.edge_weight = 0
            
            if mode == 'ridges':
                self._snake_settings.line_weight = 1.0
            else:
                self._snake_settings.line_weight = -1.0
        elif mode == 'edges':
            self._snake_settings.edge_weight = 1.0
        
            
    def snake_refine_trace(self, event=None, sender=None, **kwargs):
        print('Refining selection')
        if self.lock_mode == 'None' or not self.do.selectionMode == self.do.SELECTION_SQUIGGLE:
            return
        else:
            try:
                from skimage.segmentation import active_contour
                from scipy import ndimage
                
                im = ndimage.gaussian_filter(self.do.ds[:,:,self.do.zp].squeeze(), float(self._snake_settings.prefilter_sigma)).T
                
                pts = np.array(self.do.selection_trace)
                
                self.do.selection_trace = active_contour(im, pts,
                                                         alpha=self._snake_settings.length_weight,
                                                         beta=self._snake_settings.smoothness,
                                                         w_line=self._snake_settings.line_weight,
                                                         w_edge=self._snake_settings.edge_weight,
                                                         bc=self._snake_settings.boundaries)
                
                self.dsviewer.Refresh()
                self.dsviewer.Update()
            except ImportError:
                pass

    def DrawOverlays(self, view, dc):
        if self.show_annotations and (len(self._annotations) > 0):
            bounds = view.visible_bounds
            #vx, vy, vz = self.image.voxelsize
            visible_annotations = [c for c in self._annotations if self._visibletest(c, bounds)]
        
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
        
            for c in visible_annotations:
                pts = np.array(c['points'])
                x, y = pts.T
                z = int(c['z'])
                pFoc = np.vstack(view.pixel_to_screen_coordinates3D(x, y, z)).T

                if c in self.selected_annotations:
                    wf = 2.0
                else:
                    wf = 1.0
            
                dc.SetPen(wx.Pen(self.penColsA[c['labelID'] % 16], wf*max(2, c['width']*2.0**(self.do.scale))))
                
                if (c['type'] == 'polygon'):
                    dc.SetBrush(wx.TheBrushList.FindOrCreateBrush(self.brushColsA[c['labelID'] % 16], wx.BRUSHSTYLE_CROSS_HATCH))
                    dc.DrawPolygon(pFoc)
                    dc.SetBrush(wx.TRANSPARENT_BRUSH)
                else:
                    dc.DrawLines(pFoc)

    def render(self, gl_canvas):
        import  OpenGL.GL as gl
        if self.visible:
            with self.shader_program:
                gl.glDisable(gl.GL_LIGHTING)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glDisable(gl.GL_DEPTH_TEST)

                for c in self._annotations:
                    pts = np.array(c['points'])
                    x, y = pts.T
                    z = int(c['z'])

                    if c in self.selected_annotations:
                        gl.glLineWidth(2.0)
                    else:
                        gl.glLineWidth(1.0)
                    
                    vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
                    vertices = vertices.T.ravel().reshape(len(x), 3)

                    normals = -0.69 * np.ones(vertices.shape)
                    cols = np.ones_like(x)[:,None]*self.penColsA[c['labelID'] % 16][None,:]
                    #n_vertices = vertices.shape[0]
                
                    gl.glVertexPointerf(vertices)
                    gl.glNormalPointerf(normals)
                    gl.glColorPointerf(cols)

                    gl.glDrawArrays(gl.GL_LINE_STRIP, 0, 3*len(x))


    def select_annotation(self, pos):
        for c in self._annotations:
            if self._hittest(c, pos):
                if c in self.selected_annotations:
                    self.selected_annotations.remove(c)
                    print('deselecting annotation)')
                else:
                    self.selected_annotations.append(c)
                    print('selecting annotation')

                
                self.dsviewer.Refresh()
                self.dsviewer.Update()
                return True

        else:
            self.selected_annotations.clear()
            print('clearing selections')
            self.dsviewer.Refresh()
            self.dsviewer.Update()
            return False
    
    def clear_selected(self, event=None):
        for c in self.selected_annotations:
            self._annotations.remove(c)

        self.selected_annotations.clear()
        self.dsviewer.Refresh()
        self.dsviewer.Update()

    def _visibletest(self, clump, bounds):
    
        xb, yb, zb = bounds
        
        x, y = np.array(clump['points']).T
        t = clump['z']
    
        return np.any((x >= xb[0]) * (y >= yb[0]) * (t >= (zb[0] - 1)) * (x < xb[1]) * (y < yb[1]) * (t < (zb[1] + 1)))

    def _hittest(self, clump, pos):
        xp, yp, zp = pos
    
        bounds = [(xp - 2, xp + 2), (yp - 2, yp + 2), (zp, zp + 1)]
    
        return self._visibletest(clump, bounds)
    
    def _calc_spline_gradients(self, pts):
        from scipy import sparse
        from scipy.sparse import linalg
        N = len(pts)
        OD = np.ones(N-1)
        Dg = 4*np.ones(N)
        Dg[0] = 2
        Dg[-1] = 2
        
        Yvs = np.array(pts)[:,1]
        
        b = np.zeros(N)
        b[1:-1] = Yvs[2:] - Yvs[:-2]
        b[0] = Yvs[1] - Yvs[0]
        b[-1] = Yvs[-1] - Yvs[-2]
        
        A = sparse.diags([OD, Dg, OD], [-1, 0, 1])
        
        D = linalg.spsolve(A, b)
        
        return D
        
    
    def _draw_line_segment(self, P0, P1, width, label, output, X, Y):
        x1, y1 = P0
        x2, y2 = P1
        
        hwidth = width/2.0
        pad_width = int(np.ceil(hwidth + 1))
        
        xb_0 = int(max(min(x1, x2) - pad_width, 0))
        xb_1 = int(min(max(x1, x2) + pad_width, output.shape[0]))
        yb_0 = int(max(min(y1, y2) - pad_width, 0))
        yb_1 = int(min(max(y1, y2) + pad_width, output.shape[1]))
        
        X_ = X[xb_0:xb_1, yb_0:yb_1]
        Y_ = Y[xb_0:xb_1, yb_0:yb_1]
        
        #im = output[xb_0:xb_1, yb_0:yb_1]
        
        dx = x2 - x1
        dy = y2 - y1
        dist = np.abs(dy*X_ - dx*Y_ + (x2*y1 - y2*x1))/np.sqrt(dx*dx + dy*dy)
        mask = dist <= hwidth
        output[xb_0:xb_1, yb_0:yb_1][mask] = label
        
    
    def rasterize(self, z):
        output = np.zeros(self.do.ds.shape[:2], 'uint8')
        X, Y = np.mgrid[:output.shape[0], :output.shape[1]]
        
        for a in self._annotations:
            if a['z'] == z:
                pts = a['points']
                
                label = int(a['labelID'])

                if a['type'] in ['curve', 'line']:
                    for i in range(1, len(pts)):
                        sp = pts[i-1]
                        ep = pts[i]
                        
                        self._draw_line_segment(sp, ep, a['width'], label, output, X, Y)
                elif a['type'] == 'polygon':
                    from skimage import draw
                    rr, cc = draw.polygon(*np.array(pts).T, shape=output.shape)
                    #rr = np.clip(rr, 0, output.shape[0] -1, output)
                    output[rr, cc] = label
                    
        return output
    
    def get_label_image(self, event=None, mask=False):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        lab = np.concatenate([np.atleast_3d(self.rasterize(z)) for z in range(self.do.ds.shape[2])], 2)
        if mask:
            im = ImageStack(lab > 0.5, titleStub='Mask')
        else:
            im = ImageStack(lab, titleStub='Labels')
            
        im.mdh.copyEntriesFrom(self.dsviewer.image.mdh)

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.lv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
        
    def apply_mask(self, event=None):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        
        masked = np.concatenate([np.atleast_3d((self.rasterize(z)>0)*self.do.ds[:,:,z]) for z in range(self.do.ds.shape[2])], 2)
        im = ImageStack(masked, titleStub='Masked')

        im.mdh.copyEntriesFrom(self.dsviewer.image.mdh)

        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'

        self.lv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))

    
    def train_svm(self, event=None):
        from PYME.Analysis import svmSegment
    
        #from PYME.IO.image import ImageStack
        #from PYME.DSView import ViewIm3D
    
        #if not 'cf' in dir(self):
        self.cf = svmSegment.svmClassifier()
        self.cf.train(self.dsviewer.image.data[:, :, self.do.zp, 0].squeeze(), self.rasterize(self.do.zp))

        self._mi_save.Enable(True)
        self._mi_run.Enable(True)
        self.svm_segment()

    def train_naive_bayes(self, event=None):
        from PYME.Analysis import svmSegment
        from sklearn.naive_bayes import GaussianNB
    
        #from PYME.IO.image import ImageStack
        #from PYME.DSView import ViewIm3D
        
        clf = GaussianNB()
    
        #if not 'cf' in dir(self):
        #if not 'cf' in dir(self):
        self.cf = svmSegment.svmClassifier(clf=clf)
    
        self.cf.train(self.dsviewer.image.data[:, :, self.do.zp, 0].squeeze(), self.rasterize(self.do.zp))
        self._mi_save.Enable(True)
        self._mi_run.Enable(True)
        self.svm_segment()

    def svm_segment(self, event=None):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        # import pylab
        import matplotlib.cm
        #sp = self.image.data.shape[:3]
        #if len(sp)
        lab2 = self.cf.classify(self.dsviewer.image.data[:, :, self.do.zp, 0].squeeze())#, self.image.labels[:,:,self.do.zp])
        #self.vmax = 0
        #self.image.labels = self.mask
    
        im = ImageStack(lab2, titleStub='Segmentation')
        im.mdh.copyEntriesFrom(self.dsviewer.image.mdh)
    
        #im.mdh['Processing.CropROI'] = roi
    
        if self.dsviewer.mode == 'visGUI':
            mode = 'visGUI'
        else:
            mode = 'lite'
    
        self.dv = ViewIm3D(im, mode=mode, glCanvas=self.dsviewer.glCanvas, parent=wx.GetTopLevelParent(self.dsviewer))
    
        self.rois = []
    
        #set scaling to (0,10)
        for i in range(im.data.shape[3]):
            self.dv.do.Gains[i] = .1
            self.dv.do.cmaps[i] = matplotlib.cm.labeled
    
        self.dv.Refresh()
        self.dv.Update()

    def OnSaveClassifier(self, event=None):
        filename = wx.FileSelector("Save classifier as:", wildcard="*.pkl", flags=wx.FD_SAVE)
        if not filename == '':
            self.cf.save(filename)

    def OnLoadClassifier(self, event=None):
        from PYME.Analysis import svmSegment
        filename = wx.FileSelector("Load Classifier:", wildcard="*.pkl", flags=wx.FD_OPEN)
        if not filename == '':
            self.cf = svmSegment.svmClassifier(filename=filename)
            self._mi_run.Enable(True)
                    

def Plug(dsviewer):
    return Annotater(dsviewer)
