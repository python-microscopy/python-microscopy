from PYME.recipes.traits import HasTraits, Float, CStr
import numpy as np

import matplotlib.pyplot as plt

import wx
import wx.lib.agw.aui as aui

from PYME.ui import selection

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
        self.bAdjustSnake.SetToolTip('Adjust the parameters of fo the "snake" (active contour) used for curve locking')
        
        sbsizer.Add(self.bAdjustSnake, 0, wx.ALIGN_CENTER_VERTICAL|wx.LEFT, 2)
        
        vsizer.Add(sbsizer, 0, wx.ALL|wx.EXPAND, 5)
        
        self.lLabels = myListCtrl(self, -1, style=wx.LC_REPORT)
        self.lLabels.InsertColumn(0, 'Label')
        self.lLabels.InsertColumn(1, 'Structure')
        
        for i in range(10):
            self.lLabels.InsertItem(i, '%d' % i)
            self.lLabels.SetItem(i, 1, 'Structure %d' % i)
        
        self.lLabels.SetItem(0, 1, 'No label')
        
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
        self.bAddLine.SetToolTip('Add a curve annotation (ctrl-L / cmd-L)')
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


def minimum_distance_to_poly(poly, points, closed=True):
    """
    calculate minimum distances between a set of 2D line segments and
    a set of 2D points

    Parameters
    ----------

    poly : Nx2 ndarray
        line segment or polygon

    points : Mx2 ndarray
        points to test

    Returns
    -------

    Mx1 array of (signed?) distances
    """

    if closed:
        #closed polygon, connect first and last points
        a = poly
        b = np.roll(poly, 1, axis=0)
    else:
        #open line
        a = poly[:-1, :]
        b = poly[1:, :]

    a_b = b - a
    #length squared of line segments
    l2 = (a_b*a_b).sum(1)
    #print(l2)

    #allocate arrays for squared distance and indices
    #set squared distance to an unreasonably high initial value
    d2 = 1e12*np.ones(points.shape[0])
    distance_sign = np.zeros(points.shape[0]) #keep track of which segment was closest (so we can check sign)

    # loop over polygon segments
    # chosen this way as N is expected to be << M, but sufficiently large
    # that we don't want to create a dense NxM matrix with broadcasting
    for i in range(len(a)):
        if l2[i] > 0: #skip over zero length sides 
            #find vectors from start of segment to each point
            ai_p = points - a[i, :][None,:]

            #project on the line segment, and clamp to [0,1]
            adp = (ai_p*a_b[i, :][None,:]).sum(1)
            t = np.clip(adp/l2[i], 0, 1) 
            proji = a[i,:][None,:] + t[:,None]*a_b[i, :][None,:]

            #calc squared distance from projected point to point
            vd = points - proji
            d2i = (vd*vd).sum(1)

            #compare to current value of d2
            distance_sign[d2i < d2] = np.sign(ai_p[:, 0]*a_b[i, 1] - ai_p[:, 1]*a_b[i, 0])[d2i < d2]
            d2 = np.minimum(d2, d2i)

    #check winding of polygon
    abb = (a_b[1:, :]*a_b[:-1, :]).sum(1)
    poly_sign = np.sign(np.sign(abb).sum())

    return np.sqrt(d2)*distance_sign*poly_sign

class AnnotationList(list):
    def __init__(self, iterable=None, json=None, filename=None):

        if (json is not None) or (filename is not None):
            raise NotImplementedError('serialisation is not implemented yet')

        if iterable is None:
            list.__init__(self)
        else:
            list.__init__(self, iterable)

    def add(self, type, points, label, z=None, width=1):
        assert type in ['curve', 'line', 'polygon', 'rectangle']
        self.append({'type' : type, 'points' : points,
                                      'labelID' : label, 'z':z,
                                      'width' : width})

    def add_curve(self, points, label, z=None, width=1):
        if isinstance(points, np.ndarray):
                points = points.tolist()
        self.add('curve', points, label, z, width)

    def add_line(self, start, finish, label, z=None, width=1):
        points = [tuple(start[:2]), tuple(finish[:2])]
        self.add('line', points, label, z, width)

    def add_polygon(self, points, label, z=None, width=1):
        if isinstance(points, np.ndarray):
                points = points.tolist()
        self.add('polygon', points, label, z, width)

    def add_rectangle(self, start, finish, label, z=None, width=1):
        points = [tuple(start[:2]), tuple(finish[:2])]
        self.add('rectangle', points, label, z, width)

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
        
    
    def rasterize(self, z, shape):
        #output = np.zeros(self.do.ds.shape[:2], 'uint8')
        output = np.zeros(shape, 'uint8')
        X, Y = np.mgrid[:output.shape[0], :output.shape[1]]
        
        for a in self:
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

    def label_points(self, points):
        """
        Label points based on the current set of annotations

        Parameters
        ----------

            points : Mx2 ndarray

        Returns
        -------

            labels: Mx1 array of labels


        """

        out = np.zeros(points.shape[0], 'i4')

        for a in self:
            pts = a['points']
            
            label = int(a['labelID'])

            #calculate the minimum signed distance from each point to the contour
            dist = minimum_distance_to_poly(np.array(pts), points, closed=a['type']=='polygon')

            if a['type'] in ['curve', 'line']:
                out[np.abs(dist) <= (a['width']/2)] = label
            elif a['type'] == 'polygon':
                out[dist <= 0] = label
                    
        return out

    

class AnnotateBase(object):
    def __init__(self, win, selector, minimize=False):
        """
        Parameters
        ==========
        win : PYME.ui.AUIFrame (or derived class) instance. 
            Window to add annotation contols to. Must implement AddMenuItem()
        selector : PYME.ui.selection.Selection instance
            class holding info about the current selection.
        """
        self.cur_label_index = 1
        self.line_width = 1
        self.lock_mode = 'None'
        self.visible = True
    
        self._annotations = AnnotationList()
        self.show_annotations = True
        self._snake_settings = _Snake_Settings()
        
        self.selected_annotations = []

        self._selector = selector

        self.penColsA = [wx.Colour(*plt.cm.hsv(v, alpha=0.5, bytes=True)) for v in np.linspace(0, 1, 16)]
        self.brushColsA = [wx.Colour(*plt.cm.hsv(v, alpha=0.2, bytes=True)) for v in np.linspace(0, 1, 16)]
        #self.trackPens = [wx.Pen(c, 4) for c in self.penColsA]

        win.AddMenuItem('Annotation', "Refine selection\tCtrl-R", self.snake_refine_trace)
        win.AddMenuItem('Annotation', "Draw line\tCtrl-L", self.add_curved_line)
        win.AddMenuItem('Annotation', "Draw filled polygon\tCtrl-F", self.add_filled_polygon)
        win.AddMenuItem('Annotation', "Clear all annotations", self.clear_all)
        win.AddMenuItem('Annotation', 'Clear selected anotations\tCtrl-Back', self.clear_selected)
        

        self.labelPanel = LabelPanel(win, self)
        self.labelPanel.SetSize(self.labelPanel.GetBestSize())

        pinfo2 = aui.AuiPaneInfo().Name("labelPanel").Right().Caption('Annotation').CloseButton(False).MinimizeButton(
            True).MinimizeMode(aui.AUI_MINIMIZE_CAPT_SMART | aui.AUI_MINIMIZE_POS_RIGHT)#.CaptionVisible(False)
        win._mgr.AddPane(self.labelPanel, pinfo2)
        if minimize:
            win._mgr.MinimizePane(pinfo2)
        
    def _update_view(self):
        """ Over-ride in derived classes"""
        pass

    def _zp(self):
        """
        Return the current z-position

        Override in derived classes
        """
        return None    
    
    def add_curved_line(self, event=None):
        if self._selector.mode == selection.SELECTION_SQUIGGLE:
            l = self._selector.trace
            if len(l) < 1:
                print('Line must have at least 1 point')
                return
                
            self._annotations.add_curve(l, label=self.cur_label_index, z=self._zp(), width=self.line_width)
            self._selector.trace = []
            
        elif self._selector.mode == selection.SELECTION_LINE:
            self._annotations.add_line(self._selector.start, self._selector.finish, label=self.cur_label_index, z=self._zp(), width=self.line_width)
            
        self._update_view()

    def add_filled_polygon(self, event=None):
        if self._selector.mode == selection.SELECTION_SQUIGGLE:
            self._annotations.add_polygon(self._selector.trace, label=self.cur_label_index, z=self._zp(), width=self.line_width)
            self._selector.trace = []
    
        elif self._selector.mode == selection.SELECTION_RECTANGLE:
            self._annotations.add_rectangle(self._selector.start, self._selector.finish, label=self.cur_label_index, z=self._zp(), width=self.line_width)
    
        self._update_view()
        
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
        if self.lock_mode == 'None' or not self._selector.mode == selection.SELECTION_SQUIGGLE:
            return
        else:
            try:
                from skimage.segmentation import active_contour
                from scipy import ndimage
                
                im = ndimage.gaussian_filter(self.do.ds[:,:,self._zp()].squeeze(), float(self._snake_settings.prefilter_sigma)).T
                
                pts = np.array(self._selector.trace)
                
                self._selector.trace = active_contour(im, pts,
                                                         alpha=self._snake_settings.length_weight,
                                                         beta=self._snake_settings.smoothness,
                                                         w_line=self._snake_settings.line_weight,
                                                         w_edge=self._snake_settings.edge_weight,
                                                         bc=self._snake_settings.boundaries)
                
                self._update_view()
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
        
        gl.glDisable(gl.GL_LIGHTING)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDisable(gl.GL_DEPTH_TEST)

        for c in self._annotations:
            pts = np.array(c['points'])
            x, y = pts.T
            z = c['z']
            if z:
                z = int(c['z'])*np.ones_like(x)
            else:
                z = np.zeros_like(x)

            if c in self.selected_annotations:
                gl.glLineWidth(8.0)
            else:
                gl.glLineWidth(4.0)
            
            vertices = np.vstack((x.ravel(), y.ravel(), z.ravel()))
            vertices = vertices.T.ravel().reshape(len(x), 3)

            normals = -0.69 * np.ones(vertices.shape)
            
            cols = np.ones_like(x)[:,None]*np.array(self.penColsA[c['labelID'] % 16])[None,:]/255.
            #n_vertices = vertices.shape[0]
            #gl.glColor4fv(np.array(self.penColsA[c['labelID'] % 16]))

            gl.glVertexPointerf(vertices)
            gl.glNormalPointerf(normals)
            gl.glColorPointerf(cols)

            if c['type'] == 'polygon':
                gl.glDrawArrays(gl.GL_LINE_LOOP, 0, len(x))
            else:
                # line, curve
                gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(x))


    def select_annotation(self, pos):
        for c in self._annotations:
            if self._hittest(c, pos):
                if c in self.selected_annotations:
                    self.selected_annotations.remove(c)
                    print('deselecting annotation)')
                else:
                    self.selected_annotations.append(c)
                    print('selecting annotation')

                
                self._update_view()
                return True

        else:
            self.selected_annotations.clear()
            print('clearing selections')
            self._update_view()
            return False
    
    def clear_selected(self, event=None):
        for c in self.selected_annotations:
            self._annotations.remove(c)

        self.selected_annotations.clear()
        self._update_view()

    def clear_all(self, event=None):
        self._annotations.clear()
        self.selected_annotations.clear()

        self._update_view()

    def _visibletest(self, clump, bounds):
    
        xb, yb, zb, tb = bounds
        
        x, y = np.array(clump['points']).T
        t = clump['z']
    
        return np.any((x >= xb[0]) * (y >= yb[0]) * (t >= (zb[0] - 1)) * (x < xb[1]) * (y < yb[1]) * (t < (zb[1] + 1)))

    def _hittest(self, clump, pos):
        xp, yp, zp = pos
    
        bounds = [(xp - 2, xp + 2), (yp - 2, yp + 2), (zp, zp + 1), (0, 100)]
    
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

    
        
    

    
    

class Annotater(Plugin, AnnotateBase):
    def __init__(self, dsviewer):
        Plugin.__init__(self, dsviewer)
        AnnotateBase.__init__(self, dsviewer, dsviewer.do.selection)

        win = dsviewer

        win.AddMenuItem('Annotation', itemType='separator')
        win.AddMenuItem('Annotation', 'Generate label image', self.get_label_image)
        win.AddMenuItem('Annotation', 'Generate mask', lambda e : self.get_label_image(mask=True))
        win.AddMenuItem('Annotation', 'Apply mask to image', self.apply_mask)
        win.AddMenuItem('Annotation', itemType='separator')
        win.AddMenuItem('Annotation', "Train SVM Classifier", self.train_svm)
        win.AddMenuItem('Annotation', "Train Naive Bayes Classifier", self.train_naive_bayes)
        
        self._mi_save = win.AddMenuItem('Segmentation', 'Save Classifier', self.OnSaveClassifier)
        self._mi_save.Enable(False)
        win.AddMenuItem('Segmentation', 'Load Classifier', self.OnLoadClassifier)
        self._mi_run = win.AddMenuItem('Segmentation', "Run Classifier", self.svm_segment)
        self._mi_run.Enable(False)
        
        self.do.on_selection_end.connect(self.snake_refine_trace)
        self.view.add_overlay(self.DrawOverlays, 'Annotations')
        self.view.selectHandlers.append(self.select_annotation)

    def _update_view(self):
        self.dsviewer.Refresh()
        self.dsviewer.Update()

    def _zp(self):
        return self.do.zp

    def get_label_image(self, event=None, mask=False):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        lab = np.concatenate([np.atleast_3d(self._annotations.rasterize(z, self.do.ds.shape[:2])) for z in range(self.do.ds.shape[2])], 2)
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
        
        masked = np.concatenate([np.atleast_3d((self._annotations.rasterize(z, self.do.ds.shape[:2])>0)*self.do.ds[:,:,z]) for z in range(self.do.ds.shape[2])], 2)
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
        self.cf.train(self.dsviewer.image.data[:, :, self._zp(), 0].squeeze().astype('f'), self._annotations.rasterize(self._zp(), self.do.ds.shape[:2]))

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
    
        self.cf.train(self.dsviewer.image.data[:, :, self._zp(), 0].squeeze(), self._annotations.rasterize(self._zp(), self.do.ds.shape[:2]))
        self._mi_save.Enable(True)
        self._mi_run.Enable(True)
        self.svm_segment()

    def svm_segment(self, event=None):
        from PYME.IO.image import ImageStack
        from PYME.DSView import ViewIm3D
        # import pylab
        from PYME.misc.colormaps import cm
        #sp = self.image.data.shape[:3]
        #if len(sp)
        lab2 = self.cf.classify(self.dsviewer.image.data[:, :, self._zp(), 0].squeeze().astype('f'))#, self.image.labels[:,:,self.do.zp])
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
            self.dv.do.cmaps[i] = cm.labeled
    
        self.dv.Refresh()
        self.dv.Update()

    def OnSaveClassifier(self, event=None):
        filename = wx.FileSelector("Save classifier as:", wildcard="*.joblib", flags=wx.FD_SAVE)
        if not filename == '':
            self.cf.save(filename)

    def OnLoadClassifier(self, event=None):
        from PYME.Analysis import svmSegment
        filename = wx.FileSelector("Load Classifier:", wildcard="*.joblib", flags=wx.FD_OPEN)
        if not filename == '':
            self.cf = svmSegment.svmClassifier(filename=filename)
            self._mi_run.Enable(True)

    
                    

def Plug(dsviewer):
    return Annotater(dsviewer)
