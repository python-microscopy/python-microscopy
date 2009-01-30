import  wx
import wx.lib.foldpanelbar as fpb
import gl_render
import sys
import inpFilt
import editFilterDialog

# ----------------------------------------------------------------------------
# Visualisation of analysed localisation microscopy data
#
# David Baddeley 2009
#
# Some of the code in this file borrowed from the wxPython examples
# ----------------------------------------------------------------------------


def GetCollapsedIconData():
    return \
'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x8eIDAT8\x8d\xa5\x93-n\xe4@\x10\x85?g\x03\n6lh)\xc4\xd2\x12\xc3\x81\
\xd6\xa2I\x90\x154\xb9\x81\x8f1G\xc8\x11\x16\x86\xcd\xa0\x99F\xb3A\x91\xa1\
\xc9J&\x96L"5lX\xcc\x0bl\xf7v\xb2\x7fZ\xa5\x98\xebU\xbdz\xf5\\\x9deW\x9f\xf8\
H\\\xbfO|{y\x9dT\x15P\x04\x01\x01UPUD\x84\xdb/7YZ\x9f\xa5\n\xce\x97aRU\x8a\
\xdc`\xacA\x00\x04P\xf0!0\xf6\x81\xa0\xf0p\xff9\xfb\x85\xe0|\x19&T)K\x8b\x18\
\xf9\xa3\xe4\xbe\xf3\x8c^#\xc9\xd5\n\xa8*\xc5?\x9a\x01\x8a\xd2b\r\x1cN\xc3\
\x14\t\xce\x97a\xb2F0Ks\xd58\xaa\xc6\xc5\xa6\xf7\xdfya\xe7\xbdR\x13M2\xf9\
\xf9qKQ\x1fi\xf6-\x00~T\xfac\x1dq#\x82,\xe5q\x05\x91D\xba@\xefj\xba1\xf0\xdc\
zzW\xcff&\xb8,\x89\xa8@Q\xd6\xaaf\xdfRm,\xee\xb1BDxr#\xae\xf5|\xddo\xd6\xe2H\
\x18\x15\x84\xa0q@]\xe54\x8d\xa3\xedf\x05M\xe3\xd8Uy\xc4\x15\x8d\xf5\xd7\x8b\
~\x82\x0fh\x0e"\xb0\xad,\xee\xb8c\xbb\x18\xe7\x8e;6\xa5\x89\x04\xde\xff\x1c\
\x16\xef\xe0p\xfa>\x19\x11\xca\x8d\x8d\xe0\x93\x1b\x01\xd8m\xf3(;x\xa5\xef=\
\xb7w\xf3\x1d$\x7f\xc1\xe0\xbd\xa7\xeb\xa0(,"Kc\x12\xc1+\xfd\xe8\tI\xee\xed)\
\xbf\xbcN\xc1{D\x04k\x05#\x12\xfd\xf2a\xde[\x81\x87\xbb\xdf\x9cr\x1a\x87\xd3\
0)\xba>\x83\xd5\xb97o\xe0\xaf\x04\xff\x13?\x00\xd2\xfb\xa9`z\xac\x80w\x00\
\x00\x00\x00IEND\xaeB`\x82' 

def GetCollapsedIconBitmap():
    return wx.BitmapFromImage(GetCollapsedIconImage())

def GetCollapsedIconImage():
    import cStringIO
    stream = cStringIO.StringIO(GetCollapsedIconData())
    return wx.ImageFromStream(stream)

#----------------------------------------------------------------------
def GetExpandedIconData():
    return \
'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\
\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\
\x00\x01\x9fIDAT8\x8d\x95\x93\xa1\x8e\xdc0\x14EO\xb2\xc4\xd0\xd2\x12\xb7(mI\
\xa4%V\xd1lQT4[4-\x9a\xfe\xc1\xc2|\xc6\xc2~BY\x83:A3E\xd3\xa0*\xa4\xd2\x90H!\
\x95\x0c\r\r\x1fK\x81g\xb2\x99\x84\xb4\x0fY\xd6\xbb\xc7\xf7>=\'Iz\xc3\xbcv\
\xfbn\xb8\x9c\x15 \xe7\xf3\xc7\x0fw\xc9\xbc7\x99\x03\x0e\xfbn0\x99F+\x85R\
\x80RH\x10\x82\x08\xde\x05\x1ef\x90+\xc0\xe1\xd8\ryn\xd0Z-\\A\xb4\xd2\xf7\
\x9e\xfbwoF\xc8\x088\x1c\xbbae\xb3\xe8y&\x9a\xdf\xf5\xbd\xe7\xfem\x84\xa4\
\x97\xccYf\x16\x8d\xdb\xb2a]\xfeX\x18\xc9s\xc3\xe1\x18\xe7\x94\x12cb\xcc\xb5\
\xfa\xb1l8\xf5\x01\xe7\x84\xc7\xb2Y@\xb2\xcc0\x02\xb4\x9a\x88%\xbe\xdc\xb4\
\x9e\xb6Zs\xaa74\xadg[6\x88<\xb7]\xc6\x14\x1dL\x86\xe6\x83\xa0\x81\xba\xda\
\x10\x02x/\xd4\xd5\x06\r\x840!\x9c\x1fM\x92\xf4\x86\x9f\xbf\xfe\x0c\xd6\x9ae\
\xd6u\x8d \xf4\xf5\x165\x9b\x8f\x04\xe1\xc5\xcb\xdb$\x05\x90\xa97@\x04lQas\
\xcd*7\x14\xdb\x9aY\xcb\xb8\\\xe9E\x10|\xbc\xf2^\xb0E\x85\xc95_\x9f\n\xaa/\
\x05\x10\x81\xce\xc9\xa8\xf6><G\xd8\xed\xbbA)X\xd9\x0c\x01\x9a\xc6Q\x14\xd9h\
[\x04\xda\xd6c\xadFkE\xf0\xc2\xab\xd7\xb7\xc9\x08\x00\xf8\xf6\xbd\x1b\x8cQ\
\xd8|\xb9\x0f\xd3\x9a\x8a\xc7\x08\x00\x9f?\xdd%\xde\x07\xda\x93\xc3{\x19C\
\x8a\x9c\x03\x0b8\x17\xe8\x9d\xbf\x02.>\x13\xc0n\xff{PJ\xc5\xfdP\x11""<\xbc\
\xff\x87\xdf\xf8\xbf\xf5\x17FF\xaf\x8f\x8b\xd3\xe6K\x00\x00\x00\x00IEND\xaeB\
`\x82' 

def GetExpandedIconBitmap():
    return wx.BitmapFromImage(GetExpandedIconImage())

def GetExpandedIconImage():
    import cStringIO
    stream = cStringIO.StringIO(GetExpandedIconData())
    return wx.ImageFromStream(stream)



class VisGUIFrame(wx.Frame):
    
    def __init__(self, parent, id=wx.ID_ANY, title="PYME Visualise", pos=wx.DefaultPosition,
                 size=(700,650), style=wx.DEFAULT_FRAME_STYLE):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)

        self._flags = 0
        
        #self.SetIcon(GetMondrianIcon())
        self.SetMenuBar(self.CreateMenuBar())

        self.statusbar = self.CreateStatusBar(2, wx.ST_SIZEGRIP)
        self.statusbar.SetStatusWidths([-4, -4])
        self.statusbar.SetStatusText("", 0)
        self.statusbar.SetStatusText("", 1)

        self._leftWindow1 = wx.SashLayoutWindow(self, 101, wx.DefaultPosition,
                                                wx.Size(200, 1000), wx.NO_BORDER |
                                                wx.SW_3D | wx.CLIP_CHILDREN)

        self._leftWindow1.SetDefaultSize(wx.Size(220, 1000))
        self._leftWindow1.SetOrientation(wx.LAYOUT_VERTICAL)
        self._leftWindow1.SetAlignment(wx.LAYOUT_LEFT)
        self._leftWindow1.SetSashVisible(wx.SASH_RIGHT, True)
        self._leftWindow1.SetExtraBorderSize(10)

        self._pnl = 0

        # will occupy the space not used by the Layout Algorithm
        #self.remainingSpace = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        #self.glCanvas = gl_render.LMGLCanvas(self.remainingSpace)
        self.glCanvas = gl_render.LMGLCanvas(self)

        self.ID_WINDOW_TOP = 100
        self.ID_WINDOW_LEFT1 = 101
        self.ID_WINDOW_RIGHT1 = 102
        self.ID_WINDOW_BOTTOM = 103
    
        self._leftWindow1.Bind(wx.EVT_SASH_DRAGGED_RANGE, self.OnFoldPanelBarDrag,
                               id=100, id2=103)
        self.Bind(wx.EVT_SIZE, self.OnSize)

        
        self.dataSources = []
        self.selectedDataSource = None
        self.filterKeys = {'error_x': (0,30), 'A':(0,30), 'sig' : (150/2.35, 350/2.35)}

        self.filter = None

        self.CreateFoldPanel(fpb.FPB_SINGLE_FOLD)
        

    def OnSize(self, event):

        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        event.Skip()
        

    def OnQuit(self, event):
 
        self.Destroy()


    def OnAbout(self, event):

        msg = "PYME Visualise\n\n Visualisation of localisation microscopy data\nDavid Baddeley 2009"
              
        dlg = wx.MessageDialog(self, msg, "About PYME Visualise",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.SetFont(wx.Font(8, wx.NORMAL, wx.NORMAL, wx.NORMAL, False, "Verdana"))
        dlg.ShowModal()
        dlg.Destroy()


    def OnToggleWindow(self, event):
        
        self._leftWindow1.Show(not self._leftWindow1.IsShown())
        # Leaves bits of itself behind sometimes
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        self.glCanvas.Refresh()

        event.Skip()
        

    def OnFoldPanelBarDrag(self, event):

        if event.GetDragStatus() == wx.SASH_STATUS_OUT_OF_RANGE:
            return

        if event.GetId() == self.ID_WINDOW_LEFT1:
            self._leftWindow1.SetDefaultSize(wx.Size(event.GetDragRect().width, 1000))


        # Leaves bits of itself behind sometimes
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        self.glCanvas.Refresh()

        event.Skip()
        

    def CreateFoldPanel(self, fpb_flags):

        # delete earlier panel
        self._leftWindow1.DestroyChildren()

        # recreate the foldpanelbar

        self._pnl = fpb.FoldPanelBar(self._leftWindow1, -1, wx.DefaultPosition,
                                     wx.Size(-1,-1), fpb.FPB_DEFAULT_STYLE, fpb_flags)

        self.Images = wx.ImageList(16,16)
        self.Images.Add(GetExpandedIconBitmap())
        self.Images.Add(GetCollapsedIconBitmap())
            
        self.GenDataSourcePanel()
        self.GenFilterPanel()

        

       

        #item = self._pnl.AddFoldPanel("Filters", False, foldIcons=self.Images)
        item = self._pnl.AddFoldPanel("Visualisation", False, foldIcons=self.Images)
        wx.LayoutAlgorithm().LayoutWindow(self, self.glCanvas)
        self.glCanvas.Refresh()


    def GenDataSourcePanel(self):
        item = self._pnl.AddFoldPanel("Data Source", collapsed=True,
                                      foldIcons=self.Images)
        
        self.dsRadioIds = []
        for ds in self.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.setValue(ds == self.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            self._pnl.AddFoldPanelWindow(item, rb, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10) 


    def GenDisplayPanel(self):
        item = self._pnl.AddFoldPanel("Display", collapsed=True,
                                      foldIcons=self.Images)
        
        self.dsRadioIds = []
        for ds in self.dataSources:
            rbid = wx.NewId()
            self.dsRadioIds.append(rbid)
            rb = wx.RadioButton(item, rbid, ds._name)
            rb.setValue(ds == self.selectedDataSource)

            rb.Bind(wx.EVT_RADIOBUTTON, self.OnSourceChange)
            self._pnl.AddFoldPanelWindow(item, rb, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10) 

    def OnSourceChange(self, event):
        dsind = self.dsRadioIds.index(event.GetID())
        self.selectedDataSource = self.dataSources[dsind]
        self.RegenFilter()
            
    def GenFilterPanel(self):
        item = self._pnl.AddFoldPanel("Filter", collapsed=True,
                                      foldIcons=self.Images)

        self.lFiltKeys = wx.ListCtrl(item, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 200))

        self._pnl.AddFoldPanelWindow(item, self.lFiltKeys, fpb.FPB_ALIGN_WIDTH, fpb.FPB_DEFAULT_SPACING, 10)

        self.lFiltKeys.InsertColumn(0, 'Key')
        self.lFiltKeys.InsertColumn(1, 'Min')
        self.lFiltKeys.InsertColumn(2, 'Max')

        for key, value in self.filterKeys.items():
            ind = self.lFiltKeys.InsertStringItem(sys.maxint, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % value[0])
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % value[1])

        self.lFiltKeys.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(2, wx.LIST_AUTOSIZE)

        # only do this part the first time so the events are only bound once
        if not hasattr(self, "ID_FILT_ADD"):
            self.ID_FILT_ADD = wx.NewId()
            self.ID_FILT_DELETE = wx.NewId()
            self.ID_FILT_EDIT = wx.NewId()
           
            self.Bind(wx.EVT_MENU, self.OnFilterAdd, id=self.ID_FILT_ADD)
            self.Bind(wx.EVT_MENU, self.OnFilterDelete, id=self.ID_FILT_DELETE)
            self.Bind(wx.EVT_MENU, self.OnFilterEdit, id=self.ID_FILT_EDIT)

        # for wxMSW
        self.lFiltKeys.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnFilterListRightClick)

        # for wxGTK
        self.lFiltKeys.Bind(wx.EVT_RIGHT_UP, self.OnFilterListRightClick)
        
    def OnFilterListRightClick(self, event):

        x = event.GetX()
        y = event.GetY()

        item, flags = self.lFiltKeys.HitTest((x, y))

 
        # make a menu
        menu = wx.Menu()
        # add some items
        menu.Append(self.ID_FILT_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.currentFilterItem = item
            self.lFiltKeys.Select(item)
        
            menu.Append(self.ID_FILT_DELETE, "Delete")
            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

    def OnFilterItemSelected(self, event):
        self.currentFilterItem = event.m_itemIndex

        event.Skip()

    def OnFilterItemDeselected(self, event):
        self.currentFilterItem = None

        event.Skip()

    def OnFilterAdd(self, event):
        #key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        possibleKeys = []
        if not self.selectedDataSource == None:
            possibleKeys = self.selectedDataSource.keys()

        dlg = editFilterDialog.FilterEditDialog(self, mode='new', possibleKeys=possibleKeys)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            minVal = float(dlg.tMin.GetValue())
            maxVal = float(dlg.tMax.GetValue())

            key = dlg.cbKey.GetValue()

            if key == "":
                return

            self.filterKeys[key] = (minVal, maxVal)

            ind = self.lFiltKeys.InsertStringItem(sys.maxint, key)
            self.lFiltKeys.SetStringItem(ind,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(ind,2, '%3.2f' % maxVal)

        self.RegenFilter()

    def OnFilterDelete(self, event):
        it = self.lFiltKeys.GetItem(self.currentFilterItem)
        self.lFiltKeys.DeleteItem(self.currentFilterItem)
        self.filterKeys.pop(it.GetText())

        self.RegenFilter()
        
    def OnFilterEdit(self, event):
        key = self.lFiltKeys.GetItem(self.currentFilterItem).GetText()

        dlg = editFilterDialog.FilterEditDialog(self, mode='edit', possibleKeys=[], key=key, minVal=self.filterKeys[key][0], maxVal=self.filterKeys[key][1])

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            minVal = float(dlg.tMin.GetValue())
            maxVal = float(dlg.tMax.GetValue())

            self.filterKeys[key] = (minVal, maxVal)

            self.lFiltKeys.SetStringItem(self.currentFilterItem,1, '%3.2f' % minVal)
            self.lFiltKeys.SetStringItem(self.currentFilterItem,2, '%3.2f' % maxVal)
        self.RegenFilter()

    def RegenFilter(self):
        if not self.selectedDataSource == None:
            self.filter = inpFilt.resultsFilter(self.selectedDataSource, **self.filterKeys)

    def CreateMenuBar(self):

        # Make a menubar
        file_menu = wx.Menu()

        ID_OPEN = wx.NewId()
        ID_QUIT = wx.NewId()
        
        ID_GEN_JIT_TRI = wx.NewId()
        ID_GEN_QUADS = wx.NewId()

        ID_GEN_GAUSS = wx.NewId()
        ID_GEN_HIST = wx.NewId()

        ID_GEN_CURRENT = wx.NewId()

        ID_TOGGLE_SETTINGS = wx.NewId()

        ID_ABOUT = wx.NewId()
        
        
        file_menu.Append(ID_OPEN, "&Open")
        
        file_menu.AppendSeparator()
        
        file_menu.Append(ID_QUIT, "&Exit")

        gen_menu = wx.Menu()
        gen_menu.Append(ID_GEN_CURRENT, "&Current")
        
        gen_menu.AppendSeparator()
        gen_menu.Append(ID_GEN_GAUSS, "&Gaussian")
        gen_menu.Append(ID_GEN_HIST, "&Histogram")

        gen_menu.AppendSeparator()
        gen_menu.Append(ID_GEN_JIT_TRI, "&Triangulation")
        gen_menu.Append(ID_GEN_QUADS, "&QuadTree")
        
        view_menu = wx.Menu()
        view_menu.AppendCheckItem(ID_TOGGLE_SETTINGS, "Show Settings")
        view_menu.Check(ID_TOGGLE_SETTINGS, True)
        

        help_menu = wx.Menu()
        help_menu.Append(ID_ABOUT, "&About")

        menu_bar = wx.MenuBar()

        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(gen_menu, "&Generate Image")
        menu_bar.Append(view_menu, "&View")
        

            
        menu_bar.Append(help_menu, "&Help")

        self.Bind(wx.EVT_MENU, self.OnAbout, id=ID_ABOUT)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=ID_QUIT)
        self.Bind(wx.EVT_MENU, self.OnToggleWindow, id=ID_TOGGLE_SETTINGS)


        return menu_bar





class VisGuiApp(wx.App):
    def __init__(self, options, *args):
        self.options = options
        wx.App.__init__(self, *args)
        
        
    def OnInit(self):
        wx.InitAllImageHandlers()
        self.main = VisGUIFrame(None)#, self.options)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True


def main():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--init-file", dest="initFile", help="Read initialisation from file [defaults to init.py]", metavar="FILE")
        
    (options, args) = parser.parse_args()
    
    application = VisGuiApp(options, 0)
    application.MainLoop()

if __name__ == '__main__':
    if False: #not '__IPYTHON__' in dir(__builtins__):
        main()
    else:

        visFr = VisGUIFrame(None)
        visFr.Show()

