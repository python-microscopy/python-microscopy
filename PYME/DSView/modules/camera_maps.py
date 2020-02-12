import wx
import os.path

def on_map(image, parentWindow=None, glCanvas=None):
    from PYME.Analysis import gen_sCMOS_maps
    from PYME.recipes.processing import DarkAndVarianceMap
    from PYME.DSView import ViewIm3D

    dvm = DarkAndVarianceMap()
    if dvm.configure_traits(kind='modal'):
        namespace = {'input' : image}
        dvm.execute(namespace)

        ViewIm3D(namespace['dark'], title='Dark Map', parent=parentWindow, glCanvas=glCanvas)
        ViewIm3D(namespace['variance'], title='Variance Map', parent=parentWindow, glCanvas=glCanvas)

        darkmapname = gen_sCMOS_maps.mkDefaultPath('dark', image.mdh)
        varmapname = gen_sCMOS_maps.mkDefaultPath('variance', image.mdh)

        dark_dlg = wx.FileDialog(parentWindow, message="Save dark map as...",
                                 defaultDir=os.path.dirname(darkmapname),
                                 defaultFile=os.path.basename(darkmapname), 
                                 wildcard='TIFF (*.tif)|*.tif', 
                                 style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dark_dlg.ShowModal() == wx.ID_OK:
            darkfn = dark_dlg.GetPath()
            namespace['dark'].Save(filename=darkfn)

        var_dlg = wx.FileDialog(parentWindow, message="Save variance map as...",  
                                 defaultDir=os.path.dirname(varmapname),
                                 defaultFile=os.path.basename(varmapname), 
                                 wildcard='TIFF (*.tif)|*.tif', 
                                 style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if var_dlg.ShowModal() == wx.ID_OK:
            varfn = var_dlg.GetPath()
            namespace['variance'].Save(filename=varfn)

def Plug(dsviewer):
    dsviewer.AddMenuItem(menuName='Processing', itemName='Create Dark and Variance Maps', itemCallback = lambda e : on_map(dsviewer.image, dsviewer, dsviewer.glCanvas))
