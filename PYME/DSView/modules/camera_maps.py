import wx
import os.path

def on_map(image, parentWindow=None, glCanvas=None):
    from PYME.Analysis import gen_sCMOS_maps
    from PYME.DSView import ViewIm3D

    im_dark, im_variance = gen_sCMOS_maps.generate_maps(image, 0, -1)

    ViewIm3D(im_dark, title='Dark Map', parent=parentWindow, glCanvas=glCanvas)
    ViewIm3D(im_variance, title='Variance Map', parent=parentWindow, glCanvas=glCanvas)

    # TODO - check if we generated the map from a sub ROI and deflect from the default path if needed ???
    darkmapname = gen_sCMOS_maps.mkDefaultPath('dark', image.mdh)
    varmapname = gen_sCMOS_maps.mkDefaultPath('variance', image.mdh)

    dark_dlg = wx.FileDialog(parentWindow, message="Save dark map as...",
                             defaultDir=os.path.dirname(darkmapname),
                             defaultFile=os.path.basename(darkmapname), 
                             wildcard='TIFF (*.tif)|*.tif', 
                             style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
    
    if dark_dlg.ShowModal() == wx.ID_OK:
        darkfn = dark_dlg.GetPath()
        im_dark.Save(filename=darkfn)

    var_dlg = wx.FileDialog(parentWindow, message="Save variance map as...",  
                             defaultDir=os.path.dirname(varmapname),
                             defaultFile=os.path.basename(varmapname), 
                             wildcard='TIFF (*.tif)|*.tif', 
                             style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
    
    if var_dlg.ShowModal() == wx.ID_OK:
        varfn = var_dlg.GetPath()
        im_variance.Save(filename=varfn)

def Plug(dsviewer):
    dsviewer.AddMenuItem(menuName='Processing', itemName='Create Dark and Variance Maps', itemCallback = lambda e : on_map(dsviewer.image, dsviewer, dsviewer.glCanvas))
