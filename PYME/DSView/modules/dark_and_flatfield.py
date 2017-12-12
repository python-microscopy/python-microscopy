def dark(dsviewer):
    from PYME.DSView import ViewIm3D, ImageStack
    
    im = ImageStack(dsviewer.image.data[:, :, :].astype('f').mean(2), mdh=dsviewer.image.mdh)
    
    ViewIm3D(im)


def flat(dsviewer):
    from PYME.DSView import ViewIm3D, ImageStack
    import numpy as np
    
    dark = ImageStack(load_prompt="Select corresponding DARK image (should have been calculated first)").data[:,:,0].squeeze().astype('f')
    
    illum = 0*dark
    
    for i in range(dsviewer.image.data.shape[2]):
        illum += dsviewer.image.data[:, :, i].squeeze().astype('f') - dark
        
    illum /= dsviewer.image.data.shape[2]
    
    im = ImageStack(illum.mean()/illum, mdh=dsviewer.image.mdh)
    
    ViewIm3D(im)


def Plug(dsviewer):
    dsviewer.AddMenuItem('Processing', "Extract dark calibration image from series", lambda e: dark(dsviewer))
    dsviewer.AddMenuItem('Processing', "Extract flatfield calibration image from series", lambda e: flat(dsviewer))
    #dsviewer.AddMenuItem('Processing', "Extract dark calibration image from series", lambda e: dark(dsviewer))
    