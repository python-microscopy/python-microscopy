from .visgui import GLImageView

def Plug(dsviewer):
    # gl canvas doesn't currently work for 3D images, crashes on linux
    dsviewer._gl_im = GLImageView(dsviewer, image=dsviewer.image, glCanvas=dsviewer.glCanvas, display_opts=dsviewer.do)
    dsviewer.AddPage(page=dsviewer._gl_im, select=True, caption='GLComp')