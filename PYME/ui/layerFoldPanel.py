from . import manualFoldPanel
import PYME.resources
import wx

import logging
logger = logging.getLogger(__name__)

LAYER_CAPTION_STYLE = {
'HEIGHT'              : 20,
'FONT_COLOUR'         : 'BLACK',
#'FONT_WEIGHT' : wx.BOLD,
#'FONT_SIZE'           : 12,
'CAPTION_INDENT'      : 5,
'BACKGROUND_COLOUR_1' : (198, 198, 198 + 0), #default AUI caption colours
'BACKGROUND_COLOUR_2' : (226, 226, 226),
'INACTIVE_PIN_COLOUR' : (170, 170, 170),
'ACTIVE_PIN_COLOUR'   : (0, 0, 0),
'ACTIVE_EYE_COLOUR'   : (50, 50, 198),
'ELLIPSES_COLOUR'     : (170, 170, 170),
'ELLIPSES_RADIUS'     : 2,
'HAS_PIN' : False,
}

LAYER_CAPTION_STYLE = {
'HEIGHT'              : 20,
'FONT_COLOUR'         : 'BLACK',
#'FONT_WEIGHT' : wx.BOLD,
#'FONT_SIZE'           : 12,
'CAPTION_INDENT'      : 5,
'BACKGROUND_COLOUR_1' : (198, 198, 198 + 0), #default AUI caption colours
'BACKGROUND_COLOUR_2' : (226, 226, 226),
'INACTIVE_PIN_COLOUR' : (170, 170, 170),
'ACTIVE_PIN_COLOUR'   : (0, 0, 0),
'ACTIVE_EYE_COLOUR'   : (50, 50, 198),
'ELLIPSES_COLOUR'     : (170, 170, 170),
'ELLIPSES_RADIUS'     : 2,
'HAS_PIN' : True,
}

LAYER_CAPTION_STYLE_DARK = {
    'HEIGHT'              : 20,
    'FONT_COLOUR'         : 'WHITE',
    #'FONT_WEIGHT' : wx.BOLD,
    #'FONT_SIZE'           : 12,
    'CAPTION_INDENT'      : 5,
    'BACKGROUND_COLOUR_1' : (38, 38, 38), #default AUI caption colours
    'BACKGROUND_COLOUR_2' : (82, 82, 82),
    'INACTIVE_PIN_COLOUR' : (125, 125, 125),
    'ACTIVE_PIN_COLOUR'   : (255, 255, 255),
    'ACTIVE_EYE_COLOUR'   : (50, 50, 198),
    'ELLIPSES_COLOUR'     : (125, 125, 125),
    'ELLIPSES_RADIUS'     : 2,
    'HAS_PIN' : True,
    }


#modified from https://www.iconfinder.com/icons/1608688/eye_icon
eye_bits = b'\xff\xff\xff\xff\xff\xff\x3f\xfc\x0f\xf0\xc3\xc0\x29\x90\x2c\x30' \
           b'\x0e\x70\x1c\x38\x39\x9c\xf3\xcf\xc7\xe3\x1f\xf8\xff\xff\xff\xff'


class LUTBitmap(manualFoldPanel.CaptionButton):
    def __init__(self, layer):
        manualFoldPanel.CaptionButton.__init__(self, None)
        self._layer = layer
        
    @property
    def size(self):
        return (30, 10)
    
    @property
    def _active_bitmap(self):
        import numpy as np
        # from pylab import cm
        from PYME.misc.colormaps import cm
        x = np.linspace(0, 1, 30)

        if isinstance(self._layer.cmap, str):
            img = (255*cm[self._layer.cmap](np.ones(10)[:, None]*x[None, :]))[:,:,:3].astype('uint8')
        else:
            # assume an actual colourmap instance - TODO make the check explicit on a colormap base class??
            img = (255*self._layer.cmap(np.ones(10)[:, None]*x[None, :]))[:,:,:3].astype('uint8')
        
        #print img.shape, img.strides

        
        return wx.Bitmap.FromBuffer(30, 10, img)
    
    @_active_bitmap.setter
    def _active_bitmap(self, val):
        pass
        
        
class LayerCaptionBar(manualFoldPanel.CaptionBar):
    def __init__(self, parent, layer, **kwargs):
        from PYME.LMVis.layers import pointcloud, mesh, tracks
        self._caption= ""

        manualFoldPanel.CaptionBar.__init__(self, parent, **kwargs)
        
        self._layer = layer

        self._inactive_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['INACTIVE_PIN_COLOUR']))
        self._active_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['ACTIVE_EYE_COLOUR']))

        
        
        self.buttons.append(manualFoldPanel.CaptionButton(self._active_eye_bitmap, self._inactive_eye_bitmap,
                                          show_fcn=lambda: True,
                                          active_fcn=lambda: self._layer.visible,
                                          onclick=lambda: self.ToggleVis()))

        self.buttons.append(LUTBitmap(self._layer))
        
        self._icon = None
        
        if isinstance(layer, pointcloud.PointCloudRenderLayer):
            self._icon = wx.Bitmap(PYME.resources.getIconPath('points.png'))
        elif isinstance(layer, mesh.TriangleRenderLayer):
            self._icon = wx.Bitmap(PYME.resources.getIconPath('mesh.png'))
        elif isinstance(layer, tracks.TrackRenderLayer):
            self._icon = wx.Bitmap(PYME.resources.getIconPath('tracks.png'))

        self._layer.on_trait_change(self._refresh)

        self.Bind(wx.EVT_CLOSE, self._on_close)
        
        
        
    # def OnLeftClick(self, event):
    #     if wx.Rect(*self.eyeButtonRect).Contains(event.GetPosition()):
    #         self.ToggleVis()
    #     else:
    #         super(LayerCaptionBar, self).OnLeftClick(event)
    
    def _refresh(self, event=None):
        try:
            self.Refresh()
        except:
            logger.exception('Error refreshing layer caption')
        
    def _on_close(self, event):
        self._layer.on_trait_change(self._refresh, remove=True)
        #logger.debug('Removed layer handler')
        self.Destroy()
    
    def ToggleVis(self):
         self._layer.visible = not self._layer.visible
        
    def DrawIcon(self, gc):
        if self._icon is None:
            return 0
        else:
            w, h = self._icon.GetWidth(), self._icon.GetHeight()
            y0 = self.style['HEIGHT'] / 2. - h / 2.
            gc.DrawBitmap(self._icon, 0, y0, w, h)
            
            return w
        
    @property
    def caption(self):
        return self._caption + ' - ' + getattr(self._layer, 'dsname', '')
    
    @caption.setter
    def caption(self, caption):
        self._caption = caption

class LayerFoldingPane(manualFoldPanel.foldingPane):
    def __init__(self, parent, layer=None, **kwargs):
        self._layer = layer
        manualFoldPanel.foldingPane.__init__(self, parent, **kwargs)
        
    def _create_caption_bar(self):
        """ This is over-rideable in derived classes so that they can implement their own caption bars"""
        cbstyle = LAYER_CAPTION_STYLE
        
        try:
            if wx.SystemSettings.GetAppearance().IsDark():
                cbstyle=LAYER_CAPTION_STYLE_DARK
        except AttributeError:
            # wx < 4.1 doesn't have GetAppearance
            pass
        
        
        return LayerCaptionBar(self, layer=self._layer, caption=self.caption, cbstyle=cbstyle)


class _ChannelLayer(object):
    """ Proxy so we can spoof a layer for a channel 
    TODO - Revisit
    """

    def __init__(self, display_opts, chan):
        self._do = display_opts # type: PYME.DSView.displayOptions.DisplayOpts
        self._chan = chan # type: int

    @property
    def cmap(self):
        return self._do.cmaps[self._chan]

class ChannelCaptionBar(manualFoldPanel.CaptionBar):
    def __init__(self, parent, display_opts, chan, **kwargs):
        self._do = display_opts
        self._chan = chan

        manualFoldPanel.CaptionBar.__init__(self, parent, **kwargs)

        self._inactive_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['INACTIVE_PIN_COLOUR']))
        self._active_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['ACTIVE_EYE_COLOUR']))
        
        self.buttons.append(manualFoldPanel.CaptionButton(self._active_eye_bitmap, self._inactive_eye_bitmap,
                                          show_fcn=lambda: True,
                                          active_fcn=lambda: self._do.show[self._chan],
                                          onclick=lambda: self.ToggleVis()))

        self.buttons.append(LUTBitmap(_ChannelLayer(self._do, self._chan)))

    def ToggleVis(self):
        if self._do.show[self._chan]:
            self._do.Show(self._chan, False)
        else:
            self._do.Show(self._chan, True)
        
class ChannelFoldingPane(manualFoldPanel.foldingPane):
    def __init__(self, parent, display_opts, chan, **kwargs):
        self._do = display_opts
        self._chan = chan
        manualFoldPanel.foldingPane.__init__(self, parent, **kwargs)
        
    def _create_caption_bar(self):
        """ This is over-rideable in derived classes so that they can implement their own caption bars"""
        return ChannelCaptionBar(self, display_opts=self._do, chan=self._chan, caption=self.caption, cbstyle=LAYER_CAPTION_STYLE)