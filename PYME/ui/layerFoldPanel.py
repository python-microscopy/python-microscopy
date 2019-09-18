from . import manualFoldPanel
import wx

LAYER_CAPTION_STYLE = {
'HEIGHT'              : 20,
'FONT_COLOUR'         : 'BLACK',
#'FONT_WEIGHT' : wx.BOLD,
#'FONT_SIZE'           : 12,
'CAPTION_INDENT'      : 5,
'BACKGROUND_COLOUR_1' : (198, 198, 198 + 20), #default AUI caption colours
'BACKGROUND_COLOUR_2' : (226, 226, 226),
'INACTIVE_PIN_COLOUR' : (170, 170, 170),
'ACTIVE_PIN_COLOUR'   : (0, 0, 0),
'ACTIVE_EYE_COLOUR'   : (0, 0, 0),
'ELLIPSES_COLOUR'     : (170, 170, 170),
'ELLIPSES_RADIUS'     : 2,
'HAS_PIN' : False,
}

eye_bits = b'\xff\xff\xff\xff\xff\xff\xff\xff\x1f\xf8\xc7\xe3\x63\xc6\x6d\xb6' \
           b'\x6d\xb6\xe3\xc7\xc7\xe3\x1f\xf8\xff\xff\xff\xff\xff\xff\xff\xff'

class LayerCaptionBar(manualFoldPanel.CaptionBar):
    def __init__(self, parent, layer, **kwargs):
        self._caption= ""

        manualFoldPanel.CaptionBar.__init__(self, parent, **kwargs)
        
        self._layer = layer

        self._inactive_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['INACTIVE_PIN_COLOUR']))
        self._active_eye_bitmap = manualFoldPanel.BitmapFromBits(eye_bits, 16, 16, manualFoldPanel.ColourFromStyle(self.style['ACTIVE_EYE_COLOUR']))

        self.buttons.append(manualFoldPanel.CaptionButton(self._active_eye_bitmap, self._inactive_eye_bitmap,
                                          show_fcn=lambda: True,
                                          active_fcn=lambda: self._layer.visible,
                                          onclick=lambda: self.ToggleVis()))
        
    # def OnLeftClick(self, event):
    #     if wx.Rect(*self.eyeButtonRect).Contains(event.GetPosition()):
    #         self.ToggleVis()
    #     else:
    #         super(LayerCaptionBar, self).OnLeftClick(event)
    #
    def ToggleVis(self):
         self._layer.visible = not self._layer.visible
        
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
        return LayerCaptionBar(self, layer=self._layer, caption=self.caption, cbstyle=LAYER_CAPTION_STYLE)