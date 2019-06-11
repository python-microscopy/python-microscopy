from . import manualFoldPanel

LAYER_CAPTION_STYLE = {
'HEIGHT'              : 20,
'FONT_COLOUR'         : 'BLACK',
#'FONT_WEIGHT' : wx.BOLD,
#'FONT_SIZE'           : 12,
'CAPTION_INDENT'      : 5,
'BACKGROUND_COLOUR_1' : (198, 198, 198), #default AUI caption colours
'BACKGROUND_COLOUR_2' : (226, 226, 226),
'INACTIVE_PIN_COLOUR' : (170, 170, 170),
'ACTIVE_PIN_COLOUR'   : (0, 0, 0),
'ELLIPSES_COLOUR'     : (170, 170, 170),
'ELLIPSES_RADIUS'     : 2,
'HAS_PIN' : False,
}

class LayerCaptionBar(manualFoldPanel.CaptionBar):
    pass

class LayerFoldingPane(manualFoldPanel.foldingPane):
    pass