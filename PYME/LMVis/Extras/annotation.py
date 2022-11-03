from PYME.DSView.modules import annotation as ann
from PYME.LMVis.layers.OverlayLayer import OverlayLayer

class AnnotationOverlayLayer(OverlayLayer):

    """
    This OverlayLayer creates the lines of a box.
    The dimensions of the box are determined by the selection_settings
    """
    def __init__(self, annotator, **kwargs):
        super(AnnotationOverlayLayer, self).__init__([0, 0], **kwargs)
        self._annotator = annotator

    def render(self, gl_canvas):
        """

        Parameters
        ----------
        gl_canvas
            zc is used to set the z value of the Overlay
        Returns
        -------

        """
        if not self.visible:
            return
        
        self._clear_shader_clipping(gl_canvas)
        with self.get_shader_program(gl_canvas):
            self._annotator.render(gl_canvas)

class Annotator(ann.AnnotateBase):
    def __init__(self, vis_fr):
        ann.AnnotateBase.__init__(self, vis_fr, vis_fr.glCanvas.selectionSettings, minimize=True)
        vis_fr.glCanvas.overlays.append(AnnotationOverlayLayer(self))

def Plug(vis_fr):
    return Annotator(vis_fr)