import numpy as np
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

        vis_fr.AddMenuItem('Annotation', 'Label points', self.apply_labels_to_points)

        self.vis_fr = vis_fr

        self._recipe_apply_mod = None

    def _update_view(self):
        self.vis_fr.Refresh()

    def apply_labels_to_points(self, event=None):
        from PYME.IO import tabular
        from PYME.recipes import machine_learning
        
        # FIXME - this is really, really crude
        # fix by a) separating out annotation class logic from UI
        # and b) inserting annotations as a new datasource into pipeline
        # namespace along with an ApplyPointAnnotations recipe module

        pipeline=self.vis_fr.pipeline

        # add (or replace) annotations in recipe namespace
        # TODO - is there a cleaner way to do this, and/or use a funky name?
        pipeline.recipe.namespace['annotations'] = self._annotations

        if self._recipe_apply_mod is None:
            self._recipe_apply_mod = machine_learning.AnnotatePoints(parent=pipeline.recipe, 
                                                                     inputLocalisations=pipeline.selectedDataSourceKey,
                                                                     inputAnnotations='annotations',
                                                                     outputName='labeled')

            pipeline.recipe.add_modules_and_execute([self._recipe_apply_mod, ])
        else:
            self._recipe_apply_mod.invalidate_parent()
        

        pipeline.selectDataSource('labeled')
        #self.vis_fr.pipeline.Rebuild()


def Plug(vis_fr):
    return Annotator(vis_fr)