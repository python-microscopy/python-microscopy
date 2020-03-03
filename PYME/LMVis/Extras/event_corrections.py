
import logging

logger = logging.getLogger(__name__)


class EventCorrections(object):
    """
    Class with methods to flag localizations for filtering with respect to information stored in the events table
    """
    def __init__(self, vis_frame):
        self.pipeline = vis_frame.pipeline

        logging.debug('Adding menu items for event-based corrections')

        vis_frame.AddMenuItem('Corrections', 'Flag piezo movement', self.OnFlagPiezoMovement,
                              helpText='Using PiezoOnTarget events, flag localizations with uncertain z positions')

    def OnFlagPiezoMovement(self, event=None):
        """

        """
        from PYME.recipes.localisations import FlagPiezoMovement
        recipe = self.pipeline.recipe

        # hold off auto-running the recipe until we configure things
        recipe.trait_set(execute_on_invalidation=False)
        try:
            mod = FlagPiezoMovement(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                    output_name='motion_flagged')

            recipe.add_module(mod)
            if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
                return

            recipe.execute()
            self.pipeline.selectDataSource('motion_flagged')
        finally:  # make sure that we configure the pipeline recipe as it was
            recipe.trait_set(execute_on_invalidation=True)

    def OnCorrectFocusTargets(self, event=None):
        """

        """
        from PYME.recipes.localisations import CorrectFocusTargets
        recipe = self.pipeline.recipe

        # hold off auto-running the recipe until we configure things
        recipe.trait_set(execute_on_invalidation=False)
        try:
            mod = CorrectFocusTargets(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                    output_name='target_focus_corrected')

            recipe.add_module(mod)
            if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
                return

            recipe.execute()
            self.pipeline.selectDataSource('target_focus_corrected')
        finally:  # make sure that we configure the pipeline recipe as it was
            recipe.trait_set(execute_on_invalidation=True)


def Plug(vis_frame):
    """Plugs this module into the gui"""
    vis_frame.event_corrections = EventCorrections(vis_frame)
