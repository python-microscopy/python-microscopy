
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
        from PYME.recipes.event_corrections import FlagPiezoMovement
        recipe = self.pipeline.recipe

        # hold off auto-running the recipe until we configure things
        recipe.trait_set(execute_on_invalidation=False)
        try:
            # make sure we have events accessible
            try:
                recipe.namespace[self.pipeline.selectedDataSourceKey].events
            except AttributeError:
                logger.debug('Using pipeline.events for flagging piezo movement in %s data source' %
                             self.pipeline.selectedDataSourceKey)
                recipe.namespace[self.pipeline.selectedDataSourceKey].events = self.pipeline.events
            mod = FlagPiezoMovement(recipe, input_name=self.pipeline.selectedDataSourceKey,
                                    input_events='')

            recipe.add_module(mod)
            if not recipe.configure_traits(view=recipe.pipeline_view, kind='modal'):
                return

            recipe.execute()
            self.pipeline.selectDataSource('motion_flagged')
        finally:  # make sure that we configure the pipeline recipe as it was
            recipe.trait_set(execute_on_invalidation=True)


def Plug(vis_frame):
    """Plugs this module into the gui"""
    vis_frame.event_corrections = EventCorrections(vis_frame)
