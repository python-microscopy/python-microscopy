
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, CStr
from PYME.IO import tabular

@register_module('FlagPiezoMovement')
class FlagPiezoMovement(ModuleBase):
    """
    """
    input_name = Input('localizations')
    input_events = Input('')
    column_name = CStr('piezo_moving')
    output_name = Output('motion_flagged')

    def execute(self, namespace):
        from PYME.Analysis.points import piezo_movement_correction

        points = namespace[self.input_name]
        events = namespace[self.input_events] if self.input_events != '' else points.events

        mapped = tabular.MappingFilter(points)

        moving = piezo_movement_correction.flag_piezo_movement(points['t'], events, points.mdh)

        mapped.addColumn(self.column_name, moving)

        mapped.mdh = points.mdh
        mapped.events = points.events

        namespace[self.output_name] = mapped

@register_module('CorrectFocusTargets')
class CorrectFocusTargets(ModuleBase):
    """
    """
    input_name = Input('localizations')
    input_events = Input('')
    column_name = CStr('focus')
    output_name = Output('target_focus_corrected')

    def execute(self, namespace):
        from PYME.Analysis.points import piezo_movement_correction

        points = namespace[self.input_name]
        events = namespace[self.input_events] if self.input_events != '' else points.events

        mapped = tabular.MappingFilter(points)

        focus = piezo_movement_correction.correct_target_positions(points['t'], events, points.mdh)

        mapped.addColumn(self.column_name, focus)

        mapped.mdh = points.mdh
        mapped.events = points.events

        namespace[self.output_name] = mapped
