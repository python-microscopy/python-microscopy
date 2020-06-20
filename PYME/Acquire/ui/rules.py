
import wx
from  PYME.ui import manualFoldPanel
from PYME.cluster.rules import LocalizationRule

class RuleChainListCtrl(wx.ListCtrl):
    def __init__(self, rule_chain, wx_parent):
        """

        Parameters
        ----------
        rule_chain: PYME.cluster.rules.RuleChain
        wx_parent
        """

        wx.ListCtrl.__init__(self, wx_parent, style=wx.LC_REPORT | wx.BORDER_SUNKEN | wx.LC_VIRTUAL | wx.LC_VRULES)
        self._rule_chain = rule_chain

        self.InsertColumn(0, 'Type', width=50)
        self.InsertColumn(1, 'ID', width=100)

    @property
    def localization_rule_indices(self):
        return [ind for ind, rule in enumerate(self._rule_chain) if rule.template['type'] == 'localization']

    def add_rule(self, rule, index=None):
        """

        Parameters
        ----------
        rule: PYME.cluster.rules.Rule

        Returns
        -------

        """
        if index is None:
            self._rule_chain.append(rule)
        else:
            self._rule_chain.insert(rule, index)

        self.update_list()

    def replace_rule(self, rule, index):
        self._rule_chain[index] = rule

    def OnGetItemText(self, item, col):
        """
        Note that this is overriding the wxListCtrl method as required for wxLC_VIRTUAL style
        Parameters
        ----------
        item : long
            wx list item
        col : long
            column specifier for wxListCtrl

        Returns
        -------
        str : Returns string of column 'col' for item 'item'

        """
        try:
            if col == 0:
                return self._rule_chain[item].template['type']
            if col == 1:
                return str(item)
        except:
            return ''
        else:
            return ''

    def update_list(self, sender=None, **kwargs):
        self.SetItemCount(len(self._rule_chain))
        self.Update()
        self.Refresh()

    def delete_rules(self, indices=None):
        selected_indices = self.get_selected_items() if indices is None else indices

        for ind in reversed(sorted(selected_indices)):  # delete in reverse order so we can pop without changing indices
            self._rule_chain.pop(ind)
            self.DeleteItem(ind)

        self.update_list()

    def get_selected_items(self):
        selection = []
        current = -1
        next = 0
        while next != -1:
            next = self.get_next_selected(current)
            if next != -1:
                selection.append(next)
                current = next
        return selection

    def get_next_selected(self, current):
        return self.GetNextItem(current, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)

    def clear_localization_rules(self):
        self.delete_rules(self.localization_rule_indices)



class ChainedRulePanel(wx.Panel):
    _RULE_LAUNCH_MODES = ['off', 'spool start', 'spool stop']
    def __init__(self, parent, rule_chain, recipe_manager, spool_controller):
        """

        Parameters
        ----------
        parent:
        rule_chain: PYME.cluster.rules.RuleChain
        recipe_manager: PYME.recipes.recipeGui.RecipeManager
        spool_controller: PYME.Acquire.SpoolController.SpoolController
        """
        wx.Panel.__init__(self, parent, -1)

        self._rule_chain = rule_chain
        self._recipe_manager = recipe_manager
        self._spool_controller = spool_controller

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)

        h_sizer.Add(wx.StaticText(self, -1, 'Post automatically: '), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 2)
        self.choice_launch = wx.Choice(self, -1, choices=self._RULE_LAUNCH_MODES)
        self.choice_launch.SetSelection(0)
        self.choice_launch.Bind(wx.EVT_CHOICE, self.OnToggleAuto)
        h_sizer.Add(self.choice_launch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        v_sizer.Add(h_sizer, 0, wx.ALL | wx.EXPAND, 2)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._rule_list = RuleChainListCtrl(self._rule_chain, self)
        h_sizer.Add(self._rule_list)
        v_sizer.Add(h_sizer, 0, wx.EXPAND|wx.TOP, 0)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.button_add = wx.Button(self, -1, 'Add from Recipe Panel')
        self.button_add.Bind(wx.EVT_BUTTON, self.OnAddFromRecipePanel)
        h_sizer.Add(self.button_add, 0, wx.ALL, 2)  # todo - (disable until activeRecipe.modules) > 0

        self.button_del = wx.Button(self, -1, 'Delete')
        self.button_del.Bind(wx.EVT_BUTTON, self.OnRemove)
        h_sizer.Add(self.button_del, 0, wx.ALL, 2)

        v_sizer.Add(h_sizer)

        self.SetSizerAndFit(v_sizer)

    def OnAddFromRecipePanel(self, wx_event=None):
        from PYME.cluster.rules import RecipeRule
        if len(self._recipe_manager.activeRecipe.modules) > 0:
            rule = RecipeRule(self._recipe_manager.activeRecipe.toYAML())
            self._rule_list.add_rule(rule)

    def OnRemove(self, wx_event=None):
        self._rule_list.delete_rules()

    def OnToggleAuto(self, wx_event=None):
        mode = self.choice_launch.GetSelection()

        # todo - do we need to try/except these?
        self._spool_controller.onSpoolStart.disconnect(self.post_rules)
        self._spool_controller.onSpoolStop.disconnect(self.post_rules)

        if mode == 1:  # series start
            self._spool_controller.onSpoolStart.connect(self.post_rules)
        elif mode == 2:  # series stop
            self._spool_controller.onSpoolStop.connect(self.post_rules)

    def post_rules(self, **kwargs):
        """
        pipe input series name into rule chain and post them all

        Parameters
        ----------
        kwargs: dict
            present here to allow us to call this method through a dispatch.Signal.send

        """
        # pipe the input series name into the rule list
        series_uri = self._spool_controller.spooler.getURL()
        self._rule_chain.set_chain_input([{'input': series_uri}])
        self._rule_chain.post_all()

    @staticmethod
    def plug(main_frame, scope):
        from PYME.recipes.recipeGui import RecipeView, RecipeManager
        from PYME.cluster.rules import RuleChain

        scope._recipe_manager = RecipeManager()
        main_frame.recipe_view = RecipeView(main_frame, scope._recipe_manager)
        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Chained Recipe')

        scope._rule_chain = RuleChain(scope.spoolController._analysis_launchers)
        chained_analysis = ChainedRulePanel(main_frame, scope._rule_chain, scope._recipe_manager, scope.spoolController)

        main_frame.anPanels.append((chained_analysis, 'Chained Analysis', True))


class SMLMChainedAnalysisPanel(manualFoldPanel.foldingPane):
    def __init__(self, wx_parent, rule_chain, recipe_manager, localization_settings, spool_controller):
        """

        Parameters
        ----------
        wx_parent
        localization_settings: PYME.ui.AnalysisSettingsUI.AnalysisSettings
        rule_list_ctrl: RuleChainListCtrl
        """
        from PYME.ui.autoFoldPanel import collapsingPane
        from PYME.Acquire.ui import AnalysisSettingsUI

        manualFoldPanel.foldingPane.__init__(self, wx_parent, caption='Localization Analysis')

        # add checkbox to propagate rule to rule chain
        localization_checkbox_panel = wx.Panel(self, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.checkbox_propagate = wx.CheckBox(localization_checkbox_panel, -1, 'Chain Localization Rule')
        self.checkbox_propagate.SetValue(False)
        self.checkbox_propagate.Bind(wx.EVT_CHECKBOX, self.OnTogglePropagate)
        h_sizer.Add(self.checkbox_propagate, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        v_sizer.Add(h_sizer, 0, wx.ALL | wx.EXPAND, 2)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.checkbox_view_live = wx.CheckBox(localization_checkbox_panel, -1, 'Open (Live) View')
        self.checkbox_view_live.SetValue(False)
        if not self.checkbox_propagate.GetValue():
            self.checkbox_view_live.Disable()
        self.checkbox_view_live.Bind(wx.EVT_CHECKBOX, self.OnToggleLiveView)
        v_sizer.Add(self.checkbox_view_live)

        localization_checkbox_panel.SetSizer(v_sizer)
        self.AddNewElement(localization_checkbox_panel)

        clp = collapsingPane(self, caption='settings ...')
        clp.AddNewElement(AnalysisSettingsUI.AnalysisSettingsPanel(clp, localization_settings,
                                                                   localization_settings.onMetadataChanged))
        clp.AddNewElement(AnalysisSettingsUI.AnalysisDetailsPanel(clp, localization_settings,
                                                                  localization_settings.onMetadataChanged))
        self.AddNewElement(clp)

        self._localization_settings = localization_settings
        self._localization_settings.onMetadataChanged.connect(self.update_localization_rule)

        rule_panel = ChainedRulePanel(self, rule_chain, recipe_manager, spool_controller)
        self.AddNewElement(rule_panel)
        self._rule_list_ctrl = rule_panel._rule_list

    def OnTogglePropagate(self, wx_event=None):
        # for now, assume max of one localization rule per chain, and assume it's controlled by this panel
        if self.checkbox_propagate.GetValue():
            loc_rule_indices = self._rule_list_ctrl.localization_rule_indices
            if len(loc_rule_indices) < 1:
                self._rule_list_ctrl.add_rule(LocalizationRule(self._localization_settings.analysisMDH))
            self.checkbox_view_live.Enable()
        else:
            self._rule_list_ctrl.clear_localization_rules()
            self.checkbox_view_live.Disable()
            self.checkbox_view_live.SetValue(False)
    
    def OnToggleLiveView(self, wx_event=None):
        if self.checkbox_view_live.GetValue() and 0 in self._rule_list_ctrl.localization_rule_indices:
            self._rule_list_ctrl._rule_chain.posted.connect(self._open_live_view)
        else:
            self._rule_list_ctrl._rule_chain.posted.disconnect(self._open_live_view)

    def update_localization_rule(self):
        # NB - panel will only modify first localization rule in the chain
        if self.checkbox_propagate.GetValue():
            rule = LocalizationRule(self._localization_settings.analysisMDH)
            self._rule_list_ctrl.replace_rule(rule, self._rule_list_ctrl.localization_rule_indices[0])
    
    def _open_live_view(self, **kwargs):
        """
        Open PYMEVisualize on a freshly spooled series which is being localized

        Parameters
        ----------
        kwargs: dict
            present here to allow us to call this method through a dispatch.Signal.send

        """
        import subprocess
        # get the URL
        uri = self._rule_list_ctrl._rule_chain[self._rule_list_ctrl.localization_rule_indices[0]].outputs[0]['input'] + '/live'
        print(uri)
        subprocess.Popen('visgui %s' % uri, shell=True)



    @staticmethod
    def plug(main_frame, scope):
        from PYME.recipes.recipeGui import RecipeView, RecipeManager
        from PYME.Acquire.ui.AnalysisSettingsUI import AnalysisSettings
        from PYME.cluster.rules import RuleChain

        scope._recipe_manager = RecipeManager()
        main_frame.recipe_view = RecipeView(main_frame, scope._recipe_manager)
        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Chained Recipe')

        scope._rule_chain = RuleChain(scope.spoolController._analysis_launchers)

        scope._localization_settings = AnalysisSettings()

        chained_analysis = SMLMChainedAnalysisPanel(main_frame, scope._rule_chain, scope._recipe_manager,
                                                    scope._localization_settings, scope.spoolController)
        main_frame.anPanels.append((chained_analysis, 'Chained Analysis', True))
