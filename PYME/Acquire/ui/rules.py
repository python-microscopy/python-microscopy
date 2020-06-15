
import wx

class RuleChainListCtrl(wx.ListCtrl):
    def __init__(self, rule_chain, wx_parent):

        wx.ListCtrl.__init__(self, wx_parent, style=wx.LC_REPORT | wx.BORDER_SUNKEN | wx.LC_VIRTUAL | wx.LC_VRULES)
        self._rule_chain = rule_chain

        self.InsertColumn(0, 'Type', width=50)
        self.InsertColumn(1, 'ID', width=100)

    @property
    def first_is_localization(self):
        try:
            return self._rule_chain[0].template['type'] == 'localization'
        except IndexError:
            return False

    def add_rule(self, rule):
        """

        Parameters
        ----------
        rule: PYME.cluster.rules.Rule

        Returns
        -------

        """
        self._rule_chain.append(rule)
        self.update_list()

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

        if col == 0:
            return self._rule_chain[item].template['type']
        if col == 1:
            return str(item)
        else:
            return ''

    def update_list(self, sender=None, **kwargs):
        self.SetItemCount(len(self._rule_chain))
        self.Update()
        self.Refresh()

    def delete_rules(self):
        selected_indices = self.get_selected_items()

        for ind in selected_indices:
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

    def prepend_localization_rule(self, rule):
        if self.first_is_localization:  # simply replace the rule
            self._rule_chain[0] = rule
        else:
            self.add_rule(rule)

    def clear_localization_rule(self):
        if self.first_is_localization:
            self._rule_chain.pop(0)
            self.update_list()



class ChainedRulePanel(wx.Panel):
    _RULE_LAUNCH_MODES = ['off', 'series start', 'series stop']
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

        # self._localization_settings = localization_settings
        # if self._localization_settings is not None:
        #     self._localization_settings.onMetadataChanged.connect(self.update_localization_rule)

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
        if len(self._recipe_manager.activeRecipe) > 0:
            rule = RecipeRule(self._recipe_manager.activeRecipe.toYAML())
            self._rule_list.add_rule(rule)

    def OnRemove(self, wx_event=None):
        self._rule_list.delete_rules()

    # def update_localization_rule(self, sender=None, **kwargs):
    #     from PYME.cluster.rules import LocalizationRule
    #     if self._localization_settings.propagateToAcquisisitonMetadata:
    #         self._rule_list.prepend_localization_rule(LocalizationRule(self._localization_settings.analysisMDH))
    #     else:
    #         self._rule_list.clear_localization_rule()

    def OnToggleAuto(self, wx_event=None):
        mode = self.choice_launch.GetSelection()

        # todo - do we need to try/except these?
        self._spool_controller.onSpoolStart.disconnect(self.post_rules)
        self._spool_controller.onSpoolStop.disconnect(self.post_rules)

        if mode == 1:  # series start
            self._spool_controller.onSpoolStart.connect(self.post_rules)
        elif mode == 2:  # series stop
            self._spool_controller.onSpoolStop.connect(self.post_rules)

    def post_rules(self):
        # pipe the input series name into the rule list
        series_uri = self._spool_controller._get_queue_name
        self._rule_chain.set_chain_input({'input': series_uri})
        self._rule_chain.post_all()
