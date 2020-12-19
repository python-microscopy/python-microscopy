
import wx
# from  PYME.ui import manualFoldPanel
from PYME.cluster.rules import LocalisationRuleFactory as LocalizationRuleFactory
from PYME.cluster.rules import RecipeRuleFactory
from collections import OrderedDict
import queue
import os
import posixpath
import logging
from PYME.contrib import dispatch
from PYME.recipes.traits import HasTraits, Enum, Float, CStr

logger = logging.getLogger(__name__)


class RuleTile(HasTraits):
    task_timeout = Float(60 * 10)
    rule_timeout = Float(60 * 10)

    def get_params(self):
        editable = self.class_editable_traits()
        return editable 

    @property
    def default_view(self):
        if wx.GetApp() is None:
            return None
        from traitsui.api import View, Item

        return View([Item(tn) for tn in self.get_params()], buttons=['OK'])

    def default_traits_view(self):
        """ This is the traits stock method to specify the default view"""
        return self.default_view

def get_rule_tile(rule_factory_class):
    class _RuleTile(RuleTile, rule_factory_class):
        def __init__(self, **kwargs):
            RuleTile.__init__(self)
            rule_factory_class.__init__(self, **kwargs)
    return _RuleTile


class RuleChain(HasTraits):
    post_on = Enum(['off', 'spool start', 'spool stop'])
    protocol = CStr('')
    
    def __init__(self, rule_factories=None, *args, **kwargs):
        if rule_factories is None:
            rule_factories = list()
        self.rule_factories = rule_factories
        HasTraits.__init__(self, *args, **kwargs)


class ProtocolRules(OrderedDict):
    """
    Container for associating sets of analysis rules with specific acquisition
    protocols
    """
    def __init__(self, posting_thread_queue_size=5):
        """[summary]
        Parameters
        ----------
        posting_thread_queue_size : int, optional
            sets the size of a queue to hold rule posting threads to ensure they
            have time to execute, by default 5. .. seealso:: modules :py:mod:`PYME.cluster.rules`
        """
        import queue

        OrderedDict.__init__(self)

        self.posting_thread_queue = queue.Queue(posting_thread_queue_size)
        
        self['default'] = RuleChain()

        self._updated = dispatch.Signal()


class ProtocolRuleFactoryListCtrl(wx.ListCtrl):
    def __init__(self, protocol_rules, wx_parent):
        """
        Parameters
        ----------
        protocol_rules: dict
            acquisition protocols (keys) and their associated rule factory 
            chains
        wx_parent
        """
        wx.ListCtrl.__init__(self, wx_parent, style=wx.LC_REPORT | wx.BORDER_SUNKEN | wx.LC_VIRTUAL | wx.LC_VRULES)

        self._protocol_rules = protocol_rules

        self.InsertColumn(0, 'Protocol', width=75)
        self.InsertColumn(1, '# Rules', width=50)
        self.InsertColumn(2, 'Post', width=75)

        self.update_list()
        self._protocol_rules._updated.connect(self.update_list)

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
                return list(self._protocol_rules.keys())[item]
            if col == 1:
                chains = list(self._protocol_rules.rule_factories.values())
                return str(len(chains[item]))
            if col == 2:
                chains = list(self._protocol_rules.rule_factories.values())
                return chains[item].post_on
        except:
            return ''
        else:
            return ''

    def update_list(self, sender=None, **kwargs):
        self.SetItemCount(len(self._protocol_rules.keys()))
        self.Update()
        self.Refresh()

    def delete_rule_chains(self, indices=None):
        selected_indices = self.get_selected_items() if indices is None else indices

        for ind in reversed(sorted(selected_indices)):  # delete in reverse order so we can pop without changing indices
            if self.GetItemText(ind, col=0) == 'default':
                logger.error('Cannot delete the default rule chain')
                continue  # try to keep people from deleting the default chain
            self._protocol_rules.popitem(ind)
            self.DeleteItem(ind)
        
        self._protocol_rules._updated.send(self)

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

class ChainedAnalysisPage(wx.Panel):
    def __init__(self, parent, protocol_rules, recipe_manager, 
                 spool_controller, default_pairings=None):
        """

        Parameters
        ----------
        parent : wx something
        protocol_rules : dict
            [description]
        recipe_manager : PYME.recipes.recipeGui.RecipeManager
            [description]
        spool_controller : PYME.Acquire.SpoolController.SpoolController
            microscope's spool controller instance, so we can launch on spool
            start/stop
        default_pairings : dict
            protocol keys with lists of RuleFactorys as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)
        self._protocol_rules = protocol_rules
        self._selected_protocol = list(protocol_rules.keys())[0]
        self._recipe_manager = recipe_manager
        self._spool_controller = spool_controller
    
    @property
    def rule_chain(self):
        return self._protocol_rules[self._selected_protocol]

    def add_tile(self, rule_tile):
        self.rule_chain.rule_factories.append(rule_tile)
        self._protocol_rules._updated.send(self)
    
class SMLMChainedAnalysisPage(ChainedAnalysisPage):
    def __init__(self, parent, protocol_rules, recipe_manager, 
                 localization_settings, spool_controller, 
                 default_pairings=None):
        """

        Parameters
        ----------
        parent : wx something
        protocol_rules : dict
            [description]
        recipe_manager : PYME.recipes.recipeGui.RecipeManager
            [description]
        spool_controller : PYME.Acquire.SpoolController.SpoolController
            microscope's spool controller instance, so we can launch on spool
            start/stop
        default_pairings : dict
            protocol keys with lists of RuleFactorys as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)
        self._protocol_rules = protocol_rules
        self._selected_protocol = list(protocol_rules.keys())[0]
        self._recipe_manager = recipe_manager
        self._spool_controller = spool_controller
        

class ChainedAnalysisPanel(wx.Panel):
    def __init__(self, parent, protocol_rules, spool_controller, 
                 default_pairings=None):
        """

        Parameters
        ----------
        parent : wx something
        protocol_rules : dict
            [description]
        recipe_manager : PYME.recipes.recipeGui.RecipeManager
            [description]
        spool_controller : PYME.Acquire.SpoolController.SpoolController
            microscope's spool controller instance, so we can launch on spool
            start/stop
        default_pairings : dict
            protocol keys with lists of RuleFactorys as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)

        self._protocol_rules = protocol_rules
        self._spool_controller = spool_controller

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.active = True
        self.checkbox_active = wx.CheckBox(self, -1, 'active')
        self.checkbox_active.SetValue(True)
        self.checkbox_active.Bind(wx.EVT_CHECKBOX, self.OnToggleActive)
        h_sizer.Add(self.checkbox_active, 0, wx.ALL, 2)
        v_sizer.Add(h_sizer, 0, wx.EXPAND|wx.TOP, 0)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._protocol_rules_list = ProtocolRuleFactoryListCtrl(self._protocol_rules, self)
        h_sizer.Add(self._protocol_rules_list)
        v_sizer.Add(h_sizer, 0, wx.EXPAND|wx.TOP, 0)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.button_del_chain = wx.Button(self, -1, 'Delete pair')
        self.button_del_chain.Bind(wx.EVT_BUTTON, self.OnRemoveProtocolRule)
        h_sizer.Add(self.button_del_chain, 0, wx.ALL, 2)

        self.button_edit_chain = wx.Button(self, -1, 'Edit in tab')
        self.button_edit_chain.Bind(wx.EVT_BUTTON, self.OnEditRuleChain)
        h_sizer.Add(self.button_edit_chain, 0, wx.ALL, 2)
        v_sizer.Add(h_sizer)

        self.SetSizerAndFit(v_sizer)

        if default_pairings is not None:
            self._set_up_defaults(default_pairings)
    
    def _set_up_defaults(self, pairings):
        for protocol_name, rule_factory_list in pairings.items():
            # make sure we've chained the rules
            n_rules = len(rule_factory_list)
            for s_ind, f_ind in enumerate(range(1, n_rules)):
                rule_factory_list[s_ind].chain(rule_factory_list[f_ind])

            # add them to the protocol rules dict
            self._protocol_rules[protocol_name] = rule_factory_list
        
        # update the GUI
        # self._protocol_rules_list.update_list()
        self._protocol_rules._updated.send(self)

    def OnRemoveProtocolRule(self, wx_event=None):
        self._protocol_rules_list.delete_rule_chains()
    
    def OnEditRuleChain(self, wx_event=None):
        raise NotImplementedError()

    def OnToggleActive(self, wx_event):
        self.active = self.checkbox_active.GetValue()

    def post_rules(self, **kwargs):
        """
        pipe input series name into rule chain and post them all
        Parameters
        ----------
        kwargs: dict
            present here to allow us to call this method through a dispatch.Signal.send
        """
        if not self.active:
            logger.info('inactive, check "active" to turn on auto analysis')
            return
        prot_filename = self._spool_controller.spooler.protocol.filename
        prot_filename = '' if prot_filename is None else prot_filename
        protocol_name = os.path.splitext(os.path.split(prot_filename)[-1])[0]
        logger.info('protocol name : %s' % protocol_name)

        try:
            rule_factory_chain = self._protocol_rules[protocol_name]
        except KeyError:
            rule_factory_chain = self._protocol_rules['default']
        
        if len(rule_factory_chain) == 0:
            logger.info('no rules in chain')
            return
        
        # set the context based on the input series
        series_uri = self._spool_controller.spooler.getURL()
        spool_dir, series_stub = posixpath.split(series_uri)
        series_stub = posixpath.splitext(series_stub)[0]
        context = {'spool_dir': spool_dir, 'series_stub': series_stub,
                   'seriesName': series_uri, 'inputs': {'input': [series_uri]},
                   'output_dir': posixpath.join(spool_dir, 'analysis')}

        # rule chain is already linked, add context and push
        rule_factory_chain.rule_factories[0].get_rule(context=context).push()

    @staticmethod
    def plug(main_frame, scope, default_pairings=None):
        """
        Adds a ChainedAnalysisPanel to a microscope gui during start-up
        Parameters
        ----------
        main_frame : PYME.Acquire.acquiremainframe.PYMEMainFrame
            microscope gui application
        scope : PYME.Acquire.microscope.microscope
            the microscope itself
        default_pairings : dict
            [optional] protocol keys with lists of RuleFactorys as values to
            prepopulate panel on start up. By default, None
        """
        from PYME.recipes.recipeGui import RuleRecipeView, RuleRecipeManager

        scope.protocol_rules = ProtocolRules()
        scope._recipe_manager = RuleRecipeManager()

        main_frame.chained_analysis_page = ChainedAnalysisPage(main_frame, 
                                                               scope.protocol_rules,
                                                               scope._recipe_manager,
                                                               scope.spoolController,
                                                               default_pairings)
        
        scope._recipe_manager.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.recipe_view = RuleRecipeView(main_frame, scope._recipe_manager)

        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Recipe')
        main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                           caption='Chained Analysis')

        # add this panel
        chained_analysis = ChainedAnalysisPanel(main_frame, scope.protocol_rules,
                                                scope.spoolController,
                                                default_pairings)
        main_frame.anPanels.append((chained_analysis, 'Automatic Analysis', 
                                    True))


class SMLMChainedAnalysisPanel(ChainedAnalysisPanel):
    def __init__(self, wx_parent, protocol_rules, spool_controller, 
                 default_pairings=None):
        """
        Parameters
        ----------
        wx_parent
        localization_settings: PYME.ui.AnalysisSettingsUI.AnalysisSettings
        rule_list_ctrl: RuleChainListCtrl
        default_pairings : dict
            [optional] protocol keys with lists of RuleFactorys as values to
            prepopulate panel on start up. By default, None
        """
        ChainedAnalysisPanel.__init__(self, wx_parent, protocol_rules, 
                                      spool_controller, 
                                      default_pairings=None)

    # def OnToggleLiveView(self, wx_event=None):
    #     if self.checkbox_view_live.GetValue() and 0 in self._rule_list_ctrl.localization_rule_indices:
    #         self._rule_list_ctrl._rule_chain.posted.connect(self._open_live_view)
    #     else:
    #         self._rule_list_ctrl._rule_chain.posted.disconnect(self._open_live_view)

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
        subprocess.Popen('visgui %s' % uri, shell=True)

    @staticmethod
    def plug(main_frame, scope, default_pairings=None):
        """
        Adds a SMLMChainedAnalysisPanel to a microscope gui during start-up
        Parameters
        ----------
        main_frame : PYME.Acquire.acquiremainframe.PYMEMainFrame
            microscope gui application
        scope : PYME.Acquire.microscope.microscope
            the microscope itself
        default_pairings : dict
            [optional] protocol keys with lists of RuleFactorys as values to
            prepopulate panel on start up. By default, None
        """
        from PYME.recipes.recipeGui import RuleRecipeView, RuleRecipeManager
        from PYME.Acquire.ui.AnalysisSettingsUI import AnalysisSettings, LocalizationSettingsPanel

        scope.protocol_rules = ProtocolRules()
        scope._recipe_manager = RuleRecipeManager()
        scope._localization_settings = AnalysisSettings()

        main_frame.chained_analysis_page = SMLMChainedAnalysisPage(main_frame, 
                                                                   scope.protocol_rules,
                                                                   scope._recipe_manager,
                                                                   scope._localization_settings,
                                                                   scope.spoolController,
                                                                   default_pairings)
        
        scope._recipe_manager.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.recipe_view = RuleRecipeView(main_frame, scope._recipe_manager)
        main_frame.localization_settings = LocalizationSettingsPanel(main_frame,
                                                                     scope._localization_settings,
                                                                     scope._localization_settings.onMetadataChanged)
        main_frame.localization_settings.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Recipe')
        main_frame.AddPage(page=main_frame.localization_settings, select=False, caption='Localization')
        main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                           caption='Chained Analysis')
        
        # add this panel
        chained_analysis = SMLMChainedAnalysisPanel(main_frame, 
                                                    scope.protocol_rules,
                                                    scope.spoolController,
                                                    default_pairings)
        main_frame.anPanels.append((chained_analysis, 'Automatic Analysis', 
                                    True))
