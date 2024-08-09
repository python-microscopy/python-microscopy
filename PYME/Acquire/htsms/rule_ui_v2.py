
import wx
# from  PYME.ui import manualFoldPanel
from PYME.cluster.rules import LocalisationRuleFactory as LocalizationRuleFactory
from PYME.cluster.rules import RecipeRuleFactory
from collections import OrderedDict
#import queue
import os
import posixpath
import logging
from PYME.contrib import dispatch, wxPlotPanel
from PYME.recipes.traits import HasTraits, Enum, Float, CStr
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import threading

logger = logging.getLogger(__name__)

POST_CHOICES = ['off', 'spool start', 'spool stop']

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
    post_on = Enum(POST_CHOICES)
    protocol = CStr('')
    
    def __init__(self, rule_factories=None, *args, **kwargs):
        if rule_factories is None:
            rule_factories = list()
        self.rule_factories = rule_factories
        HasTraits.__init__(self, *args, **kwargs)


class RuleDict(OrderedDict):
    """
    Container for associating sets of analysis rules with specific acquisition
    protocols

    Notes
    -----
    use ordered dict for reproducibility with listctrl displays
    """
    def __init__(self):
        """
        Parameters
        ----------
        posting_thread_queue_size : int, optional
            sets the size of a queue to hold rule posting threads to ensure they
            have time to execute, by default 5. .. seealso:: modules :py:mod:`PYME.cluster.rules`
        """
        import queue

        OrderedDict.__init__(self)
        self.active = True
        #self._spool_controller = spool_controller
        #self.posting_thread_queue = queue.Queue(posting_thread_queue_size)
        self._updated = dispatch.Signal()
        self._updated.connect(self.update)
        
        self['default'] = RuleChain()

        #self._spool_controller.onSpoolStart.connect(self.on_spool_start)
        #self._spool_controller.on_stop.connect(self.on_spool_stop)
    
    # def on_spool_start(self, **kwargs):
    #     self.on_spool_event('spool start')
    
    # def on_spool_stop(self, **kwargs):
    #     self.on_spool_event('spool stop')
    
    def on_spool_event(self, event):
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
        spooler = self._spool_controller.spooler
        try:
            spooler.getURL
        except AttributeError:
            logger.exception('Rule-based analysis chaining currently requires spooling to cluster, not to file')
            raise 
        prot_filename = spooler.protocol.filename
        prot_filename = '' if prot_filename is None else prot_filename
        protocol_name = os.path.splitext(os.path.split(prot_filename)[-1])[0]
        logger.info('protocol name : %s' % protocol_name)

        try:
            rule_factory_chain = self[protocol_name]
        except KeyError:
            rule_factory_chain = self['default']
        
        if rule_factory_chain.post_on != event:
            # not the right trigger for this protocol
            return
        
        if len(rule_factory_chain.rule_factories) == 0:
            logger.info('no rules in chain')
            return
        
        # set the context based on the input series
        series_uri = spooler.getURL()
        spool_dir, series_stub = posixpath.split(series_uri)
        series_stub = posixpath.splitext(series_stub)[0]
        context = {
            'spool_dir': spool_dir,  # do we need this? or typo in rule docs
            'series_stub': series_stub,  # do we need this? or typo in rule docs
            'seriesName': series_uri,  # Localization
            'inputs': {'input': [series_uri]},  # Recipe
            'output_dir': posixpath.join(spool_dir, 'analysis'), # Recipe
            'spooler': spooler}  # SpoolLocalLocalization

        # rule chain is already linked, add context and push
        rule = rule_factory_chain.rule_factories[0].get_rule(context=context)
        t = threading.Thread(target=rule.push)
        t.start()
        if self.posting_thread_queue.full():
            self.posting_thread_queue.get_nowait().join()
        self.posting_thread_queue.put_nowait(t)
        
    
    def update(self, *args, **kwargs):
        for p in self.keys():
            factories = self[p].rule_factories
            for ind in range(len(factories) - 1):
                factories[ind].chain(factories[ind + 1])



class RulePlotPanel(wxPlotPanel.PlotPanel):
    def __init__(self, parent, protocol_rules, **kwargs):
        self.protocol_rules = protocol_rules
        self.parent = parent
        wxPlotPanel.PlotPanel.__init__(self, parent, **kwargs)
        self.figure.canvas.mpl_connect('pick_event', self.parent.OnPick)

    def draw(self):
        if not self.IsShownOnScreen():
            return
            
        if not hasattr( self, 'ax' ):
            self.ax = self.figure.add_axes([0, 0, 1, 1])

        self.ax.cla()

        rule_factories = self.parent.rule_chain.rule_factories
        if len(rule_factories) < 1:
            self.canvas.draw()
            return
        width = 1  # size of tile to draw
        height = 0.5
        nodes_x = np.arange(0, len(rule_factories) * 1.5 * width, 1.5 * width)
        nodes_y = np.ones_like(nodes_x)

        
        axis_width = self.ax.get_window_extent().width
        n_cols = max([1] + nodes_x.tolist())
        pix_per_col = axis_width / n_cols
        
        font_size = max(6, min(10, 10 * pix_per_col / 100))
        
        TW = textwrap.TextWrapper(width=max(int(1.8 * pix_per_col / font_size), 10),
                                  subsequent_indent='  ')
        TW2 = textwrap.TextWrapper(width=max(int(1.3 * pix_per_col / font_size), 10),
                                   subsequent_indent='  ')
    
        cols = {}

        # plot connecting lines
        for ind in range(1, len(rule_factories)):
            self.ax.plot([nodes_x[ind - 1] + width, nodes_x[ind]],
                         [nodes_y[ind - 1] + 0.5 * height, 
                          nodes_y[ind] + 0.5 * height], lw=2)
                
        #plot the boxes and the labels
        for ind in range(len(rule_factories)):
            # draw a box
            s = rule_factories[ind]._type
            fc = [.8,.8, 1]
            
            rect = plt.Rectangle([nodes_x[ind], nodes_y[ind]], width, height,
                                 ec='k', lw=2, fc=fc, picker=True)
            rect._data = rule_factories[ind]
            self.ax.add_patch(rect)
            
            s = TW2.wrap(s)
            if len(s) == 1:
                self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .18 , s[0], size=font_size, weight='bold')
            else:
                self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .18 - .05*(len(s) - 1) , '\n'.join(s), size=font_size, weight='bold')
        
        self.ax.set_ylim(0, 2)
        self.ax.set_xlim(-0.5 * width, nodes_x[-1] + 1.5 * width)
        
        self.ax.axis('off')
        self.ax.grid()
        
        self.canvas.draw()

class ChainedAnalysisPage(wx.Panel):
    def __init__(self, parent, rules, recipe_manager, 
                 localization_panel=None, default_pairings=None, localization_settings=None):
        """

        Parameters
        ----------
        parent : wx something
        protocol_rules : dict
            [description]
        recipe_manager : PYME.recipes.recipeGui.RecipeManager
            [description]
        default_pairings : dict
            protocol keys with lists of RuleFactorys as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)
        self._loaded_rules = rules
        self._selected_rule = list(rules.keys())[0]
        self._recipe_manager = recipe_manager
        self._localization_panel = localization_panel

        self._editor_panels = []

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        # associated protocol choicebox (default / others on top, then rest)
        hsizer.Add(wx.StaticText(self, -1, 'Rule Chain:'), 0, 
                   wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        self.c_rule_selection = wx.Choice(self, -1, choices=list(self._loaded_rules.keys()))
        self.c_rule_selection.SetSelection(0)
        self.c_rule_selection.Bind(wx.EVT_CHOICE, self.OnRuleChoice)
        hsizer.Add(self.c_rule_selection, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        hsizer.AddStretchSpacer()

        self._b_add_rule_chain = wx.Button(self, -1, 'Add')
        hsizer.Add(self._b_add_rule_chain, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self._b_add_rule_chain.Bind(wx.EVT_BUTTON, self.OnAddRuleChain)

        self._b_load_rule_chain = wx.Button(self, -1, 'Load')
        hsizer.Add(self._b_load_rule_chain, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self._b_load_rule_chain.Bind(wx.EVT_BUTTON, self.OnLoadRuleChain)

        self._b_save_rule_chain = wx.Button(self, -1, 'Save')
        hsizer.Add(self._b_save_rule_chain, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self._b_save_rule_chain.Bind(wx.EVT_BUTTON, self.OnSaveRuleChain)

        vsizer.Add(hsizer, 0, wx.TOP|wx.EXPAND, 5)

        # rule plot
        self.rule_plot = RulePlotPanel(self, self._loaded_rules, 
                                       size=(-1, 100))
        vsizer.Add(self.rule_plot, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.b_clear = wx.Button(self, -1, 'Clear Chain')
        hsizer.Add(self.b_clear, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_clear.Bind(wx.EVT_BUTTON, self.OnClear)
        
        self.b_add_localization = wx.Button(self, -1, 'Add Localization Task')
        hsizer.Add(self.b_add_localization, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_localization.Bind(wx.EVT_BUTTON, self.OnAddLocalization)

        self.b_add_recipe = wx.Button(self, -1, 'Add Recipe Task')
        hsizer.Add(self.b_add_recipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_recipe.Bind(wx.EVT_BUTTON, self.OnAddRecipe)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        v_sizer.Add(vsizer, 0, wx.EXPAND, 0)

        if self._localization_panel is None and not localization_settings is None:
            self._lpan = wx.Panel(self, -1)
            self._localization_panel = LocalizationSettingsPanel(self._lpan,localization_settings, localization_settings.onMetadataChanged)
            self._localization_panel.chained_analysis_page = self
            sbsizer = wx.StaticBoxSizer(wx.StaticBox(self._lpan, -1, 'Localisation settings'), wx.HORIZONTAL)
            sbsizer.Add(self._localization_panel, 1, wx.EXPAND, 0)
            self._lpan.SetSizerAndFit(sbsizer)
            v_sizer.Add(self._lpan, 1, wx.EXPAND|wx.ALL, 5)
            self._lpan.Hide()
            self._editor_panels.append(self._lpan)

        self._rpan = wx.Panel(self, -1)
        self._recipe_view = RuleRecipeView(self._rpan, self._recipe_manager)
        sbsizer = wx.StaticBoxSizer(wx.StaticBox(self._rpan, -1, 'Recipe Editor'), wx.HORIZONTAL)
        sbsizer.Add(self._recipe_view, 1, wx.EXPAND, 0)
        self._rpan.SetSizerAndFit(sbsizer)
        v_sizer.Add(self._rpan, 1, wx.EXPAND|wx.ALL, 5)
        self._rpan.Hide()
        self._editor_panels.append(self._rpan)


        self.SetSizerAndFit(v_sizer)
        self._loaded_rules._updated.connect(self.update)
    
    def OnClear(self, wx_event=None):
        self._loaded_rules[self._selected_rule] = RuleChain()
        self._loaded_rules._updated.send(self)

    def OnAddRuleChain(self, wx_event=None):
        # display a text enty dialog to get the name of the new rule chain
        # then add it to the list of rule chains
        from PYME import warnings

        dlg = wx.TextEntryDialog(self, 'Enter name for new rule chain', 'New Rule Chain', '')
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue()

            if name in self._loaded_rules.keys() or name == '':
                dlg.Destroy()
                warnings.warn('Name already in use or empty')
                return
            else:
                self._loaded_rules[name] = RuleChain()
                self._loaded_rules._updated.send(self) 

                self.select_rule_chain(name)

    def OnLoadRuleChain(self, wx_event=None):
        # display a dialog to select a rule file to load
        # then load it into the list of rule chains
        from PYME import warnings

        # TODO - specify the directory to start in
        # TODO - will this be YAML?
        dlg = wx.FileDialog(self, 'Choose a rule chain file', '', '', 'YAML files (*.yaml)|*.yaml', wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            name = (os.path.split(path)[-1])
            if name in self._loaded_rules.keys():
                warnings.warn('Name already in use')
                return
            
            with open(path, 'r') as f:
                self._loaded_rules[name] = RuleChain.from_yaml(f.read())

            self._loaded_rules._updated.send(self)
            self.select_rule_chain(name)

    def OnSaveRuleChain(self, wx_event=None):
        # display a dialog to select a file to save the rule chain to
        # then save it
        from PYME import warnings

        # TODO - specify the directory to start in
        # TODO - will this be YAML?
        dlg = wx.FileDialog(self, 'Choose a rule chain file', '', '', 'YAML files (*.yaml)|*.yaml', wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            
            with open(path, 'w') as f:
                f.write(self._loaded_rules[self._selected_rule].to_yaml())

    

    def OnRuleChoice(self, wx_event=None):
        self.select_rule_chain(self.c_rule_selection.GetStringSelection())
    
    def select_rule_chain(self, rule='default'):
        if rule not in self._loaded_rules.keys():
            self._loaded_rules[rule] = RuleChain()
            self._loaded_rules._updated.send(self)
        
        self._selected_rule = rule

        for e in self._editor_panels:
            e.Hide()

        # force a redraw, even though we might just have done so if we added
        self.update()
        
        
    def update(self, *args, **kwargs):
        self.c_rule_selection.SetItems(list(self._loaded_rules.keys()))
        self.c_rule_selection.SetStringSelection(self._selected_rule)
        self.rule_plot.draw()
        
    
    def edit_localisation_rule(self, rule):
        for e in self._editor_panels:
            e.Hide()

        self._localization_panel.localization_settings.analysisMDH = rule._rule_kwargs['analysisMetadata']
        self._localization_panel.localization_settings.onMetadataChanged.send_robust(self._localization_panel.localization_settings)
        self._lpan.Show()
        self.Layout()

    def edit_recipe_rule(self, rule):
        for e in self._editor_panels:
            e.Hide()

        self._recipe_view.edit_recipe_rule(rule)
        self._rpan.Show()
        self.Layout()

    def OnPick(self, event):
        k = event.artist._data
        #logger.info('picked %s' % k)

        if k.rule_type == 'localization':
            self.edit_localisation_rule(k)
        elif k.rule_type == 'recipe':
            self.edit_recipe_rule(k)
    
    @property
    def rule_chain(self):
        return self._loaded_rules[self._selected_rule]

    def add_tile(self, rule_tile):
        self.rule_chain.rule_factories.append(rule_tile)
        self._loaded_rules._updated.send(self)

        # for e in self._editor_panels:
        #     e.Hide()

        self.Layout()

    
    
    def OnAddLocalization(self, wx_event=None):
        from PYME.IO.MetaDataHandler import DictMDHandler
        from PYME.cluster.rules import SpoolLocalLocalizationRuleFactory
        # copy the metadata handler so we don't accidentally over-ride the original
        mdh = DictMDHandler(self._localization_panel.localization_settings.analysisMDH)
        loc_rule = get_rule_tile(SpoolLocalLocalizationRuleFactory)(analysisMetadata=mdh)
        self.add_tile(loc_rule)

        self.edit_localisation_rule(loc_rule)

    def OnAddRecipe(self, wx_event=None):
        rec = get_rule_tile(RecipeRuleFactory)(recipe='')
        self.add_tile(rec)
        
        self.edit_recipe_rule(rec)
        

from PYME.recipes.recipeGui import RecipeView, RecipeManager, RecipePlotPanel
class RuleRecipeView(RecipeView):
    _rule = None

    def edit_recipe_rule(self, rule):
        self._rule = rule
        self.recipes.activeRecipe.update_from_yaml(rule._rule_kwargs['recipe'])

    def update(self, *args, **kwargs):
        if self._rule is not None:
            self._rule._rule_kwargs['recipe'] = self.recipes.activeRecipe.toYAML()

        return super().update(*args, **kwargs)


class RuleRecipeManager(RecipeManager):
    def __init__(self, chained_analysis_page=None):
        RecipeManager.__init__(self)
        self.chained_analysis_page = chained_analysis_page

    def OnAddRecipeRule(self, wx_event=None):
        from PYME.cluster.rules import RecipeRuleFactory
        #from PYME.Acquire.htsms.rule_ui import get_rule_tile
        if self.chained_analysis_page is None:
            logger.error('chained_analysis_page attribute unset')

        rec = get_rule_tile(RecipeRuleFactory)(recipe=self.activeRecipe.toYAML())
        self.chained_analysis_page.add_tile(rec)

class ChainedAnalysisPanel(wx.Panel):
    def __init__(self, parent, protocol_rules, chained_analysis_page,
                 default_pairings=None):
        """

        Parameters
        ----------
        parent : PYME.ui.AUIFrame.AUIFrame
            should be the 'main frame'
        protocol_rules : dict
            [description]
        recipe_manager : PYME.recipes.recipeGui.RecipeManager
            [description]
        default_pairings : dict
            protocol keys with RuleChains as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)
        self.parent = parent
        self._protocol_rules = protocol_rules
        self._page = chained_analysis_page

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)

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
        for protocol_name, rule_chain in pairings.items():
            # add them to the protocol rules dict
            self._protocol_rules[protocol_name] = rule_chain
        
        self._protocol_rules._updated.send(self)

    def OnRemoveProtocolRule(self, wx_event=None):
        # make sure that we reset the chained analysis page just in case
        # we deleted a rule which was active
        self._page.c_protocol.SetStringSelection('default')
        self._page.OnProtocolChoice()
        self._protocol_rules_list.delete_rule_chains()
    
    def OnEditRuleChain(self, wx_event=None):
        ind = self._protocol_rules_list.get_selected_items()[0]
        protocol = self._protocol_rules_list.GetItemText(ind, col=0)
        self._page.select_rule_chain(protocol)
        self.parent._select_page_by_name('Chained Analysis')

    def OnToggleActive(self, wx_event):
        self._protocol_rules.active = self.checkbox_active.GetValue()

    @staticmethod
    def plug(main_frame, scope, default_pairings=None):
        """
        Adds a ChainedAnalysisPanel to a microscope gui during start-up
        
        Parameters
        ----------
        main_frame : PYME.Acquire.acquiremainframe.PYMEMainFrame
            microscope gui application
        scope : PYME.Acquire.microscope.Microscope
            the microscope itself
        default_pairings : dict
            [optional] protocol keys with lists of RuleFactorys as values to
            prepopulate panel on start up. By default, None
        
        """

        scope.protocol_rules = ProtocolRules(scope.spoolController)
        scope._recipe_manager = RuleRecipeManager()

        main_frame.chained_analysis_page = ChainedAnalysisPage(main_frame, 
                                                               scope.protocol_rules,
                                                               scope._recipe_manager,
                                                               default_pairings)
        
        scope._recipe_manager.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.recipe_view = RuleRecipeView(main_frame, scope._recipe_manager)

        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Recipe')
        main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                           caption='Chained Analysis')

        # add this panel
        chained_analysis = ChainedAnalysisPanel(main_frame, 
                                                scope.protocol_rules,
                                                main_frame.chained_analysis_page,
                                                default_pairings)
        main_frame.anPanels.append((chained_analysis, 'Automatic Analysis', 
                                    True))


from PYME.Acquire.ui.AnalysisSettingsUI import AnalysisSettingsPanel, AnalysisDetailsPanel, manualFoldPanel
class LocalizationSettingsPanel(wx.Panel):
    def __init__(self, wx_parent, localization_settings,
                 chained_analysis_page=None):
        #from PYME.ui.autoFoldPanel import collapsingPane
        #manualFoldPanel.foldingPane.__init__(self, wx_parent, caption='Localization Analysis')
        wx.Panel.__init__(self, wx_parent)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.localization_settings = localization_settings
        self.chained_analysis_page = chained_analysis_page
        
        #clp = collapsingPane(self, caption='settings ...')

        asp = AnalysisSettingsPanel(self,self.localization_settings,self.localization_settings.onMetadataChanged, show_save_mdh=False)
        vsizer.Add(asp, 0, wx.EXPAND|wx.ALL, 5)
        adp = AnalysisDetailsPanel(self, self.localization_settings,self.localization_settings.onMetadataChanged)
        vsizer.Add(adp, 1, wx.EXPAND|wx.ALL, 5)
       
        self.SetSizerAndFit(vsizer)

        

class SMLMChainedAnalysisPanel(ChainedAnalysisPanel):
    def __init__(self, wx_parent, protocol_rules, chained_analysis_page,
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
                                      chained_analysis_page, default_pairings)

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
        scope : PYME.Acquire.microscope.Microscope
            the microscope itself
        default_pairings : dict
            [optional] protocol keys with RuleChains as values to
            prepopulate panel on start up. By default, None
        """
        from PYME.Acquire.ui.AnalysisSettingsUI import AnalysisSettings

        scope.analysis_rules = RuleDict()
        scope._recipe_manager = RecipeManager()
        scope._localization_settings = AnalysisSettings() #TODO - we should not need this to be global.

        # localization_settings_pan = LocalizationSettingsPanel(main_frame,
        #                                                              scope._localization_settings,
        #                                                              scope._localization_settings.onMetadataChanged)

        main_frame.chained_analysis_page = ChainedAnalysisPage(main_frame, 
                                                                   scope.analysis_rules,
                                                                   scope._recipe_manager,
                                                                   None,
                                                                   default_pairings, localization_settings=scope._localization_settings)
        
        #scope._recipe_manager.chained_analysis_page = main_frame.chained_analysis_page
        #main_frame.recipe_view = RuleRecipeView(main_frame, scope._recipe_manager)
        #main_frame.localization_settings.chained_analysis_page = main_frame.chained_analysis_page
        #main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Recipe')
        #main_frame.AddPage(page=main_frame.localization_settings, select=False, caption='Localization')
        main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                           caption='Chained Analysis')
        
        # add this panel
        # chained_analysis = SMLMChainedAnalysisPanel(main_frame, 
        #                                             scope.protocol_rules,
        #                                             main_frame.chained_analysis_page,
        #                                             default_pairings)
        # main_frame.anPanels.append((chained_analysis, 'Automatic Analysis', 
        #                             False))
