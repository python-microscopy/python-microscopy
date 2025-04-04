
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

from PYME.IO.MetaDataHandler import DictMDHandler

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

            self._rule_factory_class_name = rule_factory_class.__name__

        def serialise(self):
            return(self._rule_factory_class_name, self._rule_kwargs)
    
    return _RuleTile

def get_rule_factory_class(name):
    from PYME.cluster import rules
    return getattr(rules, name)


class RuleChain(HasTraits):
    def __init__(self, rule_factories=None, *args, **kwargs):
        if rule_factories is None:
            rule_factories = list()
        self.rule_factories = rule_factories
        HasTraits.__init__(self, *args, **kwargs)

    def to_yaml(self):
        import yaml
        from PYME.recipes.base import MyDumper
        return yaml.dump([rf.serialise() for rf in self.rule_factories], Dumper=MyDumper)
    
    @classmethod
    def from_yaml(cls, yaml_str):
        import yaml
        factories = yaml.safe_load(yaml_str)
        return cls([get_rule_tile(get_rule_factory_class(cls))(**kwargs) for cls, kwargs in factories])

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
        from PYME.cluster.rules import SpoolLocalLocalizationRuleFactory

        OrderedDict.__init__(self)
        self.active = True
        #self._spool_controller = spool_controller
        #self.posting_thread_queue = queue.Queue(posting_thread_queue_size)
        self._updated = dispatch.Signal()
        self._updated.connect(self.update)
        
        self['default'] = RuleChain()
        # TODO - make the default rule chain a 2D Gaussian localization rule
        #mdh = DictMDHandler(self._localization_panel.localization_settings.analysisMDH)
        #loc_rule = get_rule_tile(SpoolLocalLocalizationRuleFactory)(analysisMetadata=mdh)


    def load_from_config(self):
        """
        Load rule chains stored in the analysis_rules subfolder of the PYME config directory
        (typically ~/.PYME/analysis_rules)

        """
        from PYME import config

        chains = config.get_analysis_rulechains()

        for k, fn in chains.items():
            with open(fn, 'r') as f:
                self[os.path.splitext(k)[0]] = RuleChain.from_yaml(f.read())
        
    
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
        height = 1
        nodes_x = np.arange(0, len(rule_factories) * 1.5 * width, 1.5 * width)
        nodes_y = 0.5*np.ones_like(nodes_x)

        
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
            rule = rule_factories[ind]
            s = rule.rule_type
            fc = [.8,.8, 1]
            
            rect = plt.Rectangle([nodes_x[ind], nodes_y[ind]], width, height,
                                 ec='k', lw=2, fc=fc, picker=True)
            rect._data = rule
            self.ax.add_patch(rect)
            
            s = TW2.wrap(s)
            if len(s) == 1:
                self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .68 , s[0], size=font_size, weight='bold')
            else:
                self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .18 - .05*(len(s) - 1) , '\n'.join(s), size=font_size, weight='bold')

            if rule.rule_type == 'localization':
                try:
                    s = TW.wrap(str(rule._rule_kwargs['analysisMetadata']['Analysis.FitModule']))
                    self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .18 - .05*(len(s)) , '\n'.join(s), size=font_size, weight='normal')
                except KeyError:
                    pass
            elif rule.rule_type == 'recipe':
                try:
                    r = str(rule._rule_kwargs['recipe'])
                    s = TW.wrap(r.split('\n')[0] + ' ...')   
                    self.ax.text(nodes_x[ind] + .05, nodes_y[ind] + .18 - .05*(len(s)) , '\n'.join(s), size=font_size, weight='normal')
                except KeyError:
                    pass
        
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

        self._displayed_rule_keys = None

        self._editor_panels = []

        v_sizer = wx.BoxSizer(wx.VERTICAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        # hsizer = wx.BoxSizer(wx.HORIZONTAL)

        # # associated protocol choicebox (default / others on top, then rest)
        # hsizer.Add(wx.StaticText(self, -1, 'Rule Chain:'), 0, 
        #            wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
        # self.c_rule_selection = wx.Choice(self, -1, choices=list(self._loaded_rules.keys()))
        # self.c_rule_selection.SetSelection(0)
        # self.c_rule_selection.Bind(wx.EVT_CHOICE, self.OnRuleChoice)
        # hsizer.Add(self.c_rule_selection, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        # vsizer.Add(hsizer, 0, wx.EXPAND, 0)

        self.l_analysis_rules = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL)
        self.l_analysis_rules.InsertColumn(0, 'Rule Name', width=150)
        self.l_analysis_rules.InsertColumn(1, 'Summary', width=wx.LIST_AUTOSIZE)

        self.l_analysis_rules.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelectRule)


        #hsizer.AddStretchSpacer()

        vsizer.Add(self.l_analysis_rules, 1, wx.EXPAND, 0)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

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

        self.update()
        self.SetSizerAndFit(v_sizer)
        self._loaded_rules._updated.connect(self.update)
    
    def OnClear(self, wx_event=None):
        self._loaded_rules[self._selected_rule] = RuleChain()
        self._loaded_rules._updated.send(self)

    def OnAddRuleChain(self, wx_event=None):
        # display a text enty dialog to get the name of the new rule chain
        # then add it to the list of rule chains
        from PYME import pyme_warnings as warnings

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
        from PYME import pyme_warnings as warnings

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
        from PYME import pyme_warnings as warnings

        # TODO - specify the directory to start in
        # TODO - will this be YAML?
        dlg = wx.FileDialog(self, 'Choose a rule chain file', '', '', 'YAML files (*.yaml)|*.yaml', wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            
            with open(path, 'w') as f:
                f.write(self._loaded_rules[self._selected_rule].to_yaml())

    

    def OnRuleChoice(self, wx_event=None):
        self.select_rule_chain(self.c_rule_selection.GetStringSelection())

    def OnSelectRule(self, wx_event=None):
        selected = self.l_analysis_rules.GetFirstSelected()
        if selected != -1:
            self.select_rule_chain(self.l_analysis_rules.GetItemText(selected))

        wx_event.Skip()
    
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
        rule_keys = list(self._loaded_rules.keys())
        #self.c_rule_selection.SetItems(list(self._loaded_rules.keys()))
        #self.c_rule_selection.SetStringSelection(self._selected_rule)

        if not tuple(rule_keys) == self._displayed_rule_keys:
            self.l_analysis_rules.DeleteAllItems()
            for ind, rule in enumerate(rule_keys):
                self.l_analysis_rules.InsertItem(ind, rule)
                #self.l_analysis_rules.SetItem(ind, 1, str(self._loaded_rules[rule].rule_factories))
            self._displayed_rule_keys = tuple(rule_keys)
        
        for ind, rule in enumerate(rule_keys):
            #self.l_analysis_rules.InsertItem(ind, rule)
            self.l_analysis_rules.SetItem(ind, 1, str(self._loaded_rules[rule].rule_factories))
            if (rule == self._selected_rule) and not self.l_analysis_rules.IsSelected(ind):
                self.l_analysis_rules.Select(ind)
            elif (rule != self._selected_rule) and self.l_analysis_rules.IsSelected(ind):
                self.l_analysis_rules.Select(ind, False)


        self.l_analysis_rules.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.l_analysis_rules.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        
        #self.l_analysis_rules.Select(rule_keys.index(self._selected_rule))

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

    
def plug(main_frame, scope, default_pairings=None):
    """
    Adds a ChainedAnalysisPane to a microscope gui during start-up
    
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
    scope.analysis_rules.load_from_config()
    _recipe_manager = RecipeManager()
    _localization_settings = AnalysisSettings() #TODO - we should not need this to be global.

    main_frame.chained_analysis_page = ChainedAnalysisPage(main_frame, 
                                                            scope.analysis_rules,
                                                            _recipe_manager,
                                                            None,
                                                            default_pairings, localization_settings=_localization_settings)
    

    main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                    caption='Chained Analysis')
    
