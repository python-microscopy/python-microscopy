
import wx
# from  PYME.ui import manualFoldPanel
from PYME.cluster.rules import LocalisationRuleFactory as LocalizationRuleFactory
from PYME.cluster.rules import RecipeRuleFactory
from collections import OrderedDict
import queue
import os
import posixpath
import logging
from PYME.contrib import dispatch, wxPlotPanel
from PYME.recipes.traits import HasTraits, Enum, Float, CStr
import textwrap
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

POST_CHOICES = ['off', 'spool start', 'spool stop']

def get_protocol_list():
    """version of PYME.Acquire.protocol.get_protocol_list which uses 'default'
    instead of '<None>' and drops all '.py' extensions
    """
    from PYME.Acquire.protocol import _get_protocol_dict
    protocol_list = ['default', ] + sorted(list(_get_protocol_dict().keys()))
    return [os.path.splitext(p)[0] for p in protocol_list]

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


class ProtocolRules(OrderedDict):
    """
    Container for associating sets of analysis rules with specific acquisition
    protocols

    Notes
    -----
    use ordered dict for reproducibility with listctrl displays
    """
    def __init__(self, spool_controller, posting_thread_queue_size=5):
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
        self._spool_controller = spool_controller
        self.posting_thread_queue = queue.Queue(posting_thread_queue_size)
        self._updated = dispatch.Signal()
        self._updated.connect(self.update)
        
        self['default'] = RuleChain()

        self._spool_controller.onSpoolStart.connect(self.on_spool_start)
        self._spool_controller.onSpoolStop.connect(self.on_spool_stop)
    
    def on_spool_start(self, **kwargs):
        self.on_spool_event('spool start')
    
    def on_spool_stop(self, **kwargs):
        self.on_spool_event('spool stop')
    
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
        rule_factory_chain.rule_factories[0].get_rule(context=context).push()
    
    def update(self, *args, **kwargs):
        for p in self.keys():
            factories = self[p].rule_factories
            for ind in range(len(factories) - 1):
                factories[ind].chain(factories[ind + 1])


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
        if col == 0:
            return list(self._protocol_rules.keys())[item]
        if col == 1:
            chains = list(self._protocol_rules.values())
            return str(len(chains[item].rule_factories))
        if col == 2:
            chains = list(self._protocol_rules.values())
            return chains[item].post_on

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
            rect._data = s
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
        default_pairings : dict
            protocol keys with lists of RuleFactorys as values to prepopulate
            panel on start up
        """
        wx.Panel.__init__(self, parent, -1)
        self._protocol_rules = protocol_rules
        self._selected_protocol = list(protocol_rules.keys())[0]
        self._recipe_manager = recipe_manager

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        # associated protocol choicebox (default / others on top, then rest)
        hsizer.Add(wx.StaticText(self, -1, 'Associated Protocol:'), 0, 
                   wx.LEFT|wx.RIGHT, 5)
        self.c_protocol = wx.Choice(self, -1, choices=get_protocol_list())
        self.c_protocol.SetSelection(0)
        self.c_protocol.Bind(wx.EVT_CHOICE, self.OnProtocolChoice)
        hsizer.Add(self.c_protocol, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        # post on choicebox
        hsizer.Add(wx.StaticText(self, -1, 'Post On:'), 0, 
                   wx.LEFT|wx.RIGHT, 5)
        self.c_post = wx.Choice(self, -1, choices=POST_CHOICES)
        self.c_post.SetSelection(0)
        self.c_post.Bind(wx.EVT_CHOICE, self.OnPostChoice)
        hsizer.Add(self.c_post, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        vsizer.Add(hsizer, 0)

        # rule plot
        self.rule_plot = RulePlotPanel(self, self._protocol_rules, 
                                       size=(-1, 400))
        vsizer.Add(self.rule_plot, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.b_clear = wx.Button(self, -1, 'Clear Chain')
        hsizer.Add(self.b_clear, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_clear.Bind(wx.EVT_BUTTON, self.OnClear)

        self.b_add_recipe = wx.Button(self, -1, 'Add Recipe')
        hsizer.Add(self.b_add_recipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_recipe.Bind(wx.EVT_BUTTON, self.OnAddRecipe)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)
        self._protocol_rules._updated.connect(self.update)
    
    def OnClear(self, wx_event=None):
        self._protocol_rules[self._selected_protocol] = RuleChain()
        self._protocol_rules._updated.send(self)

    def OnAddRecipe(self, wx_event=None):
        self._recipe_manager.OnAddRecipeRule()

    def OnProtocolChoice(self, wx_event=None):
        self.select_rule_chain(self.c_protocol.GetStringSelection())
    
    def select_rule_chain(self, protocol='default'):
        if protocol not in self._protocol_rules.keys():
            self._protocol_rules[protocol] = RuleChain()
            self._protocol_rules._updated.send(self)
        self._selected_protocol = protocol
        # force a redraw, even though we might just have done so if we added
        self.update()

    def OnPostChoice(self, wx_event=None):
        self._protocol_rules[self._selected_protocol].post_on = self.c_post.GetStringSelection()
        self._protocol_rules._updated.send(self)
        
    def _set_text_styling(self):
        from wx import stc
        ed = self.tRecipeText
        ed.SetLexer(wx.stc.STC_LEX_YAML)

        # Enable folding
        ed.SetProperty("fold", "1")

        # Highlight tab/space mixing (shouldn't be any)
        ed.SetProperty("tab.timmy.whinge.level", "1")

        # Set left and right margins
        ed.SetMargins(2, 2)

        # Set up the numbers in the margin for margin #1
        ed.SetMarginType(1, wx.stc.STC_MARGIN_NUMBER)
        # Reasonable value for, say, 4-5 digits using a mono font (40 pix)
        ed.SetMarginWidth(1, 40)

        # Indentation and tab stuff
        ed.SetIndent(4)               # Proscribed indent size for wx
        ed.SetIndentationGuides(True) # Show indent guides
        ed.SetBackSpaceUnIndents(True)# Backspace unindents rather than delete 1 space
        ed.SetTabIndents(True)        # Tab key indents
        ed.SetTabWidth(4)             # Proscribed tab size for wx
        ed.SetUseTabs(False)          # Use spaces rather than tabs, or
        # TabTimmy will complain!    
        # White space
        ed.SetViewWhiteSpace(False)   # Don't view white space

        # EOL: Since we are loading/saving ourselves, and the
        # strings will always have \n's in them, set the STC to
        # edit them that way.            
        ed.SetEOLMode(wx.stc.STC_EOL_LF)
        ed.SetViewEOL(False)

        # No right-edge mode indicator
        ed.SetEdgeMode(stc.STC_EDGE_NONE)

        # Setup a margin to hold fold markers
        ed.SetMarginType(2, stc.STC_MARGIN_SYMBOL)
        ed.SetMarginMask(2, stc.STC_MASK_FOLDERS)
        ed.SetMarginSensitive(2, True)
        ed.SetMarginWidth(2, 12)

        # and now set up the fold markers
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDEREND, stc.STC_MARK_BOXPLUSCONNECTED, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDEROPENMID, stc.STC_MARK_BOXMINUSCONNECTED, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDERMIDTAIL, stc.STC_MARK_TCORNER, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDERTAIL, stc.STC_MARK_LCORNER, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDERSUB, stc.STC_MARK_VLINE, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDER, stc.STC_MARK_BOXPLUS, "white", "black")
        ed.MarkerDefine(stc.STC_MARKNUM_FOLDEROPEN, stc.STC_MARK_BOXMINUS, "white", "black")

        # Global default style
        if wx.Platform == '__WXMSW__':
            ed.StyleSetSpec(stc.STC_STYLE_DEFAULT,
                              'fore:#000000,back:#FFFFFF,face:Courier New')
        elif wx.Platform == '__WXMAC__':
            # TODO: if this looks fine on Linux too, remove the Mac-specific case 
            # and use this whenever OS != MSW.
            ed.StyleSetSpec(stc.STC_STYLE_DEFAULT,
                              'fore:#000000,back:#FFFFFF,face:Monaco')
        else:
            defsize = wx.SystemSettings.GetFont(wx.SYS_ANSI_FIXED_FONT).GetPointSize()
            ed.StyleSetSpec(stc.STC_STYLE_DEFAULT,
                              'fore:#000000,back:#FFFFFF,face:Courier,size:%d' % defsize)

        # Clear styles and revert to default.
        ed.StyleClearAll()

        # Following style specs only indicate differences from default.
        # The rest remains unchanged.

        # Line numbers in margin
        ed.StyleSetSpec(wx.stc.STC_STYLE_LINENUMBER, 'fore:#000000,back:#99A9C2')
        # Highlighted brace
        ed.StyleSetSpec(wx.stc.STC_STYLE_BRACELIGHT, 'fore:#00009D,back:#FFFF00')
        # Unmatched brace
        ed.StyleSetSpec(wx.stc.STC_STYLE_BRACEBAD, 'fore:#00009D,back:#FF0000')
        # Indentation guide
        ed.StyleSetSpec(wx.stc.STC_STYLE_INDENTGUIDE, "fore:#CDCDCD")

        # YAML styles
        ed.StyleSetSpec(wx.stc.STC_YAML_DEFAULT, 'fore:#000000')
        ed.StyleSetSpec(wx.stc.STC_YAML_COMMENT, 'fore:#008000,back:#F0FFF0')
        ed.StyleSetSpec(wx.stc.STC_YAML_NUMBER, 'fore:#0080F0')
        ed.StyleSetSpec(wx.stc.STC_YAML_IDENTIFIER, 'fore:#80000')
        ed.StyleSetSpec(wx.stc.STC_YAML_DOCUMENT, 'fore:#E0E000') #what is this?
        ed.StyleSetSpec(wx.stc.STC_YAML_KEYWORD, 'fore:#000080,bold')
        ed.StyleSetSpec(wx.stc.STC_YAML_ERROR, 'fore:#FE2020')
        ed.StyleSetSpec(wx.stc.STC_YAML_OPERATOR, 'fore:#0000A0')
        ed.StyleSetSpec(wx.stc.STC_YAML_REFERENCE, 'fore:#E0E000')
        ed.StyleSetSpec(wx.stc.STC_YAML_TEXT, 'fore:#E0E000')

        # Caret color
        ed.SetCaretForeground("BLUE")
        # Selection background
        ed.SetSelBackground(1, '#66CCFF')
        
    def update(self, *args, **kwargs):
        self.rule_plot.draw()
        
    def OnPick(self, event):
        # FIXME - open rule in respective editing tab
        raise NotImplementedError
        from PYME.IO import tabular
        k = event.artist._data
        if not (isinstance(k, six.string_types)):
            self.configureModule(k)
        else:
            outp = self.recipes.activeRecipe.namespace[k]
            if isinstance(outp, ImageStack):
                if not 'dsviewer' in dir(self.recipes):
                    dv = ViewIm3D(outp, mode='lite')
                else:
                    if self.recipes.dsviewer.mode == 'visGUI':
                        mode = 'visGUI'
                    else:
                        mode = 'lite'
                                   
                    dv = ViewIm3D(outp, mode=mode, glCanvas=self.recipes.dsviewer.glCanvas)
                    
            elif isinstance(outp, tabular.TabularBase):
                from PYME.ui import recArrayView
                f = recArrayView.ArrayFrame(outp, parent=self, title='Data table - %s' % k)
                f.Show()
    
    @property
    def rule_chain(self):
        return self._protocol_rules[self._selected_protocol]

    def add_tile(self, rule_tile):
        self.rule_chain.rule_factories.append(rule_tile)
        self._protocol_rules._updated.send(self)


class SMLMChainedAnalysisPage(ChainedAnalysisPage):
    def __init__(self, parent, protocol_rules, recipe_manager, 
                 localization_panel, default_pairings=None):
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
        self._protocol_rules = protocol_rules
        self._selected_protocol = list(protocol_rules.keys())[0]
        self._recipe_manager = recipe_manager
        self._localization_panel = localization_panel

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        # associated protocol choicebox (default / others on top, then rest)
        hsizer.Add(wx.StaticText(self, -1, 'Associated Protocol:'), 0, 
                   wx.LEFT|wx.RIGHT, 5)
        self.c_protocol = wx.Choice(self, -1, choices=get_protocol_list())
        self.c_protocol.SetSelection(0)
        self.c_protocol.Bind(wx.EVT_CHOICE, self.OnProtocolChoice)
        hsizer.Add(self.c_protocol, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        # post on choicebox
        hsizer.Add(wx.StaticText(self, -1, 'Post On:'), 0, 
                   wx.LEFT|wx.RIGHT, 5)
        self.c_post = wx.Choice(self, -1, choices=POST_CHOICES)
        self.c_post.SetSelection(0)
        self.c_post.Bind(wx.EVT_CHOICE, self.OnPostChoice)
        hsizer.Add(self.c_post, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        vsizer.Add(hsizer, 0)

        # rule plot
        self.rule_plot = RulePlotPanel(self, self._protocol_rules, 
                                       size=(-1, 400))
        vsizer.Add(self.rule_plot, 1, wx.ALL|wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.b_clear = wx.Button(self, -1, 'Clear Chain')
        hsizer.Add(self.b_clear, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_clear.Bind(wx.EVT_BUTTON, self.OnClear)

        self.b_add_recipe = wx.Button(self, -1, 'Add Recipe')
        hsizer.Add(self.b_add_recipe, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_recipe.Bind(wx.EVT_BUTTON, self.OnAddRecipe)
        
        self.b_add_localization = wx.Button(self, -1, 'Add Localization')
        hsizer.Add(self.b_add_localization, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_localization.Bind(wx.EVT_BUTTON, self.OnAddLocalization)
        
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)
        self._protocol_rules._updated.connect(self.update)
    
    def OnAddLocalization(self, wx_event=None):
        self._localization_panel.OnAddLocalizationRule()

from PYME.recipes.recipeGui import RecipeView, RecipeManager, RecipePlotPanel
class RuleRecipeView(RecipeView):
    def __init__(self, parent, recipes):
        wx.Panel.__init__(self, parent, size=(400, 100))
        self.recipes = recipes
        recipes.recipeView = self  # weird plug
        self._editing = False #are we currently editing a recipe module? used for a hack / workaround for a a traits/matplotlib bug to disable click-throughs
        hsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        self.recipePlot = RecipePlotPanel(self, recipes, size=(-1, 400))
        vsizer.Add(self.recipePlot, 1, wx.ALL | wx.EXPAND, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.bNewRecipe = wx.Button(self, -1, 'Clear Recipe')
        hsizer.Add(self.bNewRecipe, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.bNewRecipe.Bind(wx.EVT_BUTTON, self.OnNewRecipe)
        
        self.bLoadRecipe = wx.Button(self, -1, 'Load Recipe')
        hsizer.Add(self.bLoadRecipe, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.bLoadRecipe.Bind(wx.EVT_BUTTON, self.recipes.OnLoadRecipe)
        
        self.bAddModule = wx.Button(self, -1, 'Add Module')
        hsizer.Add(self.bAddModule, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.bAddModule.Bind(wx.EVT_BUTTON, self.OnAddModule)
        
        #self.bRefresh = wx.Button(self, -1, 'Refresh')
        #hsizer.Add(self.bRefresh, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)
        
        self.bSaveRecipe = wx.Button(self, -1, 'Save Recipe')
        hsizer.Add(self.bSaveRecipe, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.bSaveRecipe.Bind(wx.EVT_BUTTON, self.recipes.OnSaveRecipe)
        
        self.b_add_recipe_rule = wx.Button(self, -1,
                                           'Add Recipe to Chained Analysis')
        hsizer.Add(self.b_add_recipe_rule, 0,
                   wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        self.b_add_recipe_rule.Bind(wx.EVT_BUTTON,
                                    self.recipes.OnAddRecipeRule)
        vsizer.Add(hsizer, 0, wx.EXPAND, 0)
        
        hsizer1.Add(vsizer, 1, wx.EXPAND | wx.ALL, 2)
        
        vsizer = wx.BoxSizer(wx.VERTICAL)
        
        #self.tRecipeText = wx.TextCtrl(self, -1, '', size=(350, -1),
        #                               style=wx.TE_MULTILINE|wx.TE_PROCESS_ENTER)
        self.tRecipeText = wx.stc.StyledTextCtrl(self, -1, size=(400, -1))
        self._set_text_styling()
        
                                       
        vsizer.Add(self.tRecipeText, 1, wx.ALL, 2)
        
        self.bApply = wx.Button(self, -1, 'Apply Text Changes')
        vsizer.Add(self.bApply, 0, wx.ALL, 2)
        self.bApply.Bind(wx.EVT_BUTTON, self.OnApplyText)
        
        self.bApply.Disable()
        self.tRecipeText.Bind(wx.stc.EVT_STC_MODIFIED, lambda e : self.bApply.Enable())
        
        hsizer1.Add(vsizer, 0, wx.EXPAND | wx.ALL, 2)
        
        self.SetSizerAndFit(hsizer1)
        
        self.recipes.LoadRecipeText('')
        
        recipes.activeRecipe.recipe_changed.connect(self.update)
        recipes.activeRecipe.recipe_executed.connect(self.update)

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
        scope : PYME.Acquire.microscope.microscope
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
class LocalizationSettingsPanel(manualFoldPanel.foldingPane):
    def __init__(self, wx_parent, localization_settings, mdh_changed_signal=None,
                 chained_analysis_page=None):
        from PYME.ui.autoFoldPanel import collapsingPane
        manualFoldPanel.foldingPane.__init__(self, wx_parent, caption='Localization Analysis')
        
        self.localization_settings = localization_settings
        self.localization_mdh = localization_settings.analysisMDH
        self.mdh_changed_signal = mdh_changed_signal
        self.chained_analysis_page = chained_analysis_page
        
        clp = collapsingPane(self, caption='settings ...')
        clp.AddNewElement(AnalysisSettingsPanel(clp,
                                                self.localization_settings,
                                                self.mdh_changed_signal))
        clp.AddNewElement(AnalysisDetailsPanel(clp, self.localization_settings,
                                               self.mdh_changed_signal))
        self.AddNewElement(clp)
        
        # add box to propagate rule to rule chain
        add_rule_panel = wx.Panel(self, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.b_add_rule = wx.Button(add_rule_panel, -1,
                                    'Add Localization to Chained Analysis')
        self.b_add_rule.Bind(wx.EVT_BUTTON, self.OnAddLocalizationRule)
        h_sizer.Add(self.b_add_rule)
        v_sizer.Add(h_sizer, 0, wx.EXPAND | wx.TOP, 0)
        add_rule_panel.SetSizerAndFit(v_sizer)
        self.AddNewElement(add_rule_panel)
    
    def OnAddLocalizationRule(self, wx_event=None):
        from PYME.cluster.rules import SpoolLocalLocalizationRuleFactory
        from PYME.IO.MetaDataHandler import DictMDHandler
        if self.chained_analysis_page is None:
            logger.error('chained_analysis_page attribute unset')
            return
        
        mdh = DictMDHandler(self.localization_mdh)
        loc_rule = get_rule_tile(SpoolLocalLocalizationRuleFactory)(analysisMetadata=mdh)
        self.chained_analysis_page.add_tile(loc_rule)

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
        scope : PYME.Acquire.microscope.microscope
            the microscope itself
        default_pairings : dict
            [optional] protocol keys with RuleChains as values to
            prepopulate panel on start up. By default, None
        """
        from PYME.Acquire.ui.AnalysisSettingsUI import AnalysisSettings

        scope.protocol_rules = ProtocolRules(scope.spoolController)
        scope._recipe_manager = RuleRecipeManager()
        scope._localization_settings = AnalysisSettings()

        main_frame.localization_settings = LocalizationSettingsPanel(main_frame,
                                                                     scope._localization_settings,
                                                                     scope._localization_settings.onMetadataChanged)

        main_frame.chained_analysis_page = SMLMChainedAnalysisPage(main_frame, 
                                                                   scope.protocol_rules,
                                                                   scope._recipe_manager,
                                                                   main_frame.localization_settings,
                                                                   default_pairings)
        
        scope._recipe_manager.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.recipe_view = RuleRecipeView(main_frame, scope._recipe_manager)
        main_frame.localization_settings.chained_analysis_page = main_frame.chained_analysis_page
        main_frame.AddPage(page=main_frame.recipe_view, select=False, caption='Recipe')
        main_frame.AddPage(page=main_frame.localization_settings, select=False, caption='Localization')
        main_frame.AddPage(page=main_frame.chained_analysis_page, select=False,
                           caption='Chained Analysis')
        
        # add this panel
        chained_analysis = SMLMChainedAnalysisPanel(main_frame, 
                                                    scope.protocol_rules,
                                                    main_frame.chained_analysis_page,
                                                    default_pairings)
        main_frame.anPanels.append((chained_analysis, 'Automatic Analysis', 
                                    True))
