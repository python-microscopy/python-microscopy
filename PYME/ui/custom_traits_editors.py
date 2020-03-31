from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

from traits.api import List, Instance

import wx
from PYME.IO import tabular

class _CBEditor (Editor):
    """
    Dropdown list of options (as strings). Generally passed a list (choices=list_name).
    Example: Used to select datasource and color channel in 
    PYME.recipes.localisations.ExtractTableChannel.
    """

    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying widget.
        """


        self.control = wx.ComboBox(parent,
                                   size=(120,-1),
                                   style = wx.CB_DROPDOWN, value=self.value, choices=self.factory.choices)
        self.control.Bind(wx.EVT_COMBOBOX, self.text_changed)
        #self.control.Bind(wx.EVT_COMBOBOX_CLOSEUP, lambda e: print('foo'))
        self.control.Bind(wx.EVT_TEXT, self.text_changed)
        return


    def text_changed(self, event=None):
        """
        Event for when calendar is selected, update/create date string.
        """
        self.value = self.control.GetValue()
        print(self.value)
        return


    def update_editor ( self ):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        if self.value:
            choices = [self.control.GetString(n) for n in range(self.control.GetCount())]
            try:
                n = choices.index(self.value)
                self.control.SetSelection(n)
            except ValueError:
                self.control.SetValue(self.value)

        return
    
    def dispose(self):
        self.control.Bind(wx.EVT_COMBOBOX, None)
        self.control.Bind(wx.EVT_TEXT, None)
        self.control = None

        print('Disposing of CBEditor')

        super(Editor, self).dispose()

class CBEditor(BasicEditorFactory):
    klass = _CBEditor

    choices = List()

class _FilterEditor (Editor):
    """
    Editable table with three columns, "Key", "Min", and "Max", for specifying
    filters to apply to a datasource. Generally passed a dataSource, and users
    are asked to specify which keys to filter on.
    Example: PYME.recipes.tablefilters.FilterTable.
    """

    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying widget.
        """
        from PYME.LMVis.filterPane import FilterPanel


        self.control = FilterPanel(parent, filterKeys=self.value, dataSource=self.factory.datasource)
        return


    def update_editor ( self ):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        if self.value:
            self.control.populate()
            #choices = [self.control.GetString(n) for n in range(self.control.GetCount())]
            #try:
            #    n = choices.index(self.value)
            #    self.control.SetSelection(n)
            #except ValueError:
            #    self.control.SetValue(self.value)

        return
    
    def dispose(self):
        self.control = None

        print('Disposing of FilterEditor')

        super(Editor, self).dispose()

class FilterEditor(BasicEditorFactory):
    klass = _FilterEditor

    datasource = Instance(tabular.TabularBase)



from . import editList

class DictFloatEditList(editList.EditListCtrl):
    def __init__(self, parent, value_dict, editor):
        editList.EditListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT|wx.LC_SINGLE_SEL|wx.SUNKEN_BORDER, size=(-1, 100))
        self.value_dict = value_dict
        self._editor = editor
        
        self.InsertColumn(0, 'Parameter')
        self.InsertColumn(1, 'Value')

        self.makeColumnEditable(1)
        self.populate()

        self.Bind(wx.EVT_LIST_END_LABEL_EDIT, self._on_values_change)
        
    def _on_values_change(self, event):
        keys = sorted(self.value_dict.keys())
    
        k = keys[event.m_itemIndex]
    
        self.value_dict[k] = float(event.m_item.GetText())
        
        self._editor.value = self.value_dict
        
    def populate(self, value_dict=None):
        from PYME.ui import UI_MAXSIZE
        
        if not value_dict is None:
            self.value_dict = value_dict
        
        self.DeleteAllItems()
        
        for k in sorted(self.value_dict.keys()):
            ind = self.InsertStringItem(UI_MAXSIZE, k)
            self.SetStringItem(ind, 1, '%1.3g' % self.value_dict[k])

        self.SetColumnWidth(0, 80)
        self.SetColumnWidth(1, 80)
        

class _DictFloatEditor(Editor):
    """
    Provides a column editor for dictionaries, similar to FilterEditor. Not
    used anywhere in PYME at the moment (31 March 2020).
    """
    
    def init(self, parent):
        """
        Finishes initializing the editor by creating the underlying widget.
        """
        #from PYME.LMVis.filterPane import FilterPanel
        
        
        self.control = DictFloatEditList(parent, value_dict=self.value, editor=self)
        return
    
    def update_editor(self):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        if self.value:
            self.control.populate(self.value)
            #choices = [self.control.GetString(n) for n in range(self.control.GetCount())]
            #try:
            #    n = choices.index(self.value)
            #    self.control.SetSelection(n)
            #except ValueError:
            #    self.control.SetValue(self.value)
        
        return
    
    def dispose(self):
        self.control = None
        
        print('Disposing of DictFloatEditor')
        
        super(Editor, self).dispose()


class DictFloatEditor(BasicEditorFactory):
    klass = _DictFloatEditor
    
    #datasource = Instance(tabular.TabularBase)
    
class _HistLimitsEditor (Editor):
    """
    Custom traits UI editor for displaying and adjusting the limits of a 
    histogram. Example: Used to set the limits of parameters used to color
    pointclouds in PYME.LMVis.layers.pointcloud.
    """

    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying widget.
        """
        from PYME.ui import histLimits
        
        l_lower, l_upper = self.value

        self.control = histLimits.HistLimitPanel(parent, -1, data=self.factory.data(), limit_lower=l_lower, limit_upper=l_upper)
                                   
        self.control.Bind(histLimits.EVT_LIMIT_CHANGE, self.limits_changed)
        
        if not self.factory.update_signal is None:
            self.factory.update_signal.connect(self.n_update)
        
        return


    def limits_changed(self, event=None):
        """
        Event for when histogram limits are changed, update/create lower/upper bound.
        """
        self.value = list(self.control.GetValue())
        print(self.value)
        return
    
    def n_update(self, *args, **kwargs):
        self.update_editor()


    def update_editor ( self ):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        import traceback
        print('hl update')
        #print traceback.print_stack()
        if self.value:
            self.control.SetData(self.factory.data(), *self.value)
            self.control.SetValue(self.value)

        return
    
    def dispose(self):
        from PYME.ui import histLimits
        self.control.Bind(histLimits.EVT_LIMIT_CHANGE, None)
        self.control = None

        print('Disposing of HistLimitsEditor')

        super(Editor, self).dispose()

class HistLimitsEditor(BasicEditorFactory):
    klass = _HistLimitsEditor

    data = Instance(object)
    update_signal = Instance(object)