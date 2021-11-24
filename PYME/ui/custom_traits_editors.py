from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

from PYME.recipes.traits import List, Instance

import wx
from PYME.IO import tabular

import logging
logger = logging.getLogger(__name__)

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
                                   style = wx.CB_DROPDOWN, value=str(self.value), choices=self.factory.choices)
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
                self.control.SetValue(str(self.value))

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
        from PYME.ui.filterPane import FilterPanel
        
        self.control = FilterPanel(parent, filterKeys=self.value, dataSource= lambda : getattr(self.object, '_ds'))
        return
    
    def update_editor ( self ):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        if self.value:
            self.control.populate()

        return
    
    def dispose(self):
        self.control = None

        print('Disposing of FilterEditor')

        super(Editor, self).dispose()

class FilterEditor(BasicEditorFactory):
    klass = _FilterEditor

class DictChoiceStrEditDialog(wx.Dialog):
    def __init__(self, parent, mode='new', possibleKeys=(), key='', val=''):
        wx.Dialog.__init__(self, parent, title='Edit ...')

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        hsizer.Add(wx.StaticText(self, -1, 'Input:'), 0, 
                   wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        self.cbKey = wx.ComboBox(self, -1, value=key, 
                                 choices=sorted(possibleKeys), 
                                 style=wx.CB_DROPDOWN, size=(150, -1))

        if not mode == 'new':
            self.cbKey.Enable(False)

        hsizer.Add(self.cbKey, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Table_Name:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.tVal = wx.TextCtrl(self, -1, val, size=(140, -1))
        

        hsizer.Add(self.tVal, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        vsizer.Add(hsizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        
        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        vsizer.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(vsizer)
        vsizer.Fit(self)


class DictChoiceStrPanel(wx.Panel):
    def __init__(self, parent, filterKeys, choices=None, unique_vals=True):
        """

        Parameters
        ----------
        parent : wx.Window

        filterKeys : dict
            Dictionary keys

        dataSource : function
            function to call to get the current data source
        """
        wx.Panel.__init__(self, parent)

        self.filterKeys = filterKeys
        self._choices = choices
        self._unique_vals = unique_vals

        
        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.lFiltKeys = wx.ListCtrl(self, -1, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.SUNKEN_BORDER, size=(-1, 25*(max(len(self.filterKeys.keys())+1, 5))))
        self.lFiltKeys.InsertColumn(0, 'Input')
        self.lFiltKeys.InsertColumn(1, 'Table Name')

        self.populate()

        self.lFiltKeys.SetColumnWidth(0, 150)  # wx.LIST_AUTOSIZE)
        self.lFiltKeys.SetColumnWidth(1, 150)  #wx.LIST_AUTOSIZE)

        # only do this part the first time so the events are only bound once
        if not hasattr(self, "ID_FILT_ADD"):
            self.ID_FILT_ADD = wx.NewId()
            self.ID_FILT_DELETE = wx.NewId()
            self.ID_FILT_EDIT = wx.NewId()

            self.Bind(wx.EVT_MENU, self.OnFilterAdd, id=self.ID_FILT_ADD)
            self.Bind(wx.EVT_MENU, self.OnFilterDelete, id=self.ID_FILT_DELETE)
            self.Bind(wx.EVT_MENU, self.OnFilterEdit, id=self.ID_FILT_EDIT)

        # for wxMSW
        self.lFiltKeys.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnFilterListRightClick)
        self.lFiltKeys.Bind(wx.EVT_COMMAND_LEFT_CLICK, self.OnFilterListRightClick)
        # for wxGTK
        self.lFiltKeys.Bind(wx.EVT_RIGHT_UP, self.OnFilterListRightClick)
        self.lFiltKeys.Bind(wx.EVT_LEFT_UP, self.OnFilterListRightClick)

        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnFilterItemSelected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnFilterItemDeselected)
        self.lFiltKeys.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnFilterEdit)
        
        vsizer.Add(self.lFiltKeys, 1, wx.ALL|wx.EXPAND, 0)
        self.SetSizerAndFit(vsizer)
        
    def update(self, filter_keys, choices):
        self.filterKeys = filter_keys
        self._choices = choices
        self.populate()

    def populate(self):
        self.lFiltKeys.DeleteAllItems()
        ind = 0
        for key, value in self.filterKeys.items():
            ind = self.lFiltKeys.InsertStringItem(ind+1, key)
            self.lFiltKeys.SetStringItem(ind, 1, value)

    @property
    def choices(self):
        if self._choices is None:
            return None
        elif callable(self._choices):
            #support passing data source as a callable
            return self._choices()
        else:
            return self._choices

    def OnFilterListRightClick(self, event):
        x = event.GetX()
        y = event.GetY()

        item, flags = self.lFiltKeys.HitTest((x, y))

        menu = wx.Menu()
        menu.Append(self.ID_FILT_ADD, "Add")

        if item != wx.NOT_FOUND and flags & wx.LIST_HITTEST_ONITEM:
            self.currentFilterItem = item
            self.lFiltKeys.Select(item)

            menu.Append(self.ID_FILT_DELETE, "Delete")
            menu.Append(self.ID_FILT_EDIT, "Edit")

        # Popup the menu.  If an item is selected then its handler
        # will be called before PopupMenu returns.
        self.PopupMenu(menu)
        menu.Destroy()

    def OnFilterItemSelected(self, event):
        self.currentFilterItem = event.GetIndex()
        event.Skip()

    def OnFilterItemDeselected(self, event):
        self.currentFilterItem = None
        event.Skip()

    def OnFilterAdd(self, event):
        import sys

        try:
            possibleKeys = list(self.choices)
        except:
            possibleKeys = []

        dlg = DictChoiceStrEditDialog(self, mode='new', 
                                      possibleKeys=possibleKeys)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            val = str(dlg.tVal.GetValue())
            key = str(dlg.cbKey.GetValue())

            if key == '':
                return
            elif self._unique_vals and (val in self.filterKeys.values()):
                raise UserWarning('Please choose a unique Table Name')

            self.filterKeys[key] = val

            ind = self.lFiltKeys.InsertStringItem(sys.maxsize, key)
            self.lFiltKeys.SetStringItem(ind, 1, val)

        dlg.Destroy()

    def OnFilterDelete(self, event):
        it = self.lFiltKeys.GetItem(self.currentFilterItem)
        self.lFiltKeys.DeleteItem(self.currentFilterItem)
        self.filterKeys.pop(it.GetText())

    def OnFilterEdit(self, event):
        key = str(self.lFiltKeys.GetItem(self.currentFilterItem).GetText())
        val = self.filterKeys[key]
        
        dlg = DictChoiceStrEditDialog(self, mode='edit', key=key, val=val)

        ret = dlg.ShowModal()

        if ret == wx.ID_OK:
            val = str(dlg.tVal.GetValue())
            if self._unique_vals and (val in self.filterKeys.values()):
                raise UserWarning('Please choose a unique Table Name')
            self.filterKeys[key] = val
            self.lFiltKeys.SetStringItem(self.currentFilterItem, 1, val)

        dlg.Destroy()

class _DictChoiceStrEditor(Editor):
    def init(self, parent):
        """
        Finishes initializing the editor by creating the underlying widget.
        """
        from PYME.ui.filterPane import FilterPanel


        self.control = DictChoiceStrPanel(parent, filterKeys=self.value, 
                                          choices=self.factory.choices)


    def update_editor(self):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        if self.value:
            self.control.populate()
    
    def dispose(self):
        self.control = None
        super(Editor, self).dispose()

class DictChoiceStrEditor(BasicEditorFactory):
    klass = _DictChoiceStrEditor
    choices = List()

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