from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

from traits.api import List, Instance

import wx
from PYME.IO import tabular

class _CBEditor (Editor):
    """
    Simple Traits UI date editor.  Shows a text box, and a date-picker widget.
    """

    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying widget.
        """


        self.control = wx.ComboBox(parent,
                                   size=(120,-1),
                                   style = wx.CB_DROPDOWN|wx.CB_SORT, value=self.value, choices=self.factory.choices)
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
    Simple Traits UI date editor.  Shows a text box, and a date-picker widget.
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
    
class _HistLimitsEditor (Editor):
    """
    Simple Traits UI date editor.  Shows a text box, and a date-picker widget.
    """

    def init ( self, parent ):
        """
        Finishes initializing the editor by creating the underlying widget.
        """
        from PYME.ui import histLimits
        
        l_lower, l_upper = self.value

        self.control = histLimits.HistLimitPanel(parent, -1, data=self.factory.data(), limit_lower=l_lower, limit_upper=l_upper)
                                   
        self.control.Bind(histLimits.EVT_LIMIT_CHANGE, self.limits_changed)
        
        return


    def limits_changed(self, event=None):
        """
        Event for when calendar is selected, update/create date string.
        """
        self.value = list(self.control.GetValue())
        print(self.value)
        return


    def update_editor ( self ):
        """
        Updates the editor when the object trait changes externally to the
        editor.
        """
        print('hl update')
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