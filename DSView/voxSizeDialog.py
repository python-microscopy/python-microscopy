import wx

class VoxSizeDialog(wx.Dialog):
    def __init__(self, parent, varnames=['Orte']):
        wx.Dialog.__init__(self, parent, title='Please enter voxel size')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, u'x [\u03BCm]:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tVoxX = wx.TextCtrl(self, -1, '0.07')

        sizer2.Add(self.tVoxX, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        sizer1.Add(wx.StaticText(self, -1, 'Field Names must evaluate to a list of field names present in the file,\n and should use "x", "y", "sig", "A"  and "error_x" to refer to the \nfitted position, std dev., amplitude, and error respectively.'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, 'Field Names:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tFieldNames = wx.TextCtrl(self, -1, '"A","x","y","error_x","error_y","sig_x","sig_y","Nphotons","t"', size=(250, -1))

        sizer2.Add(self.tFieldNames, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 0)

        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def GetFieldNames(self):
        return eval(self.tFieldNames.GetValue())

    def GetVarName(self):
        return self.cbVarNames.GetValue()