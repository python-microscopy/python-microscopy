#!/usr/bin/python

##################
# importTextDialog.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import wx
from wx.lib.scrolledpanel import ScrolledPanel


class ImportTextDialog(wx.Dialog):
    requiredVariables = {'x':'x position [nm]',
                        'y':'y position [nm]'}
    recommendedVariables = {'A':'amplitude of Gaussian',
                            't':'time [frames]',
                            'sig':'std. deviation of Gaussian [nm]',
                            'error_x':'fit error in x direction [nm]'}
    niceVariables = {}

    def __init__(self, parent, textFileName):
        wx.Dialog.__init__(self, parent, title='Import data from text file')

        self.colNames, self.dataLines = self.TextFileHeaderParse(textFileName)

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(wx.StaticText(self, -1, 'Please assign variable names to each column. Some variable names must be present for the program to function correctly'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.GenerateDataGrid(sizer1)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.stRequiredNotPresent = wx.StaticText(self, -1, 'Required variables not yet defined:\n')
        sizer2.Add(self.stRequiredNotPresent, 1,  wx.ALL, 5)

        self.stRecommendedNotPresent = wx.StaticText(self, -1, 'Recommended variables not yet defined:\n')
        sizer2.Add(self.stRecommendedNotPresent, 1, wx.ALL, 5)
        sizer1.Add(sizer2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL|wx.EXPAND, 5)
        
        btSizer = wx.StdDialogButtonSizer()

        self.btnOK = wx.Button(self, wx.ID_OK)
        self.btnOK.SetDefault()
        self.btnOK.Enable(False)

        btSizer.AddButton(self.btnOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.CheckColNames()

        self.SetSizer(sizer1)
        sizer1.Fit(self)

    def TextFileHeaderParse(self, filename):
        n = 0
        commentLines = []
        dataLines = []

        fid = open(filename, 'r')

        while n < 10:
            line = fid.readline()
            if line.startswith('#'): #check for comments
                commentLines.append(line[1:])
            else:
                dataLines.append(line.split())
                n += 1

        numCols = len(dataLines[0])

        if len(commentLines) > 0 and len(commentLines[-1].split()) == numCols:
            colNames = commentLines[-1].split()
        else:
            colNames = ['column_%d' % i for i in range(numCols)]

        return colNames, dataLines

    def CheckColNames(self):
        reqNotDef = [var for var in self.requiredVariables.keys() if not var in self.colNames]

        if len(reqNotDef) > 0:
            self.btnOK.Enable(False)
            sreq = 'Required variables not yet defined:\n'
            for k in reqNotDef:
                sreq += '\n\t%s\t-\t%s' % (k, self.requiredVariables[k])
            self.stRequiredNotPresent.SetForegroundColour(wx.RED)
        else:
            self.btnOK.Enable(True)
            sreq = 'All required variables are defined\n'
            self.stRequiredNotPresent.SetForegroundColour(wx.GREEN)

        self.stRequiredNotPresent.SetLabel(sreq)

        recNotDef = [var for var in self.recommendedVariables.keys() if not var in self.colNames]

        if len(recNotDef) > 0:
            sreq = 'Recomended variables not yet defined:\n'
            for k in recNotDef:
                sreq += '\n\t%s\t-\t%s' % (k, self.recommendedVariables[k])
            self.stRecommendedNotPresent.SetForegroundColour(wx.Colour(200, 150, 0))
        else:
            sreq = 'All recommended variables are defined\n'
            self.stRecommendedNotPresent.SetForegroundColour(wx.GREEN)

        self.stRecommendedNotPresent.SetLabel(sreq)

        

    def GenerateDataGrid(self, sizer):
        vsizer = wx.StaticBoxSizer(wx.StaticBox(self, -1, 'Column names and preview:'))
        self.scrollW = ScrolledPanel(self, -1, size=(800, 150))
        self.comboIDs = []
        self.combos = []

        fgSizer = wx.FlexGridSizer(1+len(self.dataLines), len(self.colNames), 4, 4)

        for cn in self.colNames:
            id = wx.NewId()
            self.comboIDs.append(id)

            cb = wx.ComboBox(self.scrollW, id, size=(120, -1), choices=[cn]+ self.requiredVariables.keys() +self.recommendedVariables.keys())
            self.combos.append(cb)
            cb.SetSelection(0)

            cb.Bind(wx.EVT_COMBOBOX, self.OnColNameChange)
            cb.Bind(wx.EVT_TEXT, self.OnColNameChange)

            fgSizer.Add(cb)

        for dl in self.dataLines:
            for de in dl:
                fgSizer.Add(wx.StaticText(self.scrollW, -1, de))

        self.scrollW.SetSizer(fgSizer)
        self.scrollW.SetAutoLayout(True)
        self.scrollW.SetupScrolling()

        vsizer.Add(self.scrollW, 0, wx.EXPAND|wx.ALL,5)
        sizer.Add(vsizer, 0, wx.EXPAND|wx.ALL,5)

    def OnColNameChange(self, event):
        colNum = self.comboIDs.index(event.GetId())
        self.colNames[colNum] = self.combos[colNum].GetValue()

        self.CheckColNames()


    def GetFieldNames(self):
        return self.colNames


class ImportMatDialog(wx.Dialog):
    def __init__(self, parent, varnames=['Orte']):
        wx.Dialog.__init__(self, parent, title='Import data from matlab file')

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add(wx.StaticText(self, -1, 'Matlab variable name:'), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.cbVarNames = wx.ComboBox(self, -1, choices=varnames)
        self.cbVarNames.SetSelection(0)

        sizer2.Add(self.cbVarNames, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

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
        