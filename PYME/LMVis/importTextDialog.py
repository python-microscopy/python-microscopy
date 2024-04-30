#!/usr/bin/python

##################
# importTextDialog.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import wx
from wx.lib.scrolledpanel import ScrolledPanel

def isnumber(s):
    try:
        float(s)
        return True
    except:
        return False
    
class ColumnMappingDialog(wx.Dialog):
    requiredVariables = {'x':'x position [nm]',
                        'y':'y position [nm]'}
    recommendedVariables = {'A':'amplitude of Gaussian',
                            't':'time [frames]',
                            'sig':'std. deviation of Gaussian [nm]',
                            'error_x':'fit error in x direction [nm]'}
    niceVariables = {'z':'z position [nm]', 'error_y':'fit error in y direction [nm]','error_z':'fit error in z direction [nm]'}
    fileType = 'column source'  # string to indicate file type in user dialog box

    def __init__(self, parent, fileName):
        """
        Dialog box for importing data source with arbitrary column names.
        """

        wx.Dialog.__init__(self, parent, title='Import data from {} file'.format(self.fileType))

        self.colNames, self.dataLines = self._parse_header(fileName)

        sizer1 = wx.BoxSizer(wx.VERTICAL)
        #sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(wx.StaticText(self, -1, 'Please assign variable names to each column. Some variable names must be present for the program to function correctly'), 0, wx.ALL, 5)

        # show any file-type specific UI controls
        self._custom_ui(sizer1)    
        
        self.GenerateDataGrid(sizer1)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.stRequiredNotPresent = wx.StaticText(self, -1, 'Required variables not yet defined:\n')
        sizer2.Add(self.stRequiredNotPresent, 1,  wx.ALL, 5)

        self.stRecommendedNotPresent = wx.StaticText(self, -1, 'Recommended variables not yet defined:\n')
        sizer2.Add(self.stRecommendedNotPresent, 1, wx.ALL, 5)
        sizer1.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rbLocsInNM = wx.RadioButton(self, -1, "x and y positions are in nm", style=wx.RB_GROUP)
        self.rbLocsInNM.SetValue(True)
        sizer1.Add(self.rbLocsInNM, 0, wx.LEFT | wx.EXPAND, 5)
        self.rbLocsInPixels = wx.RadioButton(self, -1, "x and y positions are in pixels")
        sizer1.Add(self.rbLocsInPixels, 0, wx.LEFT | wx.EXPAND, 5)
        
        
        self.stPixelSize = wx.StaticText(self, -1, 'Pixel size [nm]:')
        self.stPixelSize.Disable()
        sizer2.Add(self.stPixelSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.tPixelSize = wx.TextCtrl(self, -1, '1.0')
        self.tPixelSize.Disable()
        sizer2.Add(self.tPixelSize, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        sizer1.Add(sizer2, 0, wx.ALL|wx.EXPAND, 5)

        self.rbLocsInPixels.Bind(wx.EVT_RADIOBUTTON, self.on_toggle_loc_units)
        self.rbLocsInNM.Bind(wx.EVT_RADIOBUTTON, self.on_toggle_loc_units)
        
        btSizer = wx.StdDialogButtonSizer()

        self.btnOK = wx.Button(self, wx.ID_OK)
        self.btnOK.SetDefault()
        self.btnOK.Enable(False)

        btSizer.AddButton(self.btnOK)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        sizer1.Add(btSizer, 0, wx.ALIGN_RIGHT|wx.ALL, 5)

        self.CheckColNames()

        self.SetSizer(sizer1)
        sizer1.Fit(self)
        
    def _custom_ui(self, sizer):
        ''' hook for derived classes to override to add their own UI elements
        
        Last action should be to add themselves to the passed sizer
        '''
        pass
    
    def on_toggle_loc_units(self, e):
        #print('rbLocInPix:' + repr(self.rbLocsInPixels.GetValue()))
        pix = self.rbLocsInPixels.GetValue()
        self.tPixelSize.Enable(pix)
        self.stPixelSize.Enable(pix)
        
        if not pix:
            # reset pixel size to 1
            self.tPixelSize.SetValue('1.0')

    def _parse_header(self, file):
        raise NotImplementedError('Implemented in a derived class.')

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
            id = wx.NewIdRef()
            self.comboIDs.append(id)

            cb = wx.ComboBox(self.scrollW, id, size=(120, -1), choices=[cn]+ list(self.requiredVariables.keys()) +list(self.recommendedVariables.keys()) + list(self.niceVariables.keys()))
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
        
    def GetNumberComments(self):
        return self.numCommentLines
        
    def GetPixelSize(self):
        return float(self.tPixelSize.GetValue())

class ImportTextDialog(ColumnMappingDialog):
    # Text/CSV importer with variable mapping
    fileType='text'

    def _parse_header(self, file):
        import os
        from PYME.IO import csv_flavours
        # TODO - reduce duplication with csv_flavours.guess_text_options

        colNames, dataLines, self.numCommentLines, self.delim = csv_flavours.parse_csv_header(file)
        ext = os.path.splitext(file)[-1]
        self.text_flavour = csv_flavours.guess_flavour(colNames, self.delim, ext)

        self.raw_names = colNames
        colNames = csv_flavours.replace_names(colNames, self.text_flavour)

        return colNames, dataLines

    def _custom_ui(self, sizer):
        # TODO - move this out of base class
        from PYME.IO import csv_flavours

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Text file flavour:'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        flavour_choices = list(csv_flavours.csv_flavours.keys())
        self.chFlavour = wx.Choice(self, -1, choices = flavour_choices)
        self.chFlavour.SetSelection(flavour_choices.index(self.text_flavour))
        self.chFlavour.Bind(wx.EVT_CHOICE, self.OnFlavourChange)
        hsizer.Add(self.chFlavour, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 2)

        sizer.Add(hsizer, 0, wx.ALL|wx.EXPAND, 5)
    
    def GetDelim(self):
        return self.delim

    def GetFlavour(self):
        return self.text_flavour

    def change_text_flavour(self, flavour):
        from PYME.IO import csv_flavours
        self.text_flavour = flavour
        self.colNames = csv_flavours.replace_names(self.raw_names, flavour)

    def OnFlavourChange(self, event):
        self.change_text_flavour(self.chFlavour.GetStringSelection())
        self.CheckColNames()

        # Update the GUI selection
        for ci, cn in enumerate(self.colNames):
            self.combos[ci].SetValue(cn)

class ImportMatDialog(wx.Dialog):
    # Old-style MATLAB importer
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
        
class ImportMatlabDialog(ColumnMappingDialog):
    # MATLAB importer with variable mapping
    fileType = 'MATLAB'
    multichannel = False  # Is out matlab source multichannel?

    def _parse_header(self, file):
        import numpy as np

        if isinstance(file, dict):
            # We've passed the loaded file (scipy.io.loadmat returns a dict)
            mf = file
        else:
            from scipy.io import loadmat
            mf = loadmat(file)

        self.numCommentLines = 0

        colNames = [k for k in mf.keys() if not k.startswith('_')]

        dataLines = []
        for k in colNames:
            if mf[k].shape[1] > mf[k].shape[0]:
                # Multicolor
                self.multichannel = True
                dataLines.append(mf[k][0,0][:10].squeeze())
            else:
                dataLines.append(mf[k][:10].squeeze())
        dataLines = np.array(dataLines).T.astype(str).tolist()

        return colNames, dataLines

    def GetMultichannel(self):
        return self.multichannel
