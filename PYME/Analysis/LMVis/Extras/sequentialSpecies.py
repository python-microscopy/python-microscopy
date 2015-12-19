import wx
import numpy as np

class SpeciesDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(wx.StaticText(self, -1, 'Species list: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        self.tSpecString = wx.TextCtrl(self, -1, 'spec1,0,1e4,spec2,1e4,3e4', size=[200,-1])
        hsizer.Add(self.tSpecString, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL, 5)

        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        btSizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()

        btSizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)

        btSizer.AddButton(btn)

        btSizer.Realize()

        #hsizer.Add(btSizer,0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        vsizer.Add(btSizer, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.SetSizerAndFit(vsizer)
    
    def GetSpeciesDescriptor(self):
        specstring =  self.tSpecString.GetValue()
        rawlist = specstring.split(',')
        # currently we assume a simple comma-seperated list of the form
        # specname, t_start, t_end, ...
        speclist = []
        for i in range(len(rawlist)/3):
            speclist.append({'name'   : rawlist[3*i].strip(),
                             't_start': float(rawlist[3*i+1]),
                             't_end'  : float(rawlist[3*i+2])})
        return speclist

from PYME.Analysis.LMVis import renderers

class TimedSpecies:
    def __init__(self, visFr):
        self.visFr = visFr
        self.timedSpecies = None

        ID_TIMED_SPECIES = wx.NewId()
        visFr.extras_menu.Append(ID_TIMED_SPECIES, "&Sequential Imaging - Species assignment")
        visFr.Bind(wx.EVT_MENU, self.OnTimedSpecies, id=ID_TIMED_SPECIES)
        renderers.renderMetadataProviders.append(self.SaveMetadata)

    def OnTimedSpecies(self,event):
        dlg = SpeciesDialog(None)
        ret = dlg.ShowModal()

        speclist = dlg.GetSpeciesDescriptor()
        #print 'Species Descriptor: ', speclist
        if ret == wx.ID_OK:
            pipeline = self.visFr.pipeline
            if pipeline.selectedDataSource is not None:
                self.SetTimedSpecies(speclist)
                pipeline.selectedDataSource.setMapping('ColourNorm', '1.0 + 0*t')
                for species in self.timedSpecies.keys():
                    pipeline.selectedDataSource.setMapping('p_%s' % species,
                                                           '(t>= %d)*(t<%d)' % self.timedSpecies[species])
                self.visFr.RegenFilter()
                self.visFr.CreateFoldPanel()

    def SaveMetadata(self,mdh):
        mdh['TimedSpecies'] = self.timedSpecies

    def SetTimedSpecies(self,slist):
        self.timedSpecies = {}
        for spec in slist:
            self.timedSpecies[spec['name']] = (spec['t_start'],spec['t_end'])

def Plug(visFr):
    '''Plugs this module into the gui'''
    TimedSpecies(visFr)

