import wx
import numpy as np

class SpeciesDialog(wx.Dialog):
    def __init__(self, *args, **kwargs):
        wx.Dialog.__init__(self, *args, **kwargs)
        vsizer = wx.BoxSizer(wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Species 1: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS1 = [wx.TextCtrl(self, -1, 'name1', size=[50,-1])]
        hsizer.Add(self.tSpecStringS1[0], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'from'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS1.append(wx.TextCtrl(self, -1, '0', size=[30,-1]))
        hsizer.Add(self.tSpecStringS1[1], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'to'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS1.append(wx.TextCtrl(self, -1, '1e4', size=[30,-1]))
        hsizer.Add(self.tSpecStringS1[2], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Species 2: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS2 = [wx.TextCtrl(self, -1, 'name2', size=[50,-1])]
        hsizer.Add(self.tSpecStringS2[0], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'from'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS2.append(wx.TextCtrl(self, -1, '1e4', size=[30,-1]))
        hsizer.Add(self.tSpecStringS2[1], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'to'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS2.append(wx.TextCtrl(self, -1, '2e4', size=[30,-1]))
        hsizer.Add(self.tSpecStringS2[2], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self, -1, 'Species 3: '), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS3 = [wx.TextCtrl(self, -1, '', size=[50,-1])]
        hsizer.Add(self.tSpecStringS3[0], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'from'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS3.append(wx.TextCtrl(self, -1, '', size=[30,-1]))
        hsizer.Add(self.tSpecStringS3[1], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        hsizer.Add(wx.StaticText(self, -1, 'to'), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        self.tSpecStringS3.append(wx.TextCtrl(self, -1, '', size=[30,-1]))
        hsizer.Add(self.tSpecStringS3[2], 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        vsizer.Add(hsizer, 0, wx.ALL, 5)

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
        speclist = []
        for specstrings in [self.tSpecStringS1, self.tSpecStringS2, self.tSpecStringS3]:
            sname =  specstrings[0].GetValue().strip()
            if sname: # empty strings will be ignored
                speclist.append({'name'   : sname,
                                 't_start': float(specstrings[1].GetValue()),
                                 't_end'  : float(specstrings[2].GetValue())})
        print(speclist)
        return speclist

from PYME.LMVis import renderers

class TimedSpecies:
    def __init__(self, visFr):
        self.visFr = visFr
        self.timedSpecies = None

        visFr.AddMenuItem('Extras', "&Sequential Imaging - Species assignment",self.OnTimedSpecies)
        # this gives every image a TimedSpecies entry (None by default)
        # this is probably not a good idea
        # renderers.renderMetadataProviders.append(self.SaveMetadata)

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
                if len(self.timedSpecies.keys()) > 0:
                    pipeline.mdh.setEntry('TimedSpecies', speclist)
                    self.visFr.RegenFilter()
                    #self.visFr.CreateFoldPanel() #TODO - do we need this?

    def SaveMetadata(self,mdh):
        mdh['TimedSpecies'] = self.timedSpecies

    def SetTimedSpecies(self,slist):
        self.timedSpecies = {}
        for spec in slist:
            self.timedSpecies[spec['name']] = (spec['t_start'],spec['t_end'])

def Plug(visFr):
    '''Plugs this module into the gui'''
    TimedSpecies(visFr)

