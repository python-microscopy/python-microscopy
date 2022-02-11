#!/usr/bin/python

##################
# AndorControlFrame.py
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

#Boa:Frame:AndorFrame



import wx

class ucCamPanel(wx.Panel):

    def _init_ctrls(self, prnt):

        wx.Panel.__init__(self, parent=prnt)

        vsizer=wx.BoxSizer(wx.VERTICAL)

        ucGain = wx.StaticBoxSizer(wx.StaticBox(self, -1, u'Gain (%)'), wx.VERTICAL)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.l = wx.StaticText(self, -1, '%3.1f'%(100.0*self.cam.GetGain()/100)+'%') #ideally this should be self.cam.GetGain()/self.cam.MaxGain, where MaxGain is set to 100 for the specific camera.
        hsizer.Add(self.l, 0, wx.ALL, 2)
        self.sl = wx.Slider(self, -1, 100.0*self.cam.GetGain()/100, 0, 100, size=wx.Size(150,-1),style=wx.SL_HORIZONTAL | wx.SL_HORIZONTAL | wx.SL_AUTOTICKS )
        self.sl.SetTickFreq(10)
        self.Bind(wx.EVT_SCROLL,self.onSlide)
        hsizer.Add(self.sl, 1, wx.ALL|wx.EXPAND, 2)
        ucGain.Add(hsizer, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.stGainFactor = wx.StaticText(self, -1, 'Gain Factor = %3.2f' % self.cam.GetGainFactor())
        ucGain.Add(self.stGainFactor, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.stepc = wx.StaticText(self, -1, 'Electrons/Count = %3.2f' % (self.cam.noise_properties['ElectronsPerCount']))
        ucGain.Add(self.stepc, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)


        vsizer.Add(ucGain, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.SetSizer(vsizer)

    def __init__(self, parent, cam, scope):
        self.cam = cam
        self.scope = scope
        self._init_ctrls(parent)
        self.sliding = False

    def onSlide(self, event):
        self.sliding = True
        try: 
            sl = event.GetEventObject()
            self.cam.SetGain(int(round(sl.GetValue()/100.0*100)))
            self.l.SetLabel('%3.1f'%(100.0*self.cam.GetGain()/100)+'%')
            self.stGainFactor.SetLabel('Gain Factor = %3.2f' % self.cam.GetGainFactor())
            self.stepc.SetLabel('Electrons/Count = %3.2f' % self.cam.noise_properties['ElectronsPerCount'])
        finally:
            self.sliding = False

    def update(self): # only needed if we want automatic update for the panel. 
        if not self.sliding:
            self.sl.SetValue(round(self.cam.GetGain()/100))
            self.l.SetLabel(str(self.cam.GetGain()/100)+'%%')
            self.stGainFactor.SetLabel('Gain Factor = %3.2f' % self.cam.GetGainFactor())
            self.stepc.SetLabel('Electrons/Count = %3.2f' % self.cam.noise_properties['ElectronsPerCount'])

