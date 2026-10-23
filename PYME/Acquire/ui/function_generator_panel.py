#!/usr/bin/python

##################
# function_generator_panel.py
#
# GUI panel for function / arbitrary waveform generators which register
# 'FunctionGenerators.<name>.*' scope state handlers (e.g. PYME.Acquire.Hardware.Siglent.SiglentSDG)
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
import re

# (state key suffix, label) for each numeric parameter shown in the panel
PARAMETERS = [
    ('Frequency', 'Frequency (Hz)'),
    ('High', 'High (V)'),
    ('Low', 'Low (V)'),
    ('DutyCycle', 'Duty (%)'),
    ('Symmetry', 'Symmetry (%)'),
    ('Offset', 'Offset (V)'),
]


class FunctionGeneratorPanel(wx.Panel):
    def __init__(self, parent, scopeState, winid=-1):
        wx.Panel.__init__(self, parent, winid)
        self.scopeState = scopeState

        # discover our function generators
        self.generatorNames = []
        for k in self.scopeState.keys():
            m = re.match(r'FunctionGenerators\.(?P<name>.*)\.Waveform$', k)
            if m is not None:
                self.generatorNames.append(m.group('name'))

        self.generatorNames.sort()

        self.onButtons = {}
        self.waveformChoices = {}
        self.applyButtons = {}
        self.textCtrls = {}
        self.lastValues = {}

        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        for name in self.generatorNames:
            key = 'FunctionGenerators.%s.' % name
            box = wx.StaticBoxSizer(wx.StaticBox(self, -1, name), wx.VERTICAL)

            # output on/off and waveform selection
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            b = wx.ToggleButton(self, -1, 'Output', style=wx.BU_EXACTFIT)
            b.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle)
            self.onButtons[name] = b
            hsizer.Add(b, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)

            ch = wx.Choice(self, -1, choices=self.scopeState[key + 'AvailableWaveforms'])
            ch.Bind(wx.EVT_CHOICE, self.on_waveform)
            self.waveformChoices[name] = ch
            hsizer.Add(ch, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
            box.Add(hsizer, 0, wx.EXPAND, 0)

            # numeric parameters
            grid = wx.FlexGridSizer(0, 2, 2, 4)
            grid.AddGrowableCol(1)
            self.textCtrls[name] = {}
            self.lastValues[name] = {}
            for param, label in PARAMETERS:
                grid.Add(wx.StaticText(self, -1, label), 0, wx.ALIGN_CENTER_VERTICAL)
                tc = wx.TextCtrl(self, -1, '', size=wx.Size(80, -1))
                self.textCtrls[name][param] = tc
                grid.Add(tc, 0, wx.EXPAND)
            box.Add(grid, 0, wx.EXPAND | wx.ALL, 2)

            ab = wx.Button(self, -1, 'Apply')
            ab.Bind(wx.EVT_BUTTON, self.on_apply)
            self.applyButtons[name] = ab
            box.Add(ab, 0, wx.ALL | wx.ALIGN_RIGHT, 2)

            sizer_1.Add(box, 0, wx.EXPAND | wx.ALL, 2)

        self.SetSizerAndFit(sizer_1)
        self.update()

    def _name_for(self, control, controls):
        for name, c in controls.items():
            if c is control:
                return name

    def on_toggle(self, event):
        b = event.GetEventObject()
        name = self._name_for(b, self.onButtons)
        self.scopeState['FunctionGenerators.%s.On' % name] = b.GetValue()

    def on_waveform(self, event):
        ch = event.GetEventObject()
        name = self._name_for(ch, self.waveformChoices)
        self.scopeState['FunctionGenerators.%s.Waveform' % name] = ch.GetStringSelection()
        self.update()

    def on_apply(self, event):
        name = self._name_for(event.GetEventObject(), self.applyButtons)
        key = 'FunctionGenerators.%s.' % name

        active = self.scopeState[key + 'ActiveParameters']
        try:
            values = {param: float(self.textCtrls[name][param].GetValue()) for param in active}
        except ValueError:
            wx.MessageBox('Please enter numbers only', 'Invalid value', wx.OK | wx.ICON_ERROR)
            return

        order = [param for param, label in PARAMETERS if param in values and param not in ('High', 'Low')]
        if 'High' in values and 'Low' in values:
            # set high and low in an order that never makes high < low part way through
            if values['High'] > self.scopeState[key + 'Low']:
                order += ['High', 'Low']
            else:
                order += ['Low', 'High']

        for param in order:
            self.scopeState[key + param] = values[param]

        self.update()

    def update(self, **kwargs):
        for name in self.generatorNames:
            key = 'FunctionGenerators.%s.' % name

            on = self.scopeState[key + 'On']
            self.onButtons[name].SetValue(on)
            if on:
                self.onButtons[name].SetBackgroundColour("red")
            else:
                self.onButtons[name].SetBackgroundColour(wx.NullColour)

            waveform = self.scopeState[key + 'Waveform']
            ch = self.waveformChoices[name]
            if ch.GetStringSelection() != waveform:
                ch.SetStringSelection(waveform)

            active = self.scopeState[key + 'ActiveParameters']
            for param, label in PARAMETERS:
                tc = self.textCtrls[name][param]
                tc.Enable(param in active)

                # only overwrite the text box when the value has changed, so we don't clobber typing in progress
                value = self.scopeState[key + param]
                if value != self.lastValues[name].get(param):
                    tc.SetValue('%g' % value)
                    self.lastValues[name][param] = value
