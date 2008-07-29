#!/usr/bin/env python
#Boa:App:BoaApp

import wx

import smimainframe


modules ={'chanedit': [0, '', 'chanedit.py'],
 'chaneditpan': [0, '', 'chaneditpan.py'],
 'chanfr': [0, '', 'chanfr.py'],
 'chanpanel': [0, '', 'chanpanel.py'],
 'dsviewer': [0, '', 'dsviewer.py'],
 'funcs': [0, '', 'funcs.py'],
 'init': [0, '', 'init.py'],
 'intsliders': [0, '', 'intsliders.py'],
 'leica_init': [0, '', 'leica_init.py'],
 'livepreview': [0, '', 'livepreview.py'],
 'mytimer': [0, '', 'mytimer.py'],
 'myviewpanel': [0, '', 'myviewpanel.py'],
 'noclosefr': [0, '', 'noclosefr.py'],
 'piezo_e662': [0, '', 'piezo_e662.py'],
 'piezo_e816': [0, '', 'piezo_e816.py'],
 'previewaquisator': [0, '', 'previewaquisator.py'],
 'prevtimer': [0, '', 'prevtimer.py'],
 'prevviewer': [0, '', 'prevviewer.py'],
 'psliders': [0, '', 'psliders.py'],
 'seqdialog': [0, '', 'seqdialog.py'],
 'seqpanel': [0, '', 'seqpanel.py'],
 'simplesequenceaquisator': [0, '', 'simplesequenceaquisator.py'],
 'smi_init': [0, '', 'smi_init.py'],
 'smimainframe': [1, 'Main frame of Application', 'smimainframe.py'],
 'sqpan': [0, '', 'sqpan.py'],
 'stepDialog': [0, '', 'stepDialog.py'],
 'tirf_init': [0, '', 'tirf_init.py'],
 'vf2': [0, '', 'vf2.py'],
 'vfr': [0, '', 'vfr.py'],
 'vframe': [0, '', 'vframe.py'],
 'viewframe': [0, '', 'viewframe.py'],
 'viewpanel': [0, '', 'viewpanel.py']}

class BoaApp(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        self.main = smimainframe.create(None)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True

def main():
    application = BoaApp(0)
    application.MainLoop()

if __name__ == '__main__':
    main()
