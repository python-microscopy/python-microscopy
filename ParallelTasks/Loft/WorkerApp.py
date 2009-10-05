#!/usr/bin/python

##################
# WorkerApp.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#!/usr/bin/env python
#Boa:App:BoaApp

import wx

import WorkerFrame

modules ={u'WorkerFrame': [1, 'Main frame of Application', u'WorkerFrame.py']}

class BoaApp(wx.App):
    def OnInit(self):
        self.main = WorkerFrame.create(None)
        self.main.Show()
        self.SetTopWindow(self.main)
        return True

def main():
    application = BoaApp(0)
    application.MainLoop()

if __name__ == '__main__':
    main()
