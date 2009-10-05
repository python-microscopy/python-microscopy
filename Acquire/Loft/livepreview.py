#!/usr/bin/python

##################
# livepreview.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

#major point to note is that aquisition can be subsequently stopped / started using
# pa.stop() & pa.start()    NB LOWERCASE!!!

#to alter the integration time type the following (where xxx is the integration time in ms)
#
# cam.SetIntegTime(xxx) 
# cam.SetCOC()

# to see what the integration time is ...
#
# cam.GetIntegTime()
def pr_refr(source):
    prev_fr.update()

def genStatus():
    cam.GetStatus()
    stext = 'Integration time: %dms   CCD Temp: %d   Electro Temp: %d' % (cam.GetIntegTime(), cam.GetCCDTemp(), cam.GetElectroTemp)
    return stext

pa = previewaquisator.PreviewAquisator(chaninfo,cam)

pa.Prepare()

prev_fr = prevviewer.PrevViewFrame(None, "Live Preview", pa.ds)

pa.WantFrameNotification.append(pr_refr)

prev_fr.genStatusText = genStatus

prev_fr.Show()

pa.start()