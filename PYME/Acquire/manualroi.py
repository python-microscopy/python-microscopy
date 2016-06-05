#!/usr/bin/python

# set the ROI using console instead of hotkey F8
def Manualroi(MainFrame, x1, y1, x2, y2):
    MainFrame.scope.pa.stop()
    MainFrame.scope.cam.SetROI(x1,y1,x2,y2)
    MainFrame.scope.cam.SetCOC()
    MainFrame.scope.cam.GetStatus()
    MainFrame.scope.pa.Prepare()
    MainFrame.scope.vp.SetDataStack(MainFrame.scope.pa.dsa)
    MainFrame.scope.vp.do.SetSelection((x1,y1,0), (x2,y2,0))
    MainFrame.scope.pa.start()
    MainFrame.scope.vp.Refresh()
    MainFrame.scope.vp.GetParent().Refresh()
    MainFrame.roi_on = True

# example: manualroi.Manualroi(MainFrame, 1024-49,1024-49,1024+50,1024+50)