import win32com.client
from wx.lib import activexwrapper
import wx

PzWrapper = activexwrapper.MakeActiveXClass(
    win32com.client.gencache.GetClassForProgID('MGPIEZO.MGPiezoCtrl.1'),
    eventClass=None, eventObj=None)

SysWrapper = activexwrapper.MakeActiveXClass(
    win32com.client.gencache.GetClassForProgID('MG17SYSTEM.MG17SystemCtrl.1'),
    eventClass=None, eventObj=None)

def EnumeratePiezos():
    '''Return serial numbers of all attached piezos'''

    f = wx.Frame(None)
    sy = SysWrapper(f)
    sy.StartCtrl()

    numPiezos = sy.GetNumHWUnits(win32com.client.constants.USB_PIEZO_DRIVE,1)[1]

    serialNums = []

    for i in range(numPiezos):
        serialNums.append(sy.GetHWSerialNum(win32com.client.constants.USB_PIEZO_DRIVE, i, 1)[1])

    f.Close()

    return serialNums
                          
class PiezoFrame(wx.Frame):
    def __init__(self, serialNumber, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.PiezoControl = PzWrapper(self, size=self.GetClientSize())

        self.PiezoControl.HWSerialNum = serialNumber
        self.PiezoControl.StartCtrl()
        #self.show()


class TLPiezo:
    def __init__(self, serialNumber, name=''):
        self.pzFrame = PiezoFrame(serialNumber, parent=None, size=(400,300), title=name)
        self.max_travel = self.pzFrame.PiezoControl.GetMaxTravel(0,1)[1]

        #set closed loop mode
        self.pzFrame.PiezoControl.SetControlMode(0,win32com.client.constants.CLOSED_LOOP)

    def MoveTo(self, iChannel, fPos, bTimeOut=True):
        if (fPos >= 0):
            if (fPos <= self.max_travel):
                self.pzFrame.PiezoControl.SetPosOutput(0,fPos)
            else:
                self.pzFrame.PiezoControl.SetPosOutput(0,self.max_travel)
        else:
            self.pzFrame.PiezoControl.SetPosOutput(0,0.0)

    def GetPos(self, iChannel=1):
        return self.pzFrame.PiezoControl.GetPosOutput(0,1)[1]

    def GetControlReady(self):
        return True

    def GetChannelObject(self):
        return 1

    def GetChannelPhase(self):
        return 1

    def GetMin(self,iChan=1):
        return 0

    def GetMax(self, iChan=1):
        return self.max_travel
