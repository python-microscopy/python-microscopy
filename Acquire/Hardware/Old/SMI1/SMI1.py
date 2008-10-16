# This file was created automatically by SWIG 1.3.29.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _SMI1
import new
new_instancemethod = new.instancemethod
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


class CSerialOp(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CSerialOp, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CSerialOp, name)
    __repr__ = _swig_repr
    def GetControllerName(*args): return _SMI1.CSerialOp_GetControllerName(*args)
    def __init__(self, *args): 
        this = _SMI1.new_CSerialOp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _SMI1.delete_CSerialOp
    __del__ = lambda self : None;
    def SetMessage(*args): return _SMI1.CSerialOp_SetMessage(*args)
    def GetMessageKey(*args): return _SMI1.CSerialOp_GetMessageKey(*args)
    def SetPortNr(*args): return _SMI1.CSerialOp_SetPortNr(*args)
    def GetPortNr(*args): return _SMI1.CSerialOp_GetPortNr(*args)
    def GetPortName(*args): return _SMI1.CSerialOp_GetPortName(*args)
    def GetControlReady(*args): return _SMI1.CSerialOp_GetControlReady(*args)
    def SetControlReady(*args): return _SMI1.CSerialOp_SetControlReady(*args)
    def DisplayError(*args): return _SMI1.CSerialOp_DisplayError(*args)
    def CloseConnection(*args): return _SMI1.CSerialOp_CloseConnection(*args)
CSerialOp_swigregister = _SMI1.CSerialOp_swigregister
CSerialOp_swigregister(CSerialOp)

class CPiezoOp(CSerialOp):
    __swig_setmethods__ = {}
    for _s in [CSerialOp]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, CPiezoOp, name, value)
    __swig_getmethods__ = {}
    for _s in [CSerialOp]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, CPiezoOp, name)
    __repr__ = _swig_repr
    def GetRangeError(*args): return _SMI1.CPiezoOp_GetRangeError(*args)
    def GetHardRange(*args): return _SMI1.CPiezoOp_GetHardRange(*args)
    def GetFirmware(*args): return _SMI1.CPiezoOp_GetFirmware(*args)
    def VoltToMikro(*args): return _SMI1.CPiezoOp_VoltToMikro(*args)
    def MikroToVolt(*args): return _SMI1.CPiezoOp_MikroToVolt(*args)
    def Calibrate(*args): return _SMI1.CPiezoOp_Calibrate(*args)
    def __init__(self, *args): 
        this = _SMI1.new_CPiezoOp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _SMI1.delete_CPiezoOp
    __del__ = lambda self : None;
    def GetCtrlStatus(*args): return _SMI1.CPiezoOp_GetCtrlStatus(*args)
    def GetPos(*args): return _SMI1.CPiezoOp_GetPos(*args)
    def SetMin(*args): return _SMI1.CPiezoOp_SetMin(*args)
    def GetMin(*args): return _SMI1.CPiezoOp_GetMin(*args)
    def SetMax(*args): return _SMI1.CPiezoOp_SetMax(*args)
    def GetMax(*args): return _SMI1.CPiezoOp_GetMax(*args)
    def SetChannelObject(*args): return _SMI1.CPiezoOp_SetChannelObject(*args)
    def GetChannelObject(*args): return _SMI1.CPiezoOp_GetChannelObject(*args)
    def GetChannelPhase(*args): return _SMI1.CPiezoOp_GetChannelPhase(*args)
    def GetHardMin(*args): return _SMI1.CPiezoOp_GetHardMin(*args)
    def GetHardMax(*args): return _SMI1.CPiezoOp_GetHardMax(*args)
    def Init(*args): return _SMI1.CPiezoOp_Init(*args)
    def ContIO(*args): return _SMI1.CPiezoOp_ContIO(*args)
    def MoveTo(*args): return _SMI1.CPiezoOp_MoveTo(*args)
    def GetOnePosition(*args): return _SMI1.CPiezoOp_GetOnePosition(*args)
    def SetExtCtrlOnOff(*args): return _SMI1.CPiezoOp_SetExtCtrlOnOff(*args)
    def SetAllToNull(*args): return _SMI1.CPiezoOp_SetAllToNull(*args)
CPiezoOp_swigregister = _SMI1.CPiezoOp_swigregister
CPiezoOp_swigregister(CPiezoOp)

class CStepOp(CSerialOp):
    __swig_setmethods__ = {}
    for _s in [CSerialOp]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, CStepOp, name, value)
    __swig_getmethods__ = {}
    for _s in [CSerialOp]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, CStepOp, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _SMI1.new_CStepOp(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _SMI1.delete_CStepOp
    __del__ = lambda self : None;
    def GetCalStatus(*args): return _SMI1.CStepOp_GetCalStatus(*args)
    def SetZStatus(*args): return _SMI1.CStepOp_SetZStatus(*args)
    def GetZStatus(*args): return _SMI1.CStepOp_GetZStatus(*args)
    def GetCalibrationOK(*args): return _SMI1.CStepOp_GetCalibrationOK(*args)
    def GetOSStatus(*args): return _SMI1.CStepOp_GetOSStatus(*args)
    def GetVersion(*args): return _SMI1.CStepOp_GetVersion(*args)
    def GetPosX(*args): return _SMI1.CStepOp_GetPosX(*args)
    def GetPosY(*args): return _SMI1.CStepOp_GetPosY(*args)
    def GetPosZ(*args): return _SMI1.CStepOp_GetPosZ(*args)
    def GetStrX(*args): return _SMI1.CStepOp_GetStrX(*args)
    def GetStrY(*args): return _SMI1.CStepOp_GetStrY(*args)
    def GetStrZ(*args): return _SMI1.CStepOp_GetStrZ(*args)
    def GetMinX(*args): return _SMI1.CStepOp_GetMinX(*args)
    def GetMaxX(*args): return _SMI1.CStepOp_GetMaxX(*args)
    def GetMinY(*args): return _SMI1.CStepOp_GetMinY(*args)
    def GetMaxY(*args): return _SMI1.CStepOp_GetMaxY(*args)
    def GetMinZ(*args): return _SMI1.CStepOp_GetMinZ(*args)
    def GetMaxZ(*args): return _SMI1.CStepOp_GetMaxZ(*args)
    def GetStrMinX(*args): return _SMI1.CStepOp_GetStrMinX(*args)
    def GetStrMaxX(*args): return _SMI1.CStepOp_GetStrMaxX(*args)
    def GetStrMinY(*args): return _SMI1.CStepOp_GetStrMinY(*args)
    def GetStrMaxY(*args): return _SMI1.CStepOp_GetStrMaxY(*args)
    def GetStrMinZ(*args): return _SMI1.CStepOp_GetStrMinZ(*args)
    def GetStrMaxZ(*args): return _SMI1.CStepOp_GetStrMaxZ(*args)
    def GetMoveAccel(*args): return _SMI1.CStepOp_GetMoveAccel(*args)
    def GetMoveSpeed(*args): return _SMI1.CStepOp_GetMoveSpeed(*args)
    def GetJoystickStatus(*args): return _SMI1.CStepOp_GetJoystickStatus(*args)
    def GetJoystickSpeed(*args): return _SMI1.CStepOp_GetJoystickSpeed(*args)
    def GetNullX(*args): return _SMI1.CStepOp_GetNullX(*args)
    def GetNullY(*args): return _SMI1.CStepOp_GetNullY(*args)
    def GetNullZ(*args): return _SMI1.CStepOp_GetNullZ(*args)
    def Init(*args): return _SMI1.CStepOp_Init(*args)
    def Calibrate(*args): return _SMI1.CStepOp_Calibrate(*args)
    def InitAreaValues(*args): return _SMI1.CStepOp_InitAreaValues(*args)
    def SetArea(*args): return _SMI1.CStepOp_SetArea(*args)
    def Break(*args): return _SMI1.CStepOp_Break(*args)
    def StartContIO(*args): return _SMI1.CStepOp_StartContIO(*args)
    def ContIO(*args): return _SMI1.CStepOp_ContIO(*args)
    def StopContIO(*args): return _SMI1.CStepOp_StopContIO(*args)
    def MoveTo(*args): return _SMI1.CStepOp_MoveTo(*args)
    def MoveRel(*args): return _SMI1.CStepOp_MoveRel(*args)
    def MoveOSInOut(*args): return _SMI1.CStepOp_MoveOSInOut(*args)
    def SetMoveAccel(*args): return _SMI1.CStepOp_SetMoveAccel(*args)
    def SetMoveSpeed(*args): return _SMI1.CStepOp_SetMoveSpeed(*args)
    def SetJoystickOnOff(*args): return _SMI1.CStepOp_SetJoystickOnOff(*args)
    def SetJoystickSpeed(*args): return _SMI1.CStepOp_SetJoystickSpeed(*args)
    def SetNull(*args): return _SMI1.CStepOp_SetNull(*args)
    def SavePos(*args): return _SMI1.CStepOp_SavePos(*args)
    def MoveToPos(*args): return _SMI1.CStepOp_MoveToPos(*args)
CStepOp_swigregister = _SMI1.CStepOp_swigregister
CStepOp_swigregister(CStepOp)



