#!/usr/bin/python

###############
# Mercury.py
#
# Copyright David Baddeley, 2012
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
################
from . import PI_Mercury_GCS_DLL as mc
import ctypes

#class MercuryWrap:

class fcnWrap:
    def __init__(self, fcn):
        self.fcn = fcn
        self.___doc__ = fcn.__doc__

    def HandleError(self, ID):
        errno = mc.GetError(ID)
        raise RuntimeError('Mercury controller error: %d - %s' %(errno, TranslateError(errno)))

class AxesSetter(fcnWrap):
    def __call__(self, ID, szAxes, *args):
        in_args = []

        #build input arrays
        for argtype, argname, arg  in zip(self.fcn.argtypes[2:], self.fcn.argnames[2:], args):
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]

                in_args.append((basetype*len(arg))(*arg))
            else:
                in_args.append(argtype(arg))

        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in in_args]))
        succ = self.fcn(ID, szAxes, *in_args)
        if not succ:
            self.HandleError(ID)

class ArraySetter(fcnWrap):
    def __call__(self, ID, *args):
        in_args = []

        #build input arrays
        for argtype, argname, arg  in zip(self.fcn.argtypes[1:-1], self.fcn.argnames[1:-1], args):
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]

                in_args.append((basetype*len(arg))(*arg))
            else:
                in_args.append(argtype(arg))

        in_args.append(len(arg))

        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in in_args]))
        succ = self.fcn(ID, *in_args)
        if not succ:
            self.HandleError(ID)

class ArrayGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        out_args = []

        AxIDs = [1 for id in szAxes]  # have to be all 1 for mercury
        #build input arrays
        for argtype in self.fcn.argtypes[3:-1]:
            if argtype.__name__.startswith('LP_'): #array input
                basetype = ctypes.__dict__[argtype.__name__[3:]]

                out_args.append((basetype*len(szAxes))())
            else:
                out_args.append(argtype())

        out_args.append(len(szAxes))

        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(oa)for oa in out_args]))
        succ = self.fcn(ID, szAxes, AxIDs, *out_args)
        if not succ:
            self.HandleError(ID)

        if len(out_args) == 1:
            out_args = out_args[0]

        return out_args


class AxesGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        out_args = []
        #ret_args = []

        #build output arrays
        for argtype in self.fcn.argtypes[2:]:
            if argtype.__name__.startswith('LP_'): #array output
                basetype = ctypes.__dict__[argtype.__name__[3:]]

                out_args.append((basetype*len(szAxes))())
                #ret_args.append(out_args[-1])
            else:
                out_args.append(argtype())


        #print "%s(%s, '%s', '%s')" % (self.fcn.__name__, ID, szAxes, ','.join([repr(ia)for ia in out_args]))
        succ = self.fcn(ID, szAxes, *out_args)
        if not succ:
            self.HandleError(ID)
        
        if len(out_args) == 1:
            out_args = out_args[0]

        return out_args

class StringGetter(fcnWrap):
    def __call__(self, ID):
        buff = ctypes.create_string_buffer(500)
        succ = self.fcn(ID, buff, len(buff))
        if not succ:
            self.HandleError(ID)

        return buff.value.decode()

class AxesStringGetter(fcnWrap):
    def __call__(self, ID, szAxes):
        buff = ctypes.create_string_buffer(500)
        succ = self.fcn(ID, szAxes, buff, len(buff))
        if not succ:
            self.HandleError(ID)

        return buff.value.decode()

class ValGetter(fcnWrap):
    def __call__(self, ID):
        argtype = self.fcn.argtypes[-1]
        if argtype.__name__.startswith('LP_'): #array output
            basetype = ctypes.__dict__[argtype.__name__[3:]]
            buff = basetype()
            succ = self.fcn(ID, ctypes.byref(buff))
            if not succ:
                self.HandleError(ID)

            return buff.value

class NotImplemented(fcnWrap):
    def __call__(self, ID, *args):
        raise RuntimeError('Function not wrapped')
  
        
ConnectRS232  = mc.ConnectRS232
IsConnected = mc.IsConnected
CloseConnection = mc.CloseConnection
GetStatus = NotImplemented(mc.GetStatus)

IsReferencing = AxesGetter(mc.IsReferencing)
IsReferenceOK = AxesGetter(mc.IsReferenceOK)
GetRefResult = AxesGetter(mc.GetRefResult)
IsMoving = AxesGetter(mc.IsMoving)
IsRecordingMacro = ValGetter(mc.IsRecordingMacro)
IsRunningMacro = ValGetter(mc.IsRunningMacro)

GetOutputChannelNames = StringGetter(mc.GetOutputChannelNames)
GetInputChannelNames = StringGetter(mc.GetInputChannelNames)

SetErrorCheck = mc.SetErrorCheck
GetError = mc.GetError
TranslateError = StringGetter(mc.TranslateError)

GcsGetAnswerSize = NotImplemented(mc.GcsGetAnswerSize)
GcsGetAnswer = NotImplemented(mc.GcsGetAnswer)
GcsCommandset = NotImplemented(mc.GcsCommandset)
ReceiveNonGCSString = NotImplemented(mc.ReceiveNonGCSString)
SendNonGCSString = NotImplemented(mc.SendNonGCSString)

OpenUserStagesEditDialog = NotImplemented(mc.OpenUserStagesEditDialog)
OpenPiStagesEditDialog = NotImplemented(mc.OpenPiStagesEditDialog)
InterfaceSetupDlg = NotImplemented(mc.InterfaceSetupDlg)

AddStage = mc.AddStage
RemoveStage = mc.RemoveStage

GetAsyncBufferIndex = NotImplemented(mc.GetAsyncBufferIndex)

BRA = AxesSetter(mc.BRA)
#CLR = mc.CLR
CST = mc.CST
CTO = ArraySetter(mc.CTO)
DEL = mc.DEL
DFF = AxesSetter(mc.DFF)
DFH = mc.DFH
DIO = AxesSetter(mc.DIO)
GOH = mc.GOH
HLT = mc.HLT
INI = mc.INI
JDT = ArraySetter(mc.JDT)
JLT = ArraySetter(mc.JLT)
JON = ArraySetter(mc.JON)
MAC_BEG = mc.MAC_BEG
MAC_DEL = mc.MAC_DEL
MAC_END = mc.MAC_END
MAC_NSTART = mc.MAC_NSTART
MAC_START = mc.MAC_START
MEX = mc.MEX
MNL = mc.MNL
MOV = AxesSetter(mc.MOV)
MPL = mc.MPL
MVR = AxesSetter(mc.MVR)
POS = AxesSetter(mc.POS)
REF = mc.REF
RON = AxesSetter(mc.RON)
SAI = mc.SAI
SPA = AxesSetter(mc.SPA)
STP = mc.STP
SVO = AxesSetter(mc.SVO)
TRO = ArraySetter(mc.TRO)
VEL = AxesSetter(mc.VEL)
WAC = mc.WAC

qBRA = StringGetter(mc.qBRA)
qCST = AxesStringGetter(mc.qCST)
qCTO = NotImplemented(mc.qCTO)
qDFF = AxesGetter(mc.qDFF)
qDFH = AxesGetter(mc.qDFH)
qDIO = AxesGetter(mc.qDIO)
qERR = ValGetter(mc.qERR)
qHLP = StringGetter(mc.qHLP)
qHPA = StringGetter(mc.qHPA)
qIDN = StringGetter(mc.qIDN)
qJAS = ArrayGetter(mc.qJAS)
qJAX = NotImplemented(mc.qJAX)
qJBS = ArrayGetter(mc.qJBS)
qJLT = NotImplemented(mc.qJLT)
qJON = NotImplemented(mc.qJON)
qLIM = AxesGetter(mc.qLIM)
qMAC = NotImplemented(mc.qMAC)
qMOV = AxesGetter(mc.qMOV)
qONT = AxesGetter(mc.qONT)
qPOS = AxesGetter(mc.qPOS)
qREF = AxesGetter(mc.qREF)
qRON = AxesGetter(mc.qRON)
qSAI = StringGetter(mc.qSAI)
qSAI_ALL = StringGetter(mc.qSAI_ALL)
qSPA = NotImplemented(mc.qSPA)
qSRG = NotImplemented(mc.qSRG)
qSVO = AxesGetter(mc.qSVO)
qTAC = ValGetter(mc.qTAC)
qTAV = NotImplemented(mc.qTAV)
qTIO = ValGetter(mc.qTIO)
qTMN = AxesGetter(mc.qTMN)
qTMX = AxesGetter(mc.qTMX)
qTNJ = ValGetter(mc.qTNJ)
qTRO = NotImplemented(mc.qTRO)
qTVI = StringGetter(mc.qTVI)
qVEL = AxesGetter(mc.qVEL)
qVER = StringGetter(mc.qVER)
qVST = StringGetter(mc.qVST)
