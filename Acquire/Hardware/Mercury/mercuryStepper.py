#!/usr/bin/python

##################
# piezo_e816.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import Mercury as m
import time
import numpy  as np

import threading

class tPoll(threading.Thread):
    def __init__(self, stepper):
        self.stepper = stepper
        self.kill = False
        threading.Thread.__init__(self)
        
    def run(self):
        while not self.kill:
            try:
                self.stepper.RefreshPos()
            except:
                pass
            time.sleep(.2)


class mercuryStepper:
    def __init__(self, comPort=5, baud=9600, axes=['A', 'B'], steppers=['M-229.25S', 'M-229.25S']):
        self.axes = axes
        self.steppers = steppers
        self.joystickOn = False

        self.lock = threading.RLock()
        
        self.lock.acquire()

        #connect to the controller
        self.connID = m.ConnectRS232(comPort, baud)

        if self.connID == -1:
            raise RuntimeError('Could not connect to Mercury controller')

        #tell the controller which stepper motors it's driving
        m.CST(self.connID, ''.join(self.axes), '\n'.join(self.steppers))

        #initialise axes
        m.INI(self.connID, ''.join(self.axes))

        #callibrate axes using reference switch
        m.REF(self.connID, ''.join(self.axes))

        while np.any(m.IsReferencing(self.connID, ''.join(self.axes))):
            time.sleep(.5)

        self.minTravel = m.qTMN(self.connID, ''.join(self.axes))
        self.maxTravel = m.qTMX(self.connID, ''.join(self.axes))

        self.last_poss = m.qPOS(self.connID, ''.join(self.axes))

        self.lock.release()

        self.poll = tPoll(self)
        self.poll.start()

    def SetSoftLimits(self, axis, lims):
        self.lock.acquire()
        m.SPA(self.connID, self.axes[axis], [48, 21], lims, self.steppers[axis])
        self.lock.release()

    def ReInit(self):
        pass 
        
    def MoveTo(self, iChan, fPos, timeout=False):
        self.lock.acquire()
        tgt = fPos
        if (fPos >= self.minTravel[iChan]):
            if (fPos <= self.maxTravel[iChan]):
                tgt = fPos
            else:
                tgt = self.maxTravel[iChan]
        else:
            self.minTravel[iChan]

        m.MOV(self.connID, self.axes[iChan], [tgt])
        self.last_poss[iChan] = tgt
        self.lock.release()

    def GetPos(self, iChan=0):
        self.lock.acquire()
        ret = m.qPOS(self.connID, self.axes[iChan])[0]
        self.lock.release()
        return ret

    def IsMoving(self, iChan=0):
        self.lock.acquire()
        ret = m.IsMoving(self.connID, self.axes[iChan])[0]
        self.lock.release()
        return ret

    def RefreshPos(self):
        self.lock.acquire()
        self.last_poss = m.qPOS(self.connID, ''.join(self.axes))
        self.lock.release()

    def GetLastPos(self, iChan=0):
        return self.last_poss[iChan]

    def SetJoystick(self, on = True, chans=[0,1]):
        self.lock.acquire()
        jv = [on for c in chans]
        m.JON(self.connID, [c + 1 for c in chans], jv)
        self.joystickOn = on
        self.lock.release()

    def GetControlReady(self):
        return True
    def GetChannelObject(self):
        return 0
    def GetChannelPhase(self):
        return 1
    def GetMin(self,iChan=0):
        return self.minTravel[iChan]
    def GetMax(self, iChan=0):
        return self.maxTravel[iChan]

    def Cleanup(self):
        self.poll.kill = True
        m.CloseConnection(self.connID)
        

#    def __del__(self):
#        m.CloseConnection(self.connID)
