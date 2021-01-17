#!/usr/bin/python

##################
# piezo_e816.py
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

#from . import Mercury as m
from PYME.Acquire.Hardware.Piezos import base_piezo
import time
import numpy  as np
import serial

import threading
import logging
logger = logging.getLogger(__name__)

class tPoll(threading.Thread):
    def __init__(self, stepper):
        self.stepper = stepper
        self.kill = False
        threading.Thread.__init__(self)
        
    def run(self):
        while not self.kill:
            try:
                self.stepper._get_status()
                if not self.stepper.referencing:
                    self.stepper.RefreshPos()
            except:
                pass
            time.sleep(.5)

class mercuryJoystick:
    def __init__(self, stepper):
        self.stepper = stepper

    def Enable(self, enabled = True):
        if not self.IsEnabled() == enabled:
            self.stepper.SetJoystick(enabled)

    def IsEnabled(self):
        return self.stepper.joystickOn


class mercuryStepper(base_piezo.PiezoBase):
    units_um = 1000
    def __init__(self, comPort='COM4', baud=115200, axes=['X', 'Y'], steppers=['M-229.25S', 'M-229.25S']):
        self.axes = axes
        self.steppers = steppers
        self.joystickOn = False

        self.joystick = mercuryJoystick(self)

        self.lock = threading.RLock()
        


        #connect to the controller
        self.ser_port = serial.Serial(comPort, baud, timeout=2, writeTimeout=2)

        with self.lock:
            self.minTravel = [float(self.query('TMN?', a)) for a in self.axes]
            self.maxTravel = [float(self.query('TMX?', a)) for a in self.axes]

            # turn joystick off (if on)
            for i, a in enumerate(self.axes):
                self.set('JON', a, '1 0', omit_axis=True)

            #initialise and calibrate axes
            for a in self.axes:
                self.set('SVO', a, 1)
                self.set('FRF', a)

            self.referencing = True
            self.onTarget = False

            #set joystick to use parabolic lookup table
            #for a in self.axes:
            #    self.set('JDT', a, 2, omit_axis=True)



        self.poll = tPoll(self)
        self.poll.daemon = True
        self.poll.start()

        while self.referencing:
            time.sleep(.5)

        with self.lock:
            for i, a in enumerate(self.axes):
                self.set('JAX', a, '1 1 %s' %a, omit_axis=True)

                #set joystick to use parabolic loopup table
                self.set('JDT', a, '1 2', omit_axis=True)


    def _get_status(self):
        with self.lock:
            status = [int(self.query('SRG?', a, ['1',]), 16) for a in self.axes]

            moving=False
            onTarget=True
            referencing=False

            for s in status:
                moving |= ((s & (1 << 13)) > 0)
                onTarget &= ((s & (1 << 15)) > 0)
                referencing |= ((s & (1 << 14)) > 0)

            #print(moving, onTarget, referencing)
            self.moving, self.onTarget, self.referencing = moving, onTarget, referencing

    def _strip_response_address(self, resp):
        """ daisy chained controllers respond with "dest src resp", split this
        """

        #print( 'response:', resp)
        if (resp == u''):
            #no response
            return None,None

        dest = int(resp[0])
        assert(dest == 0) #destination should be the computer - ie controllerID 0
        src = int(resp[2])

        return resp[4:], src

    def run_cmd(self, command, controllerID, query=False):
        cmd = '%d %s\n' % (controllerID, command)
        #print('cmd:', cmd)
        self.ser_port.write(cmd.encode())
        self.ser_port.flush()

        if query:
            resp, src = self._strip_response_address(self.ser_port.readline().decode())
            if (src != controllerID):
                logger.error('unexpected response source')
        else:
            resp = None

        self.ser_port.write(('%d ERR?\n'% controllerID).encode())
        err = int(self._strip_response_address(self.ser_port.readline().decode())[0])

        if err != 0:
            # we have an error status
            logger.error('Error code: [%d] on command "%s", controler %d'%(err, command, controllerID))

        return resp

    def query(self, command, axis, extra_params=[]):
        assert(command.endswith('?'))
        controller = self.axes.index(axis) + 1
        cmd = ' '.join([command, axis] + extra_params)
        resp = self.run_cmd(cmd, controllerID=controller, query=True).split('=')[-1]
        return resp

    def set(self, command, axis, value=None, extra_params=[], omit_axis=False):
        controller = self.axes.index(axis) + 1
        if omit_axis:
            cmd = ' '.join([command, ] + extra_params)
        else:
            cmd = ' '.join([command, axis] + extra_params)
        if value is not None:
            cmd = ' '.join([cmd, str(value)])
        self.run_cmd(cmd, controllerID=controller)

    def SetSoftLimits(self, axis, lims):
         with self.lock:
            self.set('SPA', self.axes[axis], lims[0], ['0x30',] )
            self.set('SPA', self.axes[axis], lims[1], ['0x15', ])

    def ReInit(self):
        pass 
        
    def MoveTo(self, iChan, fPos, timeout=False):
        with self.lock:
            self._pause_joystick(False)
            tgt = fPos
            if (fPos >= self.minTravel[iChan]):
                if (fPos <= self.maxTravel[iChan]):
                    tgt = fPos
                else:
                    tgt = self.maxTravel[iChan]
            else:
                tgt = self.minTravel[iChan]

            self.onTarget = False
            self.set('MOV', self.axes[iChan], '%3.4f' % fPos)
            #m.MOV(self.connID, self.axes[iChan], [tgt])
            #self.last_poss[iChan] = tgt

            self._pause_joystick(True)

    def GetPos(self, iChan=0):
        # self.lock.acquire()
        # ret = m.qPOS(self.connID, self.axes[iChan])[0]
        # self.lock.release()
        # return ret
        return self.last_poss[iChan]

    def IsMoving(self, iChan=0):
        return self.moving

    def IsOnTarget(self):
        return self.onTarget

    def RefreshPos(self):
        self.lock.acquire()
        self.last_poss=[float(self.query('POS?', a)) for a in self.axes]
        self.lock.release()

    def GetLastPos(self, iChan=0):
        return self.last_poss[iChan]

    def _pause_joystick(self, on=False, chans=[0,1]):
        if self.joystickOn:
            if on:
                #wait until our previous move is finished
                timeout = time.time()+ 10
                while (not self.onTarget) and time.time() < timeout:
                    time.sleep(0.5)
                    self._get_status()

            self.SetJoystick(on, chans)

    def SetJoystick(self, on=True, chans=[0,1]):
        with self.lock:
            jv = '1 1' if on else '1 0'
            for c in chans:
                self.set('JON', self.axes[c], jv, omit_axis=True)

            self.joystickOn = on


    def GetVelocity(self, iChan=0):
        with self.lock:
            ret = float(self.query('VEL?', self.axes[iChan]))

        return ret

    def SetVelocity(self, iChan, velocity):
        with self.lock:
            self.set('VEL', self.axes[iChan], '%3.4f' % velocity)
        return ret

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
        #m.CloseConnection(self.connID)
        

#    def __del__(self):
#        m.CloseConnection(self.connID)
