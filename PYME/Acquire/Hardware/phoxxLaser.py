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

import serial
import time
import threading

try:
    import Queue
except ImportError:
    import queue as Queue

from PYME.Acquire.Hardware.lasers import Laser

class PhoxxLaser(Laser):
    def __init__(self, name,turnOn=False, portname='COM3', maxpower=0.14, power_fudge=1.0,**kwargs):
        self.ser_args = dict(port=portname, baudrate=500000, timeout=.1, writeTimeout=2)
        #self.ser_port = serial.Serial(portname, 500000, timeout=.1, writeTimeout=2)
        self.powerControlable = True
        #self.isOn=False
        
        self.doPoll=True
        self.maxpower = maxpower # maximum power, for our current only laser of this type it is 140mW
        # optionally de-rate the maximum power that we can set. Used as a work-around for calibration issues
        # in the built-in power meter which causes the erroneous detection of an over-power failure and triggers
        # and interlock
        self.power_fudge = power_fudge
        self.commandQueue = Queue.Queue()
        self.replyQueue = Queue.Queue()
        #self.adhocQueue = Queue.Queue()
        
        self.adHocVals = {}
        
        self.threadPoll = threading.Thread(target=self._poll)
        self.threadPoll.start()
        
        #self.TurnOff()
        
        #self.power = self._getOutputPower()
        time.sleep(1)
        self.power = self._getOutputPower()
        

        Laser.__init__(self, name, turnOn, **kwargs)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        ret, = self._query('LOn')
        if not ret == '>':
            raise RuntimeError('Error turning laser on')
        self.isOn = True

    def TurnOff(self):
        ret, = self._query('LOf')
        if not ret == '>':
            raise RuntimeError('Error turning laser on')
    
        self.isOn = False

    def SetPower(self, power):
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.power = power
        
        p = int(0xFFF*power*self.power_fudge)
        
        ps = '%03X' % p
        
        ret, = self._query('SLP', ps)
        if not ret == '>':
            raise RuntimeError('Error setting laser power')

        #if self.isOn:
        #    self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        pm = float(0xFFF)
        ret, = self._query('GLP')
        
        return int(ret, 16)/pm
        
    def _purge(self):
        try:
            while True:
                self.replyQueue.get_nowait()
        except:
            pass
    
    def _query(self, cmd, arg=None):
        self._purge()
        s = cmd
        if arg:
            s = s + arg
            
        self.commandQueue.put('?%s\r' % s)
        
        cmr, vals = self._decodeResponse(self.replyQueue.get(timeout=3))
        
        if not cmd == cmr:
            raise RuntimeError('Queried with %s but got response to %s' % (cmd, cmr))
            
        return vals
    
    def _readline(self, ser):
        s = []
        c = ser.read().decode()
        while not c in ['', '\r']:
            s.append(c)
            c = ser.read().decode()
            
        return ''.join(s)
        
    def _poll(self):
        while self.doPoll:
            #print 'p'
            with serial.Serial(**self.ser_args) as ser:
                try:
                    cmd = self.commandQueue.get(False)
                    #print cmd
                    ser.write(cmd.encode())
                    ser.flush()
                    
                except Queue.Empty:
                    pass
                
                #wait a little for reply                
                time.sleep(.1)
                ret = self._readline(ser)
                #print(ret)
                
                if not ret == '':
                    #print ret
                    #process response - either a response or an ad-hoc message
                    if not ret.startswith('$'): #normal response
                        self.replyQueue.put(ret)
                    else: #adhoc
                        self._procAdHoc(ret)
                        #self.adhocQueue.put(ret)
                    
            time.sleep(.05)
                    
    def _decodeResponse(self, resp):
        cmd = resp[1:4]
        vals = resp[4:].split('\xA7')
        
        return cmd, vals
    
    def _procAdHoc(self, ret):
        cmd, vals = self._decodeResponse(ret)
        
        self.adHocVals[cmd] = vals
        
    def Close(self):
        print('Shutting down %s' % self.name)
        #try:
        self.TurnOff()
        time.sleep(.1)        
        #finally:
        self.doPoll = False
            #self.ser_port.close()
        
    def __del__(self):
        self.Close()
        
    def GetOutputmW(self):
        return float(self.adHocVals['MDP'][0])
        
    def GetStatusText(self):
        try:
            pow = self.GetOutputmW()
            return '%s laser power: %3.3f mW' % (self.name, pow)
        except:
            return '%s laser power: ERR' % self.name

    def GetPower(self):
        return self.power
        #return self._getOutputPower()