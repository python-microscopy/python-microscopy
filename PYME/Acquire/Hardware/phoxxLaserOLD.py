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
    def __init__(self, name,turnOn=False, portname='COM3', maxpower=0.14, **kwargs):
        self.ser_port = serial.Serial(portname, 500000, timeout=.1, writeTimeout=2)
        self.powerControlable = True
        #self.isOn=False
        
        self.doPoll=True

        self.qLock = threading.Lock()
        
        self.maxpower = maxpower
        
        self.commandQueue = Queue.Queue()
        self.replyQueue = Queue.Queue()
        self.ilbit = '0'
        #self.adhocQueue = Queue.Queue()
        
        self.adHocVals = {}
        
        self.threadPoll = threading.Thread(target=self._poll)
        self.threadPoll.start()
        
        #self.TurnOff()
        
        #self.power = self._getOutputPower()
        time.sleep(1)
        
        try:
            self.power = self._getOutputPower()
        except RuntimeError:
            self.power = 0
        

        Laser.__init__(self, name, turnOn)

    def IsOn(self):
        return self.isOn

    def TurnOn(self):
        self._check_interlock()

        ret, = self._query('LOn')
        if not ret == '>':
            raise RuntimeError('Error turning laser on')
        self.isOn = True

    def TurnOff(self):
        self._check_interlock()
        ret, = self._query('LOf')
        if not ret == '>':
            raise RuntimeError('Error turning laser on')
    
        self.isOn = False

    def _check_interlock(self):
        if self.ilbit == '1':
            raise RuntimeError('Interlock failure - this should reset automatically')

        il, = self._query('GLF') # Get Latched Failure. 
        self.ilbit = format(int(il,16), '#018b')[-10] #convert the response to 16-bit binary and take the interlock bit
        #print '647 laser interlock bit:' + self.ilbit

        if self.ilbit == '1':
            raise RuntimeError('Interlock failure - this should reset automatically')

    def SetPower(self, power):
        self._check_interlock()
        if power < 0 or power > 1:
            raise RuntimeError('Error setting laser power: Power must be between 0 and 1')
        self.power = power
        
        p = 0xFFF*power
        
        ps = '%03X' %p
        
        ret, = self._query('SLP', ps)
        if not ret == '>':
            raise RuntimeError('Error setting laser power')

        #if self.isOn:
        #    self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        self._check_interlock()
        pm = float(0xFFF)
        ret, = self._query('GLP')
        
        return int(ret, 16)/pm
    
    def _query(self, cmd, arg=None):
        s = cmd
        if arg:
            s = s + arg
            
        with self.qLock:
            self.commandQueue.put('?%s\r' % s)
            
            cmr, vals = self._decodeResponse(self.replyQueue.get(timeout=3))
        
        if not cmd == cmr:
            self._flush_queues()
            raise RuntimeError('Queried with %s but got response to %s' % (cmd, cmr))
            
        return vals

    def _flush_queues(self):
        with self.qLock:
            try:
                while True:
                    self.commandQueue.get(False)
            except Empty:
                pass
            try:
                while True:
                    self.replyQueue.get(False)
            except Empty:
                pass
    
    def _readline(self):
        s = []
        c = self.ser_port.read()
        while not c in ['', '\r']:
            s.append(c)
            c = self.ser_port.read()
            
        return ''.join(s)
        
    def _poll(self):
        while self.doPoll:
            if self.ilbit == '1':
                print('Resetting 647 interlock')
                self.ser_port.write('?RsC\r') # reset controller to clear the interlock error
                time.sleep(30) # wait until reset is finished
                self.ser_port.flush()
                #flush our output queue
                # try:
                #     while True:
                #         self.replyQueue.get(False)
                # except Empty:
                #     pass
                self.ilbit = '0'
                print('647 Interlock reset')

            #print 'p'
            try:
                cmd = self.commandQueue.get(False)
                #print cmd
                self.ser_port.write(cmd)
                self.ser_port.flush()
                
            except Queue.Empty:
                pass
            
            #wait a little for reply                
            time.sleep(.05)
            ret = self._readline()
            
            if (not ret == '') and (not self.ilbit == '1'): # ignore all replies during controller reset
                #print ret
                #process response - either a response or an ad-hoc message
                if not ret.startswith('$'): #normal response
                    self.replyQueue.put(ret)
                else: #adhoc
                    self._procAdHoc(ret)
                    #self.adhocQueue.put(ret)


            

                    
    def _decodeResponse(self, resp):
        cmd = resp[1:4]
        vals = resp[4:].split('\xA7')
        
        return cmd, vals
    
    def _procAdHoc(self, ret):
        cmd, vals = self._decodeResponse(ret)
        
        self.adHocVals[cmd] = vals
        
    def Close(self):
        try:
            self.TurnOff()
        #time.sleep(1)        
        finally:
            self.doPoll = False
            self.ser_port.close()
        
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
        #try:
        #    return self._getOutputPower()
        #except:
        #    return 0