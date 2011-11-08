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

import serial
import time
import threading
import Queue

from lasers import Laser

class PhoxxLaser(Laser):
    def __init__(self, name,turnOn=False, portname='COM3'):
        self.ser_port = serial.Serial(portname, 500000, timeout=.1, writeTimeout=2)
        self.powerControlable = True
        #self.isOn=False
        
        self.doPoll=True
        
        
        self.commandQueue = Queue.Queue()
        self.replyQueue = Queue.Queue()
        #self.adhocQueue = Queue.Queue()
        
        self.adHocVals = {}
        
        self.threadPoll = threading.Thread(target=self._poll)
        self.threadPoll.start()
        
        #self.TurnOff()
        
        #self.power = self._getOutputPower()
        time.sleep(1)
        

        Laser.__init__(self, name, turnOn)

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
        
        p = 0xFFF*power
        
        ps = '%03X' %p
        
        ret, = self._query('SLP', ps)
        if not ret == '>':
            raise RuntimeError('Error setting laser power')

        #if self.isOn:
        #    self.TurnOn() #turning on actually sets power

    def _getOutputPower(self):
        pm = float(0xFFF)
        ret, = self._query('GLP')
        
        return int(ret, 16)/pm
    
    def _query(self, cmd, arg=None):
        s = cmd
        if arg:
            s = s + arg
            
        self.commandQueue.put('?%s\r' % s)
        
        cmr, vals = self._decodeResponse(self.replyQueue.get(timeout=3))
        
        if not cmd == cmr:
            raise RuntimeError('Queried with %s but got response to %s' % (cmd, cmr))
            
        return vals
    
    def _readline(self):
        s = []
        c = self.ser_port.read()
        while not c in ['', '\r']:
            s.append(c)
            c = self.ser_port.read()
            
        return ''.join(s)
        
    def _poll(self):
        while self.doPoll:
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
            
            if not ret == '':
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
        self.TurnOff()
        #time.sleep(1)        
        
        self.doPoll = False
        self.ser_port.close()
        
    def __del__(self):
        self.Close()

    def GetPower(self):
        #return self.power
        return self._getOutputPower()