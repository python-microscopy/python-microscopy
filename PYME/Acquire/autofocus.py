#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################


class AutoFocus(object):
    def __init__(self, scope, increment=0.5):
        self.scope = scope
        self.incr = increment
        self.lastMax =0
        self.lastMaxPos = 0
        
        self.lastStep = .5
        
    def tick(self, caller):
        m = self.scope.pa.dsa.std()
        if m > self.lastMax:
            #continue
            self.lastMax = m
            self.lastMaxPos = self.scope.GetPos()['z']
            
        else:
            if self.incr > 0:
                #reverse direction
                self.incr = -self.incr
            else:
                #already runing backwards
                self.scope.SetPos(z=self.lastMaxPos)
                self.scope.pa.WantFrameGroupNotification.remove(self.tick)
            
        self.scope.SetPos(z=self.lastMaxPos + self.incr)
        
        print 'af'
        
    def af(self, incr=0.5):
        self.lastMax = 0
        self.incr = incr
        self.scope.pa.WantFrameGroupNotification.append(self.tick)
