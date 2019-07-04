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
from scipy import ndimage

class AutoFocus(object):
    def __init__(self, scope, increment=0.5):
        self.scope = scope
        self.incr = increment
        self.lastMax =0
        self.lastMaxPos = 0
        
        self.lastStep = .5
        
    def OnFrameGroup(self, **kwargs):
        im_f = self.scope.frameWrangler.currentFrame.astype('f')
        self.im_d = ndimage.gaussian_filter(im_f, 1) - ndimage.gaussian_filter(im_f, 5)
        m = self.im_d.std()#self.scope.frameWrangler.currentFrame.std()
        #m = im_f.std()
        if m > self.lastMax:
            #continue
            self.lastMax = m
            self.lastMaxPos = self.scope.state['Positioning.z']
            
        else:
            if self.incr > 0:
                #reverse direction
                self.incr = -self.incr
            else:
                #already runing backwards
                self.scope.state['Positioning.z']=self.lastMaxPos
                #self.scope.frameWrangler.WantFrameGroupNotification.remove(self.tick)
                self.scope.frameWrangler.onFrameGroup.disconnect(self.OnFrameGroup)

                print('af_done')
            
        #self.scope.SetPos(z=self.lastMaxPos + self.incr)
        self.scope.state['Positioning.z'] = self.lastMaxPos + self.incr
        
        print('af, %s' % m)
        
    def af(self, incr=0.5):
        self.lastMax = 0
        self.incr = incr
        #self.scope.frameWrangler.WantFrameGroupNotification.append(self.tick)
        self.scope.frameWrangler.onFrameGroup.connect(self.OnFrameGroup)
