#!/usr/bin/python

##################
# fluor.py
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

from scipy import *
import numpy as np
import threading
import time

try:
    import Queue
except ImportError:
    import queue as Queue

try:
    from . import illuminate
    HAVE_ILLUMINATE_MOD = True
except ImportError:
    HAVE_ILLUMINATE_MOD = False
    

class states:
    caged, active, blinked, bleached = range(4)
    n=4

ALL_TRANS, FROM_ONLY, TO_ONLY = range(3)


illuminationFunctions = {}

def registerIllumFcn(fcn):
    illuminationFunctions[fcn.__name__] = fcn
    return fcn

@registerIllumFcn
def ConstIllum(fluors, position):
    return 1.0

def createSimpleTransitionMatrix(pPA=[1e6,.1,0] , pOnDark=[0,0,.1], pDarkOn=[0,.001,0], pOnBleach=[0,0,0]):
    M = zeros((states.n,states.n,len(pPA)), 'f')
    M[states.caged, states.active, :] = pPA
    M[states.active, states.blinked, :] = pOnDark
    M[states.blinked, states.active, :] = pDarkOn
    M[states.active, states.bleached,:] = pOnBleach
    return M

class fluorophore:
    def __init__(self, x, y, z, transitionProbablilities, excitationCrossections, thetas = [0,0], initialState=states.active, activeState=states.active):
        """Create a new 'fluorophore' having one dark state where:
        transitionProbablilities is a 4x4x[number of laser wavelengths + 1] tensor of transition probablilites (units = 1/mJ)
        the diagonal elements (transition from one state to itself) should be zero as they'll be calculated later to make sum(P) =1
        excitationCrossections is an array of length [number of laser wavelengths] giving the number of photons per mJ emmited when in the on state
        thetas is an array of length [number of laser wavelengths] giving the angle (in rad.) between dipole moment of the fluoropore and the laser polarisations
        initialState is the initial state of the fluorophore"""

        self.x = x;
        self.y = y;
        self.z = z;

        self.state = initialState
        self.activeState = activeState
        self.thetas = thetas
        self.transitionProbabilities = transitionProbablilities * concatenate(([1], abs(cos(thetas))),0)
        self.excitationCrossections = excitationCrossections *abs(cos(thetas))

    def illuminate(self, laserPowers, expTime):
        dose = concatenate(([1],laserPowers),0)*expTime
        #grab transition matrix
        transVec = (self.transitionProbabilities[self.state,:,:]*dose).sum(1)
        transVec[self.state]= 1 - transVec.sum()
        transCs = transVec.cumsum()
        
        r = rand()
        
        for i in range(len(transVec)):
            if (r < transCs[i]):
                self.state = i
                if (i == self.activeState):
                    return (laserPowers*self.excitationCrossections).sum()*expTime
                else:
                    return 0
        
class fluors:
    def __init__(self,x, y, z,  transitionProbablilities, excitationCrossections, thetas = [0,0], initialState=states.active, activeState=states.active):
        self.fl = zeros(len(x), [('x', 'f'),('y', 'f'),('z', 'f'),('exc', '2f'), ('abcosthetas', '2f'),('state', 'i')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        #fl['exc'][:] = abs(cos(thetas))
        self.fl['exc'][:] = excitationCrossections 
        self.fl['abcosthetas'][:] = abs(cos(thetas))
        self.fl['state'][:] = initialState 

        self.transitionTensor = transitionProbablilities.astype('f')
        self.activeState = activeState
        #self.TM = self.transitionTensor[self.fl['state'],:,:].copy()
        #self.illuminationFunction = illuminationFunction

    #return fl
    if HAVE_ILLUMINATE_MOD:
        #use faster cythoned version of function if available
        def illuminate(self,laserPowers, expTime, position=[0,0,0], illuminationFunction = 'ConstIllum'):
            dose = (np.concatenate(([1],laserPowers),0)*expTime).astype('f')
            ilFrac = illuminationFunctions[illuminationFunction](self.fl, position)
            return illuminate.illuminate(self.transitionTensor, self.fl, self.fl['state'], self.fl['abcosthetas'], dose, ilFrac, self.activeState)
    else:
        def illuminate(self, laserPowers, expTime, position=[0,0,0], illuminationFunction = 'ConstIllum'):
            dose = concatenate(([1],laserPowers),0)*expTime
            #grab transition matrix
            transMat = self.transitionTensor[self.fl['state'],:,:].copy()
            
    
            ilFrac = illuminationFunctions[illuminationFunction](self.fl, position)
    
            c0 = self.fl['abcosthetas'][:,0]*dose[1]*ilFrac
            c1 = self.fl['abcosthetas'][:,1]*dose[2]*ilFrac
            #print c0.shape
            #print transMat.shape,transMat.dtype
            #print vstack((c0,c0, c0,c0)).shape
    
            transMat[:,:,0] *= dose[0]
            transMat[:,:,1] *= c0[:,None] #vstack((c0,c0, c0,c0)).T 
            transMat[:,:,2] *= c1[:,None] #vstack((c1, c1, c1, c1)).T
            
            transVec = transMat.sum(2)
            tvs = transVec.sum(1)
            for i in range(transVec.shape[1]):
                m = self.fl['state'] == i
                transVec[m, i]= 1 - tvs[m]
            transCs = transVec.cumsum(1)
            
            r = rand(len(self.fl))
            
            self.fl['state'] = (transCs < r[:, None]).sum(1)
            
            return (self.fl['state'] == self.activeState)*(self.fl['exc'][:,0]*c0 + self.fl['exc'][:,1]*c1)


class specFluors(fluors):
    def __init__(self,x, y, z,  transitionProbablilities, excitationCrossections, thetas = [0,0], spectralSig = [1,0], initialState=states.caged, activeState=states.active):
        self.fl = zeros(len(x), [('x', 'f'),('y', 'f'),('z', 'f'),('exc', '2f'), ('abcosthetas', '2f'),('state', 'i'), ('spec', '2f')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        #fl['exc'][:] = abs(cos(thetas))
        self.fl['exc'][:] = excitationCrossections 
        self.fl['abcosthetas'][:] = abs(cos(thetas))
        self.fl['state'][:] = initialState
        self.fl['spec'][:] = spectralSig

        self.transitionTensor = transitionProbablilities.astype('f')
        self.activeState = activeState
    

class EmpiricalHistFluors(fluors):
    """
    Fluorophores with on/off times generated by empirical data.

    Note
    ----
        This class relies on empirical data. For fluorophore simulation based on
        first principles see fluors.
    """

    def __init__(self, x, y, z, histogram, spectralSig = [1,0],
                 initialState=states.active, activeState=states.active):

        self.histogram = histogram
        self.fl = zeros(len(x), [('x', 'f'), ('y', 'f'), ('z', 'f'),
                                 ('state', 'i'), ('spec', '2f')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        self.fl['state'][:] = initialState
        self.fl['spec'][:] = spectralSig

        self.laserPowers = 0
        self.expTime = 0

        self.flvec = range(len(self.fl))

        self.time = 0

        self.activeState = activeState

        self.stateQueue = Queue.Queue()
        self.stateQueueCount = 0

        self.doPoll = True

        self.threadPoll = threading.Thread(target=self._calculate_times)
        self._start_calculate_times()

    def _start_calculate_times(self):
        self.stateQueueCount = 0
        self.threadPoll.start()

    def _calculate_times(self):
        while self.doPoll:
            # make sure we're not overfilling the queue
            if self.stateQueueCount >= 2./self.expTime:
                time.sleep(.5)

            # calculate on time, off time
            # compute state vector, toss it in the queue

            has_transitioned = next_transition < self.time
            will_transition = (next_transition < (self.time + expTime)) & \
                              (~has_transitioned)
            will_transition_intensities = ilFrac * will_transition * (
            expTime - abs(next_transition - self.time)) / expTime

            # increment the number of elements in the queue
            self.stateQueueCount += 1

    def illuminate(self, laserPowers, expTime, position=[0, 0, 0],
                   illuminationFunction = 'ConstIllum'):

        # Update if necessary
        if laserPowers[1] != self.laserPowers or expTime != self.expTime:
            self.laserPowers = laserPowers[1]
            self.expTime = expTime
            self.doPoll = False
            time.sleep(.1)
            with self.stateQueue.mutex:
                self.stateQueue.queue.clear()
            time.sleep(.1)
            self._start_calculate_times()

        state = self.stateQueue.get()
        self.stateQueueCount -= 1
        self.fl['state'] = state == self.activeState

        ilFrac = illuminationFunctions[illuminationFunction](self.fl, position)*expTime*1e3

        return (self.fl['state'] == self.activeState)*ilFrac