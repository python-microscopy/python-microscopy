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

def createSimpleTransitionMatrix(pPA=[1e6,.1,0] , pOnDark=[0,0,.1], pDarkOn=[0,.001,0], pOnBleach=[0,0,0], pCagedBlinked = [0,0,0]):
    M = np.zeros((states.n,states.n,len(pPA)), 'f')
    M[states.caged, states.active, :] = pPA
    M[states.active, states.blinked, :] = pOnDark
    M[states.blinked, states.active, :] = pDarkOn
    M[states.active, states.bleached,:] = pOnBleach
    M[states.caged, states.blinked, :] = pCagedBlinked
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
        self.transitionProbabilities = transitionProbablilities * np.concatenate(([1], abs(np.cos(thetas))),0)
        self.excitationCrossections = excitationCrossections *abs(np.cos(thetas))

    def illuminate(self, laserPowers, expTime):
        dose = np.concatenate(([1],laserPowers),0)*expTime
        #grab transition matrix
        transVec = (self.transitionProbabilities[self.state,:,:]*dose).sum(1)
        transVec[self.state]= 1 - transVec.sum()
        transCs = transVec.cumsum()
        
        r = np.random.RandomState().rand()  # replace with np.random.default_rng() on later numpy versions
        
        for i in range(len(transVec)):
            if (r < transCs[i]):
                self.state = i
                if (i == self.activeState):
                    return (laserPowers*self.excitationCrossections).sum()*expTime
                else:
                    return 0
        
class fluors:
    def __init__(self,x, y, z,  transitionProbablilities, excitationCrossections, thetas = [0,0], initialState=states.active, activeState=states.active):
        self.fl = np.zeros(len(x), [('x', 'f'),('y', 'f'),('z', 'f'),('exc', '2f'), ('abcosthetas', '2f'),('state', 'i')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        #fl['exc'][:] = abs(cos(thetas))
        self.fl['exc'][:] = excitationCrossections 
        self.fl['abcosthetas'][:] = abs(np.cos(thetas))
        self.fl['state'][:] = initialState 

        self.transitionTensor = transitionProbablilities.astype('f')
        self.activeState = activeState
        #self.TM = self.transitionTensor[self.fl['state'],:,:].copy()
        #self.illuminationFunction = illuminationFunction

    #return fl
    if HAVE_ILLUMINATE_MOD:
        #use faster cythoned version of function if available
        def illuminate(self,laserPowers, expTime, position=[0,0,0], illuminationFunction = 'ConstIllum'):
            dose = (np.concatenate(([1.0],laserPowers),0)*expTime).astype('f')
            ilFrac = illuminationFunctions[illuminationFunction](self.fl, position)
            return illuminate.illuminate(self.transitionTensor, self.fl, self.fl['state'], self.fl['abcosthetas'], dose, ilFrac, self.activeState)
    else:
        def illuminate(self, laserPowers, expTime, position=[0,0,0], illuminationFunction = 'ConstIllum'):
            dose = np.concatenate(([1.0],laserPowers),0)*expTime
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
            
            r = np.random.RandomState().rand(len(self.fl))  # replace with np.random.default_rng() on later numpy versions
            
            self.fl['state'] = (transCs < r[:, None]).sum(1)
            
            return (self.fl['state'] == self.activeState)*(self.fl['exc'][:,0]*c0 + self.fl['exc'][:,1]*c1)


class specFluors(fluors):
    def __init__(self,x, y, z,  transitionProbablilities, excitationCrossections, thetas = [0,0], spectralSig = [1,0], initialState=states.caged, activeState=states.active):
        self.fl = np.zeros(len(x), [('x', 'f'),('y', 'f'),('z', 'f'),('exc', '2f'), ('abcosthetas', '2f'),('state', 'i'), ('spec', '2f')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        #fl['exc'][:] = abs(cos(thetas))
        self.fl['exc'][:] = excitationCrossections 
        self.fl['abcosthetas'][:] = abs(np.cos(thetas))
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
        self.fl = np.zeros(len(x), [('x', 'f'), ('y', 'f'), ('z', 'f'),
                                 ('state', 'i'), ('spec', '2f')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        self.fl['state'][:] = initialState
        self.fl['spec'][:] = spectralSig

        self.laserPowers = [1.0,1.0]
        self.expTime = 0.01

        self.activeState = activeState

        self.stateQueue = Queue.Queue()

        self.doPoll = True

        # initialize the state vector
        self.state_curr = self.fl['state'].copy() #np.ones(self.fl['state'].shape)
        self.times = np.zeros_like(self.state_curr, dtype='f4')
        self.active_time = np.zeros_like(self.state_curr, dtype=float)
        
        self.count_on = []
        self.count_off = []

        self.threadPoll = threading.Thread(target=self._calculate_times)
        self.threadPoll.start()
        time.sleep(.5)

    def _calculate_times(self):
        while self.doPoll:
            # make sure we're not overfilling the queue
            if self.stateQueue.qsize() >= 1.0e6/self.expTime:
                time.sleep(.1)

            #print(self.times[0:19])

            # Add the new state to the queue
            self.stateQueue.put(self.times*(self.state_curr == self.activeState))

            # calculate on time, off time
            # compute state vector, toss it in the queue

            # update the states
            self.times = np.subtract(self.times, self.expTime)
            new_locs = self.times < 0
            
            #trans_locs = (self.times > 0) & (self.times < self.expTime)
            
            #self.state_curr[new_locs] = 1 - np.round(self.state_curr[new_locs])
            
            #casting to bool and then back might not be necessary
            self.state_curr[new_locs] = ~self.state_curr[new_locs].astype(bool)
            
            #self.state_curr[trans_locs] = self.times[trans_locs]/self.expTime
            #print(self.state_curr[trans_locs].dtype)

            # find the current active states
            idxs_on = (self.state_curr == self.activeState) & new_locs
            idxs_off = (self.state_curr != self.activeState) & new_locs

            # Calculate how long these suckers will be on
            #if any(idxs_on):
            r_on = np.random.rand(sum(idxs_on.astype(int)))
            self.times[idxs_on] = self.histogram.get_time_splined(self.laserPowers[1], r_on, 'on')
            
            self.count_on.extend(self.times[idxs_on].toList()) #FIXME - What is this used for??? this will be really slow!
            
           
            #if any(idxs_off) > 0:
            r_off = np.random.rand(sum(idxs_off.astype(int)))
            self.times[idxs_off] = self.histogram.get_time_splined(self.laserPowers[1], r_off, 'off')
            
            self.count_off.extend(self.times[idxs_off].tolist()) #FIXME - this will be really slow


    def illuminate(self, laserPowers, expTime, position=[0, 0, 0],
                   illuminationFunction = 'ConstIllum'):

        # Restart simulation on parameter changes
        if not np.isclose(laserPowers[1], self.laserPowers[1]*1e3/self.histogram.prange('on')[-1], atol=1) or (expTime != self.expTime):
            #TODO - tolerance copied from Rollins/Marin branch - does a tolerance of 1 make sense??
            self.laserPowers = [p/(1e3*self.histogram.prange('on')[-1]) for p in laserPowers]
            self.expTime = expTime
            self.count_on = []
            self.count_off = []
            #print('exptime', self.expTime)
            with self.stateQueue.mutex:
                self.stateQueue.queue.clear()
            time.sleep(.5)

        #print(self.laserPowers)
        
        dose = 1e5*self.expTime #TODO - I assume this is the photon count. Break out as a parameter???

        # Grab a state off the queue and display
        #curr_state = self.stateQueue.get()
        active_time = self.stateQueue.get().copy()
        
        self.fl['state'] = active_time > 0

        ilFrac = illuminationFunctions[illuminationFunction](self.fl,position) #*expTime*\laserPowers[1]*90
        
        return np.minimum(active_time, self.expTime)*ilFrac*dose

    def __del__(self):
        self.doPoll = False
        self.threadPoll.join()
