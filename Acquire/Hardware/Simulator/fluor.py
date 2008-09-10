from scipy import *

class states:
    caged, active, blinked, bleached = range(4)
    n=4

ALL_TRANS, FROM_ONLY, TO_ONLY = range(3)

def createSimpleTransitionMatrix(pPA=[0,.1,0] , pOnDark=[0,0,.1], pDarkOn=[.1,0,0], pOnBleach=[0,0,.01]):
    M = zeros((states.n,states.n,len(pPA)), 'f')
    M[states.caged, states.active, :] = pPA
    M[states.active, states.blinked, :] = pOnDark
    M[states.blinked, states.active, :] = pDarkOn
    M[states.active, states.bleached,:] = pOnBleach
    return M

class fluorophore:
    def __init__(self, x, y, z, transitionProbablilities, excitationCrossections, thetas = [0,0], initialState=states.caged, activeState=states.active):
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
    def __init__(self,x, y, z,  transitionProbablilities, excitationCrossections, thetas = [0,0], initialState=states.caged, activeState=states.active):
        self.fl = zeros(len(x), [('x', 'f'),('y', 'f'),('z', 'f'),('exc', '2f'), ('abcosthetas', '2f'),('state', 'i')])
        self.fl['x'] = x
        self.fl['y'] = y
        self.fl['z'] = z
        #fl['exc'][:] = abs(cos(thetas))
        self.fl['exc'][:] = excitationCrossections 
        self.fl['abcosthetas'][:] = abs(cos(thetas))
        self.fl['state'][:] = initialState 

        self.transitionTensor = transitionProbablilities
        self.activeState = activeState

    #return fl

    def illuminate(self, laserPowers, expTime):
        dose = concatenate(([1],laserPowers),0)*expTime
        #grab transition matrix
        transMat = self.transitionTensor[self.fl['state'],:,:].copy()

        c0 = self.fl['abcosthetas'][:,0]*dose[1]
        c1 = self.fl['abcosthetas'][:,1]*dose[2]
        #print c0.shape
        #print transMat.shape
        #print vstack((c0,c0, c0,c0)).shape

        transMat[:,:,0] *= dose[0]
        transMat[:,:,1] *= vstack((c0,c0, c0,c0)).T 
        transMat[:,:,2] *= vstack((c1, c1, c1, c1)).T
        
        transVec = transMat.sum(2)
        tvs = transVec.sum(1)
        for i in range(transVec.shape[1]):
            transVec[self.fl['state'] == i, i]= 1 - tvs[self.fl['state'] == i]
        transCs = transVec.cumsum(1)
        
        r = rand(len(self.fl))
        
        #print transVec.shape
        #print transCs.shape
        #print transCs
        nch = ones(r.shape)
        for i in range(transVec.shape[1]):
            ch = (r < transCs[:,i]) * nch
            self.fl['state'] = i*ch + self.fl['state']*(1 - ch)
            nch *= (1 - ch) 
        
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

        self.transitionTensor = transitionProbablilities
        self.activeState = activeState
    
