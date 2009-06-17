import numpy as np
import scipy.ndimage
import scipy.stats
from pylab import *


class simClust1D:
    def __init__(self, N=50, pool=10, pAdd = .01, pAddN=.1, pStay=.1, pStayN=.4):
        self.g = np.zeros(N)

        self.pool = pool

        self.nFree = pool
        self.pAdd = pAdd
        self.pAddN = pAddN
        self.pStay = pStay
        self.pStayN = pStayN

    def p(self):
        neighbours = np.convolve(self.g, [1,0,1], 'same')

        return self.nFree*(self.pAdd + self.pAddN*neighbours) + self.pStay + self.pStayN*neighbours

    def MCStep(self):
        r = np.random.rand(len(self.g))

        self.g = r < self.p()

        self.nFree = max(self.pool - self.g.sum(), 0)


class simClust2D:
    def __init__(self, N=20, pool=20, pAdd = .00005, pAddN=.005, pStay=.85, pStayN=.14):
        #self.N = N
        self.g = np.zeros((N, N))

        self.pool = pool

        self.nFree = pool
        self.pAdd = pAdd
        self.pAddN = pAddN
        self.pStay = pStay
        self.pStayN = pStayN

    def p(self):
        neighbours = scipy.ndimage.convolve(self.g.astype('f'), np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='constant', cval=0)

        return self.nFree*(self.pAdd + self.pAddN*neighbours) + self.g*(self.pStay + self.pStayN*neighbours)

    def MCStep(self):
        r = np.random.rand(*self.g.shape)

        self.g = r < self.p()

        #self.nFree = max(self.pool - self.g.sum(), 0)


def do2DSim(Nsims=100):
    clusterSs = []
    for i in range(NSims):
        sc  = simCluster.simClust2D(N=50, pool=5, pAdd=.00005, pAddN=0.005, pStay=0.937, pStayN=0)
        for j in range(nIters):
            sc.MCStep()
        lab = scipy.ndimage.label(sc.g, np.ones((3,3)))
        clusterSs += [(lab[0] == (i+1)).sum() for i in range(lab[1])]

    return clusterSs

def genFigs(clusterSs):
    clusterSs = array(clusterSs)
    
    rc('font', size=20)
    rc('legend', fontsize='medium')

    #num RyRs with log inset
    figure(figsize=(8,6))

    #a1 = axes([0.125, 0.125, 0.775, 0.8])
    a1 = axes()

    hT, bT = np.histogram(clusterSs, arange(0, 120, 1) + 0.5)

    #step(hstack((0,bT[:-1])), 100*hstack((0, hT))/float(len(omss)), 'b',lw=2, where='post')
    a1.plot(hstack((0, 0.5 + bT[:-1])), 100*hstack((0,hT))/float(len(clusterSs)), 'b',lw=2)

    #a1.plot([N.mean(), N.mean()], [0,20], 'b:')

    

    xlabel('Cluster Size [Number of RyRs]')
    ylabel('Percentage of clusters')
    ylim(0, 20)

    a2 = axes([.4, .4, .4, .4])
    a2.semilogy(hstack((0, 0.5 + bT[:-1])), 100*hstack((0,hT))/float(len(clusterSs)) + 1e-4, 'b',lw=2)

    def goalf(p):
       return -log(scipy.stats.expon.pdf(clusterSs[clusterSs > 2], scale=p)).sum()

    mu = scipy.optimize.fmin(goalf, 14)[0]

    print mu

    a2.plot(arange(120), 100*scipy.stats.expon.pdf(arange(120), scale=mu)*sum(clusterSs>2)/len(clusterSs) + 1e-4, 'r', lw=2)

    ylim(1e-2, 20)

    #a1.plot(hstack((0, 0.5 + bT[:-1])), 100*hstack((0,hT))/float(len(omss)), 'b',lw=2)
    #a1.fill_between(hstack((0, bT[:-1] + 0.5)), maximum(dm - ds, 0), dm + ds, facecolor='b', alpha=0.5)

    gcf().savefig('/home/david/Desktop/nryr_figs/nryr_per_cluster_sim.pdf')