from scipy.stats import rv_continuous
from scipy.special import erf
from numpy import *

def sig_N(N, s, d):
    d = atleast_1d(d)
    s = atleast_1d(s)
    n = arange(N)
    #print n.shape
    #print d.shape
    u = (n[newaxis,:]*d[:,newaxis]).mean(1)
    #print u.shape
    return sqrt(((n[newaxis,:]*d[:,newaxis])**2 - 2*n[newaxis,:]*d[:,newaxis]*u[:,newaxis] + s[:,newaxis]**2 + u[:,newaxis]**2).mean(1))

class multiNorm_gen(rv_continuous):
    def __init__(self, N, **args):
        self.numargs = 4 + N
        rv_continuous.__init__(self, **args)
    def _cdf(self, x, spr, sprn,sx,d, *args):
        n_pks = len(args) + 1
        fN = [maximum(ones(args[0].shape) - array(args).sum(0), 0)] + list(args)

        #print n_pks
        #print fN

        #print x
        #print spr
        #print sx
        #print d
        
        p = fN[0]*(1 + erf((x - sig_N(1, sx,d))/sqrt(2*(spr**2 + sprn**2))))

        for i in range(1, n_pks):
            p = p + fN[i]*(1 + erf((x - sig_N(i+1, sx,d))/(sqrt(2*(spr**2+((i+1)*sprn)**2)))))

        return p/2

    def _pdf(self, x, spr,sprn,sx,d, *args):
        n_pks = len(args) + 1
        fN = [maximum(ones(args[0].shape) - array(args).sum(0), 0)] + list(args)

        #print sum(fN, 0)

        #print n_pks
        #print fN

        #print x
        #print spr
        #print sx
        #print d

        c = fN[0]*exp(-(x - sig_N(1, sx,d))**2/(2*(spr**2 + sprn**2)))/(sqrt(2*pi*(spr**2 + sprn**2)))

        for i in range(1, n_pks):
            c = c + fN[i]*exp(-(x - sig_N(i+1, sx,d))**2/(2*(spr**2 + ((i+1)*sprn)**2)))/(sqrt(2*pi*(spr**2 + (((i+1)*sprn)**2))))

        return c/sum(fN, 0)


multiNorm1 = multiNorm_gen(1,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")

multiNorm2 = multiNorm_gen(2,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")

multiNorm3 = multiNorm_gen(3,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")
                          
                          
multiNorm4 = multiNorm_gen(4,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")
                          
multiNorm5 = multiNorm_gen(5,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")
                          
                          
multiNorm6 = multiNorm_gen(6,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")
                          
multiNorm7 = multiNorm_gen(7,name='multinorm',longname="",
                          shapes="spr,sx,d,*Nf", extradoc="")
