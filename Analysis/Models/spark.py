from scipy import *

def sparkMod(p, X,T):
    maxP, Td, Tr, x0, t0, sig0, b = p

    f = ((T - t0) > 0)*maxP*exp(-(T-t0)/Td)*(1 - exp(-(T-t0)/Tr))*exp(-((X-x0)**2)/(2*(sig0*(1 + (((T - t0) > 0)*(T - t0))**.25)**2)))+ b

    return f
