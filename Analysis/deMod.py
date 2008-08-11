from scipy import *

def flF2(N,t, argS):
    Na,Nd, Nd2 = N
    Ab,Aad,Ada, Aad2, Ada2,I = argS
    dNa = -(Ab + Aad + Aad2)*I*Na + Ada*Nd
    dNd = Aad*I*Na - Ada*Nd + Ada2*Nd2
    dNd2 = Aad2*Nd - Ada2*Nd2
    return array([dNa, dNd, dNd2])


def flF2Mod(p, t):
    Na= p[0]
    b = p[1]
    Nd = 0
    Nd2 = 0
    Ab,Aad,Ada, Aad2, Ada2 = p[2:]
    I = 1
    a = integrate.odeint(flF2, [Na, Nd, Nd2], t, ((Ab,Aad,Ada, Aad2, Ada2,I),))
    return a[:,0] + b


def flFsq(N,t, argS, Ia):
    Na,Nd = N
    Ab,Aad,Ada = argS
    I = 0

    ind = floor(t/0.007)
    #print len(Ia)
    if ind < len(Ia):
        I = Ia[ind]
        #print ind
    dNa = -(Ab + Aad)*I*Na**2 + Ada*Nd
    dNd = Aad*I*Na**2 - Ada*Nd
    #print dNa
    return array([dNa, dNd])


def flFsq1(N,t, argS):
    Na,Nd = N
    Ab,Aad,Ada,I = argS

    dNa = -(Ab + Aad)*I*Na**2 + Ada*Nd
    dNd = Aad*I*Na**2 - Ada*Nd
    #print dNa
    return array([dNa, dNd])


def flFpow(N,t, argS):
    Na,Nd = N
    Ab,Aad,Ada,I, pow = argS

    dNa = - Aad*I*Na**pow -Ab*I*Na + Ada*Nd
    dNd = Aad*I*Na**pow - Ada*Nd
    #print dNa
    return array([dNa, dNd])
