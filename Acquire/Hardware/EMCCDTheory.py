from numpy import *

def FSquared(M, N):
    '''Excess noise factor as a function of multiplication (M) and number of
       Stages (N)

       From Robins and Hadwen, 2002, IEEE Trans. Electon Dev.'''

    return 2*(M-1)*M**(-(float(N)+1)/float(N)) + 1/M

def SNR(S, ReadNoise, M, N):
    return S/sqrt((ReadNoise/M)**2 + FSquared(M, N)*S)