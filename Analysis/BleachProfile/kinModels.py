def eimod(p, n):
    A, tau = p
    return A*tau*(exp(-(n-1)/tau) - exp(-n/tau))

def e2mod(p, t):
    A, tau, b = p
    return A*exp(-(t**.5)/tau) + b