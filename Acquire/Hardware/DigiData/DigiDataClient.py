import Pyro.core

def getDDClient(): #for client machine
    return Pyro.core.getProxyForURI('PYRONAME://DigiData')
