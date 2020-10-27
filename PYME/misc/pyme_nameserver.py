
class BaseNS(object):
    def __init__(self, protocol = '_pyme-pyro'):
        self._protocol = protocol
                           
    def register(self, name, URI):
        raise NotImplementedError
        
    def register_service(self, name, address, port, desc={}):
        """

        Parameters
        ----------
        name : str
            information about the process, typically including OS process ID.

        Returns
        ------
        str
            service_name, as registered
        """
        raise NotImplementedError
        
    def unregister(self, name):
        raise NotImplementedError

    def get_advertised_services(self):
        raise NotImplementedError
            
    def resolve(self, name):
        raise NotImplementedError
        
    def list(self, filterby=''):
        raise NotImplementedError
    
    def remove_inactive_services(self):
        raise NotImplementedError
