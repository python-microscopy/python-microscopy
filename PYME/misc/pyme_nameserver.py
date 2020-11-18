
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
            should be generated using the process name and 
            PYME.IO.FileUtils.nameUtils.get_service_name

        """
        raise NotImplementedError
        
    def unregister(self, name):
        """

        Parameters
        ----------
        name : str
            must be the same service name used to register
        """
        raise NotImplementedError

    def get_advertised_services(self):
        raise NotImplementedError
            
    def resolve(self, name):
        raise NotImplementedError
        
    def list(self, filterby=''):
        raise NotImplementedError
    
    def remove_inactive_services(self):
        raise NotImplementedError
