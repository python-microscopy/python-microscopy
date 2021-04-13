import os

def load_template(filename):
    dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    with open(os.path.join(dir, filename)) as f:
        return f.read()
    
    
_server_instance = None

def set_server(server):
    """
    Set the server class (used in conjunction with `add_endpoints()` to let hardware drivers etc ...
    add endpoints to the server in a backwards compatible way. Such that they a) don't need a reference to the server
    and b) this works out to be a no-op for non web-ui instances.
    
    TODO - do this stuff in an InitGUI or similar step instead?
    
    Parameters
    ----------
    server

    Returns
    -------

    """
    global _server_instance
    _server_instance = server
    
def add_endpoints(cls, prefix, fail_if_no_server=False):
    if _server_instance is None:
        if fail_if_no_server:
            raise RuntimeError('No server registered')
    else:
        _server_instance.add_endpoints(cls, prefix)