import requests


class AcquireClient(object):
    """
    Client for the Acquire server. Offers a pythonic interface to the PYMEAcquire REST API.
    """

    def __init__(self, url='127.0.0.1', port=8999):
        self.url = url
        self.port = port
        self.base_url = 'http://{}:{}'.format(self.url, self.port)
        self._state = None


    def _poll_state(self):
        """
        Polls the state of the server. Uses the long-polling endpoint.
        """
        while True:
            self._state = requests.get(self.base_url + '/scope_state_longpoll').json()

    def _start_polling(self):
        """
        Starts polling the server state.
        """
        import threading
        t = threading.Thread(target=self._poll_state)
        t.daemon = True
        t.start()

    @property
    def state(self):
        """
        Returns the current state of the server (as given by long-polling). Starts polling
        the first time it is called.
        """
        if self._state is None:
            self._state = self._get_scope_state() # get initiial state
            self._start_polling()
        
        return self._state


    def _get_scope_state(self):
        """
        Returns the current state of the scope as a dictionary.
        """
        return requests.get(self.base_url + '/get_scope_state').json()
    
    def update_scope_state(self, state:dict):
        """
        Updates the scope state with the provided dictionary.
        """
        requests.post(self.base_url + '/update_scope_state', json=state)

    def start_spooling(self, filename='', preflight_mode='abort', settings={}):
        """
        Starts spooling images to disk. If filename is not provided, the default filename will be used.
        """
        
        requests.post(self.base_url + f'/spool_controller/start_spooling?filename={filename}&preflight_mode={preflight_mode}', json=settings)

        return self.spooling_finished

    def spooling_info(self):
        """
        Returns information about the current spooling.

        TODO - Use the long-polling endpoint???
        """
        return requests.get(self.base_url + '/spool_controller/info').json()
    
    def spooling_finished(self):
        """
        Returns True if the spooling is finished, False otherwise.
        """
        return self.spooling_info()['status']['spool_complete']

    