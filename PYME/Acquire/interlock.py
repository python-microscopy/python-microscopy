
import weakref
from PYME.util import webframework
import threading
import logging

logger = logging.getLogger(__name__)


class Interlock(object):
    def __init__(self, microscope, email_info=None):
        """ 

        Parameters
        ----------
        microscope : PYME.Acquire.microscope.Microscope
        email_info : dict
            sender : str
                email address to send from
            password : str
                password for sender
            receiver : str
                destination email address
        """
        self.scope = weakref.ref(microscope)
        self.email_info = email_info
    
    @webframework.register_endpoint('/kill', output_is_json=False)
    def kill(self, message=''):
        """
        kill the lasers

        Parameters
        ----------
        message : str
            something about why you needed to kill using the interlock
        """
        try:  # kill the lasers
            self.scope().turnAllLasersOff()
        except Exception as e:
            logger.error(str(e))
        
        logger.error('interlock activated: %s' % message)
        
        try:  # pause the action queue
            self.scope().action_manager.paused = True
        except Exception as e:
            logger.error(str(e))
        
        try:  # stop spooling
            self.scope().spoolController.StopSpooling()
        except Exception as e:
            logger.error(str(e))
        
        try:  # unlock focus lock
            self.scope().focus_lock.DisableLock()
        except Exception as e:
            logger.error(str(e))
        
        try:  # lower the objective
            self.scope().piFoc.MoveTo(0, self.scope().piFoc.GetMin(0))
        except Exception as e:
            logger.error(str(e))

        # call home
        if self.email_info is not None:
            try:
                import smtplib, ssl
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", context=context) as server:
                    server.login(self.email_info['sender'],
                                self.email_info['password'])
                    server.sendmail(self.email_info['sender'], 
                                    self.email_info['receiver'], message)
            except Exception as e:
                logger.error(str(e))

class InterlockServer(webframework.APIHTTPServer, Interlock):
    def __init__(self, microscope, email_info=None, port=9119, 
                 bind_address=''):
        """

        NOTE - this will likely not be around long, as it would be preferable to
        add the interlock endpoints to `PYME.acquire_server.AcquireHTTPServer` 
        and run a single server process on the microscope computer.

        Parameters
        ----------
        microscope: PYME.Acquire.microscope.Microscope
        email_info : dict
            sender : str
                email address to send from
            password : str
                password for sender
            receiver : str
                destination email address
        port : int
            port to listen on
        bind_address : str, optional
            specifies ip address to listen on, by default '' will bind to local 
            host.
        """
        webframework.APIHTTPServer.__init__(self, (bind_address, port))
        Interlock.__init__(self, microscope, email_info)
        
        self.daemon_threads = True
        self._server_thread = threading.Thread(target=self._serve)
        self._server_thread.daemon_threads = True
        self._server_thread.start()

    def _serve(self):
        try:
            logger.info('Starting Interlock server on %s:%s' % (self.server_address[0], 
                                                                self.server_address[1]))
            self.serve_forever()
        finally:
            logger.info('Shutting down Interlock server ...')
            self.shutdown()
            self.server_close()


class InterlockClient(object):
    """
    For systems running two PYMEAcquire instances, e.g. one for drift
    tracking, this allows you to define a scope.interlock in both systems to
    kill the lasers in the main PYMEAcquire.
    """
    def __init__(self, host='127.0.0.1', port=9119, name='interlock'):
        import requests

        self.host = host
        self.port = port
        self.name = name

        self.base_url = 'http://%s:%d' % (host, port)
        self._session = requests.Session()

    def kill(self, message=''):
        return self._session.get(self.base_url + '/kill?message=%s' % message)
