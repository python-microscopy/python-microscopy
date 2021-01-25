
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
        self.scope().turnAllLasersOff()
        logger.error('shutting off lasers: %s' % message)
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
