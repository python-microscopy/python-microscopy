"""A single common terminal for all websockets.
"""
import tornado.web
# This demo requires tornado_xstatic and XStatic-term.js
#import tornado_xstatic

from terminado import TermSocket

from collections import deque
import signal
import os
import sys
import time
import threading
import logging
import io
import errno

import notebook

from tornado import gen

class TTYDummy(object):
    encoding='UTF-8'
    errors=None
    def __init__(self, filelike, fileno=None):
        self.file = filelike
        if fileno is None:
            self._fileno = self.file.fileno()
        else:
            self._fileno = fileno

    def isatty(self):
        return True

    def fileno(self):
        return self._fileno

    def write(self, text):
        if not isinstance(text, bytes):
            text = text.encode('utf8')
        self.file.write(text)
        self.file.flush()

    def read(self, size=1024):
        """Read and return at most ``size`` bytes from the pty.

        Can block if there is nothing to read. Raises :exc:`EOFError` if the
        terminal was closed.

        Unlike Pexpect's ``read_nonblocking`` method, this doesn't try to deal
        with the vagaries of EOF on platforms that do strange things, like IRIX
        or older Solaris systems. It handles the errno=EIO pattern used on
        Linux, and the empty-string return used on BSD platforms and (seemingly)
        on recent Solaris.
        """
        try:
            s = self.file.read1(size)
        except (OSError, IOError) as err:
            if err.args[0] == errno.EIO:
                # Linux-style EOF
                self.flag_eof = True
                raise EOFError('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            # BSD-style EOF (also appears to work on recent Solaris (OpenIndiana))
            self.flag_eof = True
            raise EOFError('End Of File (EOF). Empty string style platform.')
    
        return s.decode()

    def readline(self):
        """Read one line from the pseudoterminal, and return it as unicode.

        Can block if there is nothing to read. Raises :exc:`EOFError` if the
        terminal was closed.
        """
        try:
            s = self.file.readline()
        except (OSError, IOError) as err:
            if err.args[0] == errno.EIO:
                # Linux-style EOF
                self.flag_eof = True
                raise EOFError('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            # BSD-style EOF (also appears to work on recent Solaris (OpenIndiana))
            self.flag_eof = True
            raise EOFError('End Of File (EOF). Empty string style platform.')
    
        return s.decode()

    def __getattr__(self, name):
        return getattr(self.file, name)


class IPY(object):
    def __init__(self, user_ns=None, user_module=None):
        self._fd_in_r, self._fd_in_w = os.pipe()
        self.fr_in = TTYDummy(io.BufferedReader(open(self._fd_in_r, 'rb', buffering=0)))
        self.fw_in = TTYDummy(open(self._fd_in_w, 'wb', buffering=0))

        self._fd_out_r, self._fd_out_w = os.pipe()
        self.fr_out = TTYDummy(io.BufferedReader(open(self._fd_out_r, 'rb', buffering=0)))
        self.fw_out = TTYDummy(open(self._fd_out_w, 'wb', buffering=0))
        
        self._user_ns = user_ns
        self._user_module = user_module
        
    def _start_ipython(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._old_stdin = sys.stdin
        
        try:
            print('Starting IPython')
            print('ns:', self._user_ns)
            sys.stdout = self.fw_out
            sys.__stdout__ = self.fw_out
            sys.stdin = self.fr_in
            sys.__stdin__ = self.fr_in
            #sys.stderr = self.fw_out
            
            import IPython
            from traitlets.config import get_config
            c = get_config()
            c.InteractiveShellEmbed.colors = "Linux"
            IPython.embed(config=c, user_ns=self._user_ns, user_module=self._user_module)
        finally:
            sys.stdout = self._old_stdout
            sys.__stdout__ = self._old_stdout
            sys.stdin = self._old_stdin
            sys.__stdin__  = self._old_stdin
            sys.stderr = self._old_stderr
            
    def start_ipython(self, *args, **kwargs):
        self._ipy_thread = threading.Thread(target=self._start_ipython, args=args, kwargs=kwargs)
        self._ipy_thread.start()
        
    def kill(self, signal=None):
        try:
            # try to shut down interpreter cleanly
            self.fw_in.write("exit()\n")
            self._ipy_thread.join(1)
        finally:
            self.fr_in.close()
            self.fw_in.close()
    
            self.fr_out.close()
            self.fw_out.close()
        

class IPYWithClients(object):
    def __init__(self, ipy):
        self.ipy = ipy
        self.clients = []
        # Store the last few things read, so when a new client connects,
        # it can show e.g. the most recent prompt, rather than absolutely
        # nothing.
        self.read_buffer = deque([], maxlen=10)
        self.preopen_buffer = deque([])
        
    @property
    def ptyproc(self):
        return self.ipy.fw_in
    
    def resize_to_smallest(self):
        """Set the terminal size to that of the smallest client dimensions.

        A terminal not using the full space available is much nicer than a
        terminal trying to use more than the available space, so we keep it
        sized to the smallest client.
        """
        #FIXME
        pass
    
    def kill(self, sig=None):
        """Send a signal to the process in the pty"""
        self.ipy.kill()
    
    def killpg(self, sig=None):
        """Send a signal to the process group of the process in the pty"""
        self.kill()
    
    #@gen.coroutine
    def terminate(self, force=False):
        self.kill()

class IPYTermManager(object):
    """Base class for a terminal manager."""
    
    def __init__(self, server_url="", ioloop=None, user_ns=None, user_module=None):
        from concurrent import futures
        self.server_url = server_url
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.WARNING)
        
        self.ptys_by_fd = {}

        self._blocking_io_executor_is_external = False
        self.blocking_io_executor = futures.ThreadPoolExecutor(max_workers=1)
        
        if ioloop is not None:
            self.ioloop = ioloop
        else:
            import tornado.ioloop
            self.ioloop = tornado.ioloop.IOLoop.instance()

        self.terminal = None
        self._user_ns = user_ns
        self._user_module = user_module
    
    def new_terminal(self, **kwargs):
        """Make a new terminal, return a :class:`PtyWithClients` instance."""

        ipy = IPY(self._user_ns, self._user_module)
        #ipy.start_ipython()
        return IPYWithClients(ipy)
    
    def start_reading(self, ptywclients):
        """Connect a terminal to the tornado event loop to read data from it."""
        fd = ptywclients.ipy._fd_out_r
        self.ptys_by_fd[fd] = ptywclients
        self.ioloop.add_handler(fd, self.pty_read, self.ioloop.READ)
    
    def on_eof(self, ptywclients):
        """Called when the pty has closed.
        """
        # Stop trying to read from that terminal
        fd = ptywclients.ptyproc._fd_out_r
        self.log.info("EOF on FD %d; stopping reading", fd)
        del self.ptys_by_fd[fd]
        self.ioloop.remove_handler(fd)
        
        # This closes the fd, and should result in the process being reaped.
        ptywclients.ptyproc.close()
    
    def pty_read(self, fd, events=None):
        """Called by the event loop when there is pty data ready to read."""
        ptywclients = self.ptys_by_fd[fd]
        try:
            s = ptywclients.ipy.fr_out.read(65536)
            client_list = ptywclients.clients
            self.log.debug(s)
            ptywclients.read_buffer.append(s)
            if not client_list:
                # No one to consume our output: buffer it.
                ptywclients.preopen_buffer.append(s)
                return
            
            for client in client_list:
                client.on_pty_read(s)
        except EOFError:
            self.on_eof(ptywclients)
            for client in ptywclients.clients:
                client.on_pty_died()
    
    def get_terminal(self, url_component=None):
        """Override in a subclass to give a terminal to a new websocket connection

        The :class:`TermSocket` handler works with zero or one URL components
        (capturing groups in the URL spec regex). If it receives one, it is
        passed as the ``url_component`` parameter; otherwise, this is None.
        """
        if self.terminal is None:
            self.terminal = self.new_terminal()
            self.start_reading(self.terminal)
            self.terminal.ipy.start_ipython()
            
        return self.terminal
    
    def client_disconnected(self, websocket):
        """Override this to e.g. kill terminals on client disconnection.
        """
        pass
    
    @gen.coroutine
    def shutdown(self):
        yield self.kill_all()
        if not self._blocking_io_executor_is_external:
            self.blocking_io_executor.shutdown(wait=False, cancel_futures=True)
    
    @gen.coroutine
    def kill_all(self):
        futures = []
        for term in self.ptys_by_fd.values():
            futures.append(term.terminate(force=True))
        # wait for futures to finish
        for f in futures:
            yield f

class TerminalPageHandler(tornado.web.RequestHandler):
    def get(self):
        return self.render("xtermpage.html", static=self.static_url,
                           #xstatic=self.application.settings['xstatic_url'],
                           ws_url_path="/websocket")



def create_ipy_server(user_ns=None, user_module=None):
    # demo stuff
    import os.path
    import webbrowser
    import tornado.ioloop
    import terminado
    
    STATIC_DIR = os.path.join(os.path.dirname(terminado.__file__), "_static")
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
      
    #end demo stuff

    try:
        # notebook < 6.5
        ipstatic = notebook.DEFAULT_STATIC_FILES_PATH
    except AttributeError:
        # notebook >= 6.5
        import nbclassic
        ipstatic = nbclassic.DEFAULT_STATIC_FILES_PATH


    term_manager = IPYTermManager(user_ns=user_ns, user_module=user_module)
    handlers = [
                (r"/websocket", TermSocket,
                     {'term_manager': term_manager}),
                (r"/", TerminalPageHandler),
                #(r"/xstatic/(.*)", tornado_xstatic.XStaticFileHandler,
                #     {'allowed_modules': ['termjs']}),
                (r"/ipstatic/(.*)", tornado.web.StaticFileHandler, {'path' : ipstatic})
               ]
    app = tornado.web.Application(handlers, static_path=STATIC_DIR,
                      template_path=TEMPLATE_DIR,
                      #xstatic_url = tornado_xstatic.url_maker('/xstatic/')
                                  )
    app.listen(8765, 'localhost')

    loop = tornado.ioloop.IOLoop.instance()
    #loop.add_callback(webbrowser.open, "http://localhost:8765/") #launch terminal
    try:
        loop.start()
    except KeyboardInterrupt:
        print(" Shutting down on SIGINT")
    finally:
        term_manager.shutdown()
        loop.close()
        
    
def launch_ipy_server_thread(user_ns=None, user_module=None):
    def _evt_loop_and_launch():
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        create_ipy_server(user_ns, user_module)
        
    t = threading.Thread(target=_evt_loop_and_launch)
    t.start()
    return t
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    #create_ipy_server()
    launch_ipy_server_thread(user_ns={'foo':123})