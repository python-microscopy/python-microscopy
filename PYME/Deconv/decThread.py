#!/usr/bin/python

##################
# decThread.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

#import dec
import threading
import time
import inspect
import ctypes

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class decThread(threading.Thread):
    def __init__(self, dec, data, lamb, num_iters=10, weights=1, bg=0):
        self.dec = dec
        self.data = data
        self.lamb = lamb
        self.num_iters = num_iters
        self.weights = weights
        self.bg = bg

        threading.Thread.__init__(self)

    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL : this function is executed in the context of the caller thread,
        to get the identity of the thread represented by this instance.
        """
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO : in python 2.6, there's a simpler way to do : self.ident ...

        raise AssertionError("could not determine the thread's id")

    def _raiseExc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(), socket.accept(), ...) the exception
        is simply ignored.

        If you are sure that your exception should terminate the thread, one way to ensure that
        it works is:
        t = ThreadWithExc( ... )
        ...
        t.raiseExc( SomeException )
        while t.isAlive():
            time.sleep( 0.1 )
            t.raiseExc( SomeException )

        If the exception is to be caught by the thread, you need a way to check that your
        thread has caught it.

        CAREFUL : this function is executed in the context of the caller thread,
        to raise an excpetion in the context of the thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )

    def kill(self):
        #self._raiseExc(RuntimeError)
        #while self.isAlive():
        #    time.sleep(0.1)
        #    self._raiseExc(RuntimeError)

        self.dec.loopcount = 1e9


    def run(self):
        self.res = self.dec.deconv(self.data, self.lamb, self.num_iters, self.weights, bg = self.bg).reshape(self.dec.shape)



