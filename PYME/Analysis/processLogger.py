#!/usr/bin/python

##################
# <filename>.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################

import numpy as np

def _upstr(dt):
        """Force strings to a constant width dtype"""
        if dt.str.startswith('|S'):
            return '|S255'
        else: 
            return dt

def dictToRecarray(d):
        """Create a 1 entry recarray from a dictionay"""
        dt = np.dtype([(k, _upstr(np.array(v).dtype)) for k, v in d.items()])
        
        ra = np.zeros(1, dt)
        for k,v in d.items():
            ra[k] = v
        
        return ra
    

class ContextLayer(object):
    def __init__(self, parent, contextInfo):
        """A layer in the 'context stack'
        
        Parameters
        ----------
          parent  :      the preceeding layer
          contextInfo :  information about current context (a numpy record array, or a dictionary of key value pairs)


        """
        self.parent = parent
        if isinstance(contextInfo, np.ndarray):
            self.contextInfo = contextInfo
        elif isinstance(contextInfo, dict): #key value tuple
            self.contextInfo = dictToRecarray(contextInfo)
        else:        
            raise RuntimeError('Was expecting an ndarray or dictionary')
            
    
            
    
        
    def _reccat(self, reca, recb):
        """Concatenate 2 records"""
        #create a composite dtype
        #print reca.dtype.descr, recb.dtype.descr
        dt = np.dtype(reca.dtype.descr + recb.dtype.descr)
        #print dt
        
        # FIXME: this is gross (a workaround as old versions of np.concatenate didn't work with structured arrays). 
        # Is this fixed in more recent numpy versions???? 
        return np.fromstring(bytes(reca.data[:])+ bytes(recb.data[:]), dt)
        
    def AddRecord(self, table, record):
        self.parent.AddRecord(table, self._reccat(self.contextInfo, record))
        
        
class ContextManager(object):
    def __init__(self, backend):
        self.currentContext = backend
        
    def AddRecord(self, table, record):
        self.currentContext.AddRecord(table, record)
    
    def ExtendContext(self, contextInfo):
        self.currentContext = ContextLayer(self.currentContext, contextInfo)
            
    def PopContext(self):
        if isinstance(self.currentContext, ContextLayer):
            self.currentContext = self.currentContext.parent
            
    def SetBackend(self, backend):
        cc = self.currentContext
        if not isinstance(cc, ContextLayer):
            self.currentContext = backend
        else:
            bcc = cc
            while isinstance(cc, ContextLayer):
                cc = cc.parent
            
            bcc.parent = backend
            
class BackendBase(object):
    def AddRecord(self, table, record):
        print((table, record))
        
PL = ContextManager(BackendBase())

import tables        
class TablesBackend(BackendBase):
    def __init__(self, filename):
        
        self.tfile = tables.open_file(filename, 'a')
        
    def _split(self,p):
        """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
        everything after the final slash.  Either part may be empty."""
        i = p.rfind('/') + 1
        head, tail = p[:i], p[i:]
        if head and head != '/'*len(head):
            head = head.rstrip('/')
        return head, tail

        
    def AddRecord(self,table, record):
        if not table.startswith('/'):
            table = '/' + table
            
        try:
            tab = self.tfile.getNode(table)
            tab.append(record)
        except tables.NoSuchNodeError:
            h, t = self._split(table)
            self.tfile.create_table(h, t, record, createparents=True)
            
        self.tfile.flush()
