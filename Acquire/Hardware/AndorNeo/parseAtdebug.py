# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 12:25:44 2011

@author: dbad004
"""

def parseError(s):
    where, fcn, err = s.split('\t')
    
    filen = where.split('(')[0]
    
    return filen, where, fcn, err
    
def parseErrors(f):
    errs = {}
    i = 0
    
    s = f.readline()
    while not s == '':
        filen, where, fcn, err = parseError(s)

        errs[err] = (i, where, fcn)        
        
        i += 1 
        print i
        s = f.readline()
        
    return errs