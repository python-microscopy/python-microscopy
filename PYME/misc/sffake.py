# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:00:17 2015

@author: david
"""

class sffake:
    def __init__(self, val):
        self.val = val

    def ev(self, x, y):
        return self.val

    def __call__(self, x, y):
        return self.val + 0*x + 0*y