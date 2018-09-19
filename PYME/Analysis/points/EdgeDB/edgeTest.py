#!/usr/bin/python

###############
# edgeTest.py
#
# Copyright David Baddeley, 2012
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
################
import numpy as np
from . import edges
from matplotlib import tri

#import segment
def test():
    x = np.random.rand(1e6)
    
    y = np.random.rand(1e6)
    
    
    
    T = tri.Triangulation(x,y)
    
    #ed = zeros((len(x)*1.5, 16), 'int32')
    
    #edges.addEdges(ed, T.edge_db)
    
    #print ed
    
    E = edges.EdgeDB(T)
    
    print("foo")
    
    #print E.edgeArray[len(x):, 0]
    print((E.edgeArray[0]))
    
    #ei= E.edgeArray[:len(x)]
    
    #print ei[ei['numIncidentEdges']>=7, :]
    #print ei
    
    print((E.getVertexEdgeLengths(5)))
    print((E.getVertexNeighbours(5)))
    
    print((E.getNeighbourDists()))
    
    #objects = segment.segment(E, .002)
    
    #print objects
    
    objects = edges.segment(E.edgeArray, .001)
    
    #print objects
