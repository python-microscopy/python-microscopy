from pylab import *
import edges
from matplotlib import delaunay

import segment


x = rand(1e6)

y = rand(1e6)



T = delaunay.Triangulation(x,y)

#ed = zeros((len(x)*1.5, 16), 'int32')

#edges.addEdges(ed, T.edge_db)

#print ed

E = edges.EdgeDB(T)

print "foo"

#print E.edgeArray[len(x):, 0]
print E.edgeArray[0]

#ei= E.edgeArray[:len(x)]

#print ei[ei['numIncidentEdges']>=7, :]
#print ei

print E.getVertexEdgeLengths(5)
print E.getVertexNeighbours(5)

print E.getNeighbourDists()

#objects = segment.segment(E, .002)

#print objects

objects = edges.segment(E.edgeArray, .001)

#print objects