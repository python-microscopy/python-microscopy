import numpy as np
import sys
from PYME.Analysis.LMVis.visHelpers import genEdgeDB

def cluster(T):
    edb = genEdgeDB(T)
    nodeNum = -1*np.ones(T.x.shape)
    nodeInfo = sys.maxint*np.ones((len(T.x), 2))

    curNode = 0

    for i in range(len(T.x)):
        incidentEdges = T.edge_db[edb[i][0]]
        neighbourPoints = edb[i][1]

        dx = np.diff(T.x[incidentEdges])
        dy = np.diff(T.y[incidentEdges])

        dist = (dx**2 + dy**2)

        sI = argsort(dist)[::-1]

        #Do nearest neighbour - this MUST generate a new node
        iMin = np.argmin(np.sqrt(dist))
        d = dist[iMin]
        n = neighbourPoints[iMin]

        nodeInfo[curNode, 1] = d
        nodeNum[i] = curNode
        if n < i: #have already visited other point
            nn = nodeNum[n]

            if nodeInfo[nn, 1] >= d:
                nodeNum[n] = curNode
            else:
                nnp = nn
                while nodeInfo[nn, 1] < d: #traverse upwards until distance is larger
                    nnp = nn
                    nn = nodeInfo[nn, 0] #parent
                    
                nodeInfo[nnp, 0] = curNode

            nodeInfo[curNode, 0] = nn
                        
        else:
            nodeInfo[curNode,0] = len(T.x)-1 #last node
            nodeNum[n] = curNode

        curNode += 1


    return nodeNum, nodeInfo