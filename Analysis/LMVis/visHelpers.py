#!/usr/bin/python
import scipy

def genEdgeDB(T):
    #make ourselves a quicker way of getting at edge info.
    edb = []
    for i in range(len(T.x)):
        edb.append(([],[]))

    for i in range(len(T.edge_db)):
        e = T.edge_db[i]
        edb[e[0]][0].append(i)
        edb[e[0]][1].append(e[1])
        edb[e[1]][0].append(i)
        edb[e[1]][1].append(e[0])


    return edb

def calcNeighbourDists(T):
    edb = genEdgeDB(T)

    di = scipy.zeros(T.x.shape)

    for i in range(len(T.x)):
        incidentEdges = T.edge_db[edb[i][0]]
        #neighbourPoints = edb[i][1]

        #incidentEdges = T.edge_db[edb[neighbourPoints[0]][0]]
        #for j in range(1, len(neighbourPoints)):
        #    incidentEdges = scipy.vstack((incidentEdges, T.edge_db[edb[neighbourPoints[j]][0]]))
        dx = scipy.diff(T.x[incidentEdges])
        dy = scipy.diff(T.y[incidentEdges])

        dist = (dx**2 + dy**2)

        di[i] = scipy.mean(scipy.sqrt(dist))

    return di
