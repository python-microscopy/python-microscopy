import scipy

def genEdgeDB(T):
    #make ourselves a quicker way of getting at edge info.
    edb = []
    #edb = numpy.zeros((len(T.x), 2), dtype='O')
    cdef int i
    for i in range(len(T.x)):
        edb.append(([],[]))
        #edb[i] = ([],[])

    for i in range(len(T.edge_db)):
        e0, e1 = T.edge_db[i]
        edbe0 = edb[e0]
        edbe1 = edb[e1]
        edbe0[0].append(i)
        edbe0[1].append(e1)
        edbe1[0].append(i)
        edbe1[1].append(e0)


    return edb

def calcNeighbourDists(T):
    edb = genEdgeDB(T)

    di = scipy.zeros(T.x.shape)

    cdef int i

    for i in range(len(T.x)):
        incidentEdges = T.edge_db[edb[i][0]]
        #neighbourPoints = edb[i][1]

        #incidentEdges = T.edge_db[edb[neighbourPoints[0]][0]]
        #for j in range(1, len(neighbourPoints)):
        #    incidentEdges = scipy.vstack((incidentEdges, T.edge_db[edb[neighbourPoints[j]][0]]))
        #dx = scipy.diff(T.x[incidentEdges])
        #dy = scipy.diff(T.y[incidentEdges])

        xv = T.x[incidentEdges]
        dx = xv[:,1] - xv[:,0]
        yv = T.y[incidentEdges]
        dy = yv[:,1] - yv[:,0]

        dist = (dx**2 + dy**2)

        dist = scipy.sqrt(dist)

        #di[i] = scipy.mean(scipy.sqrt(dist))
        di[i] = scipy.mean(dist)

    return di