/*
##################
# edgeDB.c
#
# Copyright David Baddeley, 2010
# d.baddeley@auckland.ac.nz
#
# This file may NOT be distributed without express permision from David Baddeley
#
##################
 */


#include "Python.h"
//#include <complex.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#define RECORDSIZE 7
#define MAX_RECURSION_DEPTH 10000

typedef struct
{
    int numIncidentEdges;
    int endVertices[RECORDSIZE];
    float edgeLengths[RECORDSIZE];
    float meanEdgeLength;
    int nextRecordIndex;
} record;

typedef struct
{
    int numRecords;
    int nextFreeVertex;
    int numVertices;
    record * records;
} edgeDB;

typedef struct
{
    int dest;
    float *length;
} edgeInfo;


typedef struct listNode LISTNODE;

struct listNode
{
    int vertexNum;
    LISTNODE * next;
};

typedef struct
{
    LISTNODE * head;
    LISTNODE * tail;
} nodeList;

LISTNODE *pushNode(nodeList *list, int vertexNum)
{
    LISTNODE *newNode;
    newNode = malloc(sizeof(LISTNODE));
    if (newNode == NULL)
    {
        printf("merror\n");
        PyErr_Format(PyExc_RuntimeError, "Error allocating memory for node list");
    } else
    {
        //printf("hnn\n");
        newNode->vertexNum = vertexNum;
        newNode->next = NULL;


        //printf("hnn\n");
        if (list->tail != NULL)
            list->tail->next = newNode;
        
        list->tail = newNode;

        if (list->head ==NULL)
            list->head = newNode;
        
    }
}

int popNode(nodeList *list, int *vertNum)
{
    LISTNODE *node;
    
    node = list->head;
    
    if (node == NULL)
        return 0;
    
    vertNum[0] = node->vertexNum;
    list->head = node->next;
    
    free(node);

    return 1;
}


float getEdgeLength(edgeDB *edb, int vertexNum, int edgeNum)
{
    if (edgeNum < RECORDSIZE)
    {
        return edb->records[vertexNum].edgeLengths[edgeNum];
    } else
    {
        return getEdgeLength(edb, edb->records[vertexNum].nextRecordIndex, edgeNum - RECORDSIZE);
    }
}

int getEdgeDest(edgeDB *edb, int vertexNum, int edgeNum)
{
    if (edgeNum < RECORDSIZE)
    {
        return edb->records[vertexNum].endVertices[edgeNum];
    } else
    {
        return getEdgeDest(edb, edb->records[vertexNum].nextRecordIndex, edgeNum - RECORDSIZE);
    }
}

void getEdgeInfo(edgeDB *edb, int vertexNum, int edgeNum, edgeInfo *ei)
{
    if (edgeNum < RECORDSIZE)
    {
        ei->length = &edb->records[vertexNum].edgeLengths[edgeNum];
        ei->dest = edb->records[vertexNum].endVertices[edgeNum];
        //return ei;
    } else
    {
        getEdgeInfo(edb, edb->records[vertexNum].nextRecordIndex, edgeNum - RECORDSIZE, ei);
    }
}

void addEdge(edgeDB *edb, int vertexNum, int destVertex, float edgeLength)
{
    #define rec edb->records[vertexNum]
    
    //If the edge is easy to get ...
    if (rec.numIncidentEdges < RECORDSIZE)
    {
        rec.endVertices[rec.numIncidentEdges] = destVertex;
        rec.edgeLengths[rec.numIncidentEdges] = edgeLength;
    } else
    {
        
        if (rec.numIncidentEdges == RECORDSIZE)
        //Special case - need to create a new extension
        {
            if (edb->nextFreeVertex >= edb->numRecords)
            {
                //printf("f");
                PyErr_Format(PyExc_RuntimeError, "Overflowed our data space - try again with a biger buffer");
                return;
            }
            rec.nextRecordIndex = edb->nextFreeVertex;
            //printf("n: %d %d %d %d %d \n", edb.records[vertexNum].nextRecordIndex, edgeIndex, edb.records[vertexNum].numIncidentEdges,vertexNum,destVertex);
            edb->nextFreeVertex++;
        }

        addEdge(edb, rec.nextRecordIndex, destVertex, edgeLength);
    }

    rec.numIncidentEdges++;
    
}

static PyObject * addEdges(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *edgeArray =0;
    PyObject *deln_edges=0;

    //PyArrayObject *delnEdgeArray=0;

    //delnEdge *delnEdges;
    edgeDB edb;

    int startEdgeNum=0;
    int numEdges;
    int i = 0;

    int eStart;
    int eEnd;


    static char *kwlist[] = {"edgeArray", "triEdges", "startEdgeNum", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|i", kwlist,
         &edgeArray, &deln_edges, &startEdgeNum))
        return NULL;


    if (!PyArray_Check(edgeArray) || !PyArray_ISCONTIGUOUS(edgeArray))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data");
        return NULL;
    }

/*
    delnEdgeArray = (PyArrayObject *) PyArray_ContiguousFromObject(deln_edges, PyArray_INT32, 0, 1);
    if (delnEdgeArray == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad triangulation edges");
      return NULL;
    }
*/

    if (!PyArray_Check(deln_edges) || !PyArray_ISCONTIGUOUS(deln_edges))
    {
        PyErr_Format(PyExc_RuntimeError, "Bad triangulation edges");
        return NULL;
    }

    

    //delnEdges = (int*)PyArray_DATA(deln_edges);
    numEdges = PyArray_DIM(deln_edges, 0);

    edb.records = (record*)PyArray_DATA(edgeArray);
    edb.numRecords = PyArray_DIM(edgeArray, 0) - 1;

    //store free vertex info in last row (ugly)
    edb.nextFreeVertex = edb.records[edb.numRecords].nextRecordIndex;

    //printf("e: %d, %d, %d, %d\n", startEdgeNum, numEdges, startEdgeNum + numEdges, edb.numRecords);


    for (i=startEdgeNum; i < (startEdgeNum + numEdges); i++)
    {
        //printf("i: %d, %d, %d\n", i,delnEdges[i].start, delnEdges[i].end);
        eStart = *(int*)PyArray_GETPTR2(deln_edges, i, 0);
        eEnd = *(int*)PyArray_GETPTR2(deln_edges, i, 1);
        addEdge(&edb, eStart, eEnd, 0);
        addEdge(&edb, eEnd, eStart, 0);
    }

    //printf("foo\n");

    //store free vertex info in last row (ugly)
    edb.records[edb.numRecords].nextRecordIndex = edb.nextFreeVertex;

    //Py_DECREF(delnEdgeArray);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * calcEdgeLengths(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *edgeArray =0;
    PyObject *coordList=0;

    PyObject *coords[3];

    //delnEdge *delnEdges;
    edgeDB edb;

    edgeInfo ei;

    int startVertexNum=0;
    int stopVertexNum=-1;
    int numVertices=0;
    int numDims;
    int numEdges;
    //int dest;
    int i = 0;
    int numC=0;
    int j = 0;
    int k = 0;

    float d_j;
    float d_k;
    float x_i[3];
    double *coordsF[3];
    //Py_ssize_t j=0;

    
    static char *kwlist[] = {"edgeArray", "coords", "startVertexNum", "stopVertexNum", NULL};

    //printf("foo1\n");
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ii", kwlist,
         &edgeArray, &coordList, &startVertexNum, &stopVertexNum))
        return NULL;

    //printf("foo2\n");


    if (!PyArray_Check(edgeArray) || !PyArray_ISCONTIGUOUS(edgeArray))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data");
        return NULL;
    }

/*
    delnEdgeArray = (PyArrayObject *) PyArray_ContiguousFromObject(deln_edges, PyArray_INT32, 0, 1);
    if (delnEdgeArray == NULL)
    {
      PyErr_Format(PyExc_RuntimeError, "Bad triangulation edges");
      return NULL;
    }
*/

    if (!PySequence_Check(coordList))
    {
        PyErr_Format(PyExc_RuntimeError, "expecting an sequence  eg ... (xvals, yvals) or (xvals, yvals, zvals)");
        return NULL;
    }

    numDims = (int)PySequence_Length(coordList);

    if ((numDims < 1) || (numDims > 3))
    {
        PyErr_Format(PyExc_RuntimeError, "only support 1D, 2D, or 3D coordinates");
        return NULL;
    }

    for (numC=0; numC < numDims; numC++)
    {
        coords[numC] = PySequence_GetItem(coordList, (Py_ssize_t) numC);

        if (!PyArray_Check(coords[numC]) || (PyArray_NDIM(coords[numC]) != 1))
        {
            PyErr_Format(PyExc_RuntimeError, "coordinate should be a 1D numpy array");
            goto fail;
        }

        if (numC == 0) // the first dimension
        {
            numVertices = PyArray_DIM(coords[numC], 0);
        } else
        {
            if (PyArray_DIM(coords[numC], 0) != numVertices)
            {
                PyErr_Format(PyExc_RuntimeError, "coordinates should be the same length");
                goto fail;
            }
        }
        coordsF[numC] = PyArray_DATA(coords[numC]);
    }

    if (stopVertexNum == -1)
        stopVertexNum = numVertices;

    if ((startVertexNum <0) || (startVertexNum > numVertices))
    {
        PyErr_Format(PyExc_RuntimeError, "startVertex out of bounds");
        goto fail;
    }

    if ((stopVertexNum <0) || (stopVertexNum > numVertices))
    {
        PyErr_Format(PyExc_RuntimeError, "stopVertex out of bounds");
        goto fail;
    }


    edb.records = (record*)PyArray_DATA(edgeArray);
    edb.numRecords = PyArray_DIM(edgeArray, 0) - 1;

    //printf("foo\n");

    Py_BEGIN_ALLOW_THREADS;

    for (i = startVertexNum; i < stopVertexNum; i++)
    {
        for (j=0; j< numC; j++)
            x_i[j] = coordsF[j][i];
        
        numEdges = edb.records[i].numIncidentEdges;
        for (k=0; k< numEdges; k++)
        {
            d_k = 0;
            getEdgeInfo(&edb, i, k, &ei);

            for (j=0; j< numC;j++)
            {
                //printf("d: %d,%d,%d, %d, %d, sizeof_f: %d\n", i, k, j, ei.dest, numC, sizeof(float));
                //printf("f: %f,%f\n", coordsF[j][ei.dest],  x_i[j]);
                d_j = coordsF[j][ei.dest] - x_i[j];
                d_k += d_j*d_j;
            }
            
            ei.length[0] = sqrtf(d_k);
            edb.records[i].meanEdgeLength += ei.length[0];
        }
        edb.records[i].meanEdgeLength /= numEdges;
    }
    //printf("bar\n");

    Py_END_ALLOW_THREADS;


/*
    for (i=0; i< numC; i++)
    {
        Py_DECREF(coords[j]);
    }
*/

    Py_INCREF(Py_None);
    return Py_None;

fail:
/*
    for (i=0; i< numC; i++)
    {
        Py_DECREF(coords[j]);
    }
 */
    return NULL;

}

static PyObject * getVertexEdgeLengths(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *edgeArray =0;
    PyObject *incEdgesA = 0;
    
    edgeDB edb;

    float *incEdges = 0;

    int vertexNum;
    int i;
    int numEdges;
    int dims[2];


    static char *kwlist[] = {"edgeArray", "vertexNum", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi", kwlist,
         &edgeArray, &vertexNum))
        return NULL;


    if (!PyArray_Check(edgeArray) || !PyArray_ISCONTIGUOUS(edgeArray))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data");
        return NULL;
    }


    edb.records = (record*)PyArray_DATA(edgeArray);
    edb.numRecords = PyArray_DIM(edgeArray, 0) - 1;

    numEdges = edb.records[vertexNum].numIncidentEdges;

    //printf("n: %d, %d\n", numEdges, PyArray_INT32);

    dims[0] = numEdges;
    incEdgesA = PyArray_SimpleNew(1, dims, PyArray_FLOAT32);
    if (!incEdgesA)
    {
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for edges");
        return NULL;
    }

    incEdges = (float*)PyArray_DATA(incEdgesA);

    for (i =0; i < numEdges; i++)
    {
        incEdges[i] = getEdgeLength(&edb, vertexNum, i);
    }


    return (PyObject*) incEdgesA;
}


static PyObject * getVertexNeighbours(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *edgeArray =0;
    PyObject *neighbourA = 0;

    edgeDB edb;

    int *neighbours = 0;

    int vertexNum;
    int i;
    int numEdges;
    int dims[2];


    static char *kwlist[] = {"edgeArray", "vertexNum", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi", kwlist,
         &edgeArray, &vertexNum))
        return NULL;


    if (!PyArray_Check(edgeArray) || !PyArray_ISCONTIGUOUS(edgeArray))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data");
        return NULL;
    }


    edb.records = (record*)PyArray_DATA(edgeArray);
    edb.numRecords = PyArray_DIM(edgeArray, 0) - 1;

    numEdges = edb.records[vertexNum].numIncidentEdges;

    //printf("n: %d\n", numEdges);

    dims[0] = numEdges;
    neighbourA = PyArray_SimpleNew(1, dims, PyArray_INT32);
    if (!neighbourA)
    {
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for edges");
        return NULL;
    }

    neighbours = (int*)PyArray_DATA(neighbourA);

    for (i =0; i < numEdges; i++)
    {
        neighbours[i] = getEdgeDest(&edb, vertexNum, i);
    }


    return (PyObject*) neighbourA;
}


int collectConnected(edgeDB *edb, int vertexNum, int* objects, float lenThresh, int objectNum, int *nVisited, int *recDepth)
{
    edgeInfo ei;
    int numEdges;
    int i;

    (*recDepth)++;
    if (*recDepth > MAX_RECURSION_DEPTH)
    {
        //PyErr_Format(PyExc_RuntimeError, "Exceeded max recursion depth");
        printf("Warning: max recursion depth reached - objects might be artificially divided\n");
        return -1;
    }

    numEdges = edb->records[vertexNum].numIncidentEdges;

    for (i = 0; i < numEdges; i++)
    {
        getEdgeInfo(edb, vertexNum, i, &ei);

        if ((objects[ei.dest] == 0) && (*ei.length < lenThresh))
        {
            //printf("i: %d\tnVis: %d\toN: %d\n", vertexNum, nVisited[0], objectNum);

            objects[ei.dest] = objectNum;
            (*nVisited) ++;

            if (collectConnected(edb, ei.dest, objects, lenThresh, objectNum, nVisited, recDepth) == -1)
                return -1;
        }
    }

}

int collectConnectedNR(edgeDB *edb, int vertexNum, int* objects, float lenThresh, int objectNum, int *nVisited)
{
    edgeInfo ei;
    int numEdges;
    int i;
    int vert;

    //printf("foo\n");

    nodeList nodes;
    nodes.head = NULL;
    nodes.tail = NULL;
    
    if (pushNode(&nodes, vertexNum) == NULL)
    {
        //printf("foo\n");
        while (popNode(&nodes, &vert));
        return -1;
    }

    //printf("foo\n");

    while (popNode(&nodes, &vert))
    {
        //printf("v %d\n", vert);
        numEdges = edb->records[vert].numIncidentEdges;

        for (i = 0; i < numEdges; i++)
        {
            getEdgeInfo(edb, vert, i, &ei);

            if ((objects[ei.dest] == 0) && (*ei.length < lenThresh))
            {
                //printf("i: %d\tnVis: %d\toN: %d\n", vertexNum, nVisited[0], objectNum);

                objects[ei.dest] = objectNum;
                (*nVisited) ++;

                if (pushNode(&nodes, ei.dest) == NULL)
                {
                    while (popNode(&nodes, &vert));
                    return -1;
                }
            }
        }
    }

    return 0;

}

static PyObject * segment(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *edgeArray =0;
    PyObject *objectsArray = 0;

    edgeDB edb;

    int *objects = 0;

    float lenThresh = 0;
    int minSize = 3;

    int vertexNum = 0;
    int objectNum = 1;
    int nVisited = 0;
    int dims[2];
    int i = 0;
    int recDepth = 0;


    static char *kwlist[] = {"edgeArray", "lenThresh", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Of", kwlist,
         &edgeArray, &lenThresh))
        return NULL;


    if (!PyArray_Check(edgeArray) || !PyArray_ISCONTIGUOUS(edgeArray))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data");
        return NULL;
    }

    //printf("foo\n");


    edb.records = (record*)PyArray_DATA(edgeArray);
    edb.numRecords = PyArray_DIM(edgeArray, 0) - 1;
    
    //store num vertex info in last row (ugly)
    edb.numVertices = edb.records[edb.numRecords].numIncidentEdges;


    dims[0] = edb.numVertices;
    objectsArray = PyArray_SimpleNew(1, dims, PyArray_INT32);
    if (!objectsArray)
    {
        PyErr_Format(PyExc_RuntimeError, "Error allocating array for objects");
        return NULL;
    }

    objects = (int*)PyArray_DATA(objectsArray);

    for (i=0; i < edb.numVertices; i++)
        objects[i] = 0;

    //i = 0;

    while ((nVisited < edb.numVertices) && (vertexNum < edb.numVertices))
    //there are unvisited vertices
    {
         //skip over any vetices we've already visited
        while ((objects[vertexNum] > 0)  && (vertexNum < edb.numVertices))
        {
            vertexNum++;
        }

        objects[vertexNum] = objectNum;
        nVisited++;

        //collectConnected(&edb, vertexNum, objects, lenThresh, objectNum, &nVisited, &recDepth);
        recDepth = 0;
        collectConnected(&edb, vertexNum, objects, lenThresh, objectNum, &nVisited, &recDepth);
/*
        if (collectConnected(&edb, vertexNum, objects, lenThresh, objectNum, &nVisited, &recDepth) == -1)
            goto fail;
*/

        objectNum++;
        //i++;
    }


    return (PyObject*) objectsArray;

fail:
    return NULL;
}




static PyMethodDef edgeDBMethods[] = {
    {"addEdges",  addEdges, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"segment",  segment, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"calcEdgeLengths",  calcEdgeLengths, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"getVertexEdgeLengths",  getVertexEdgeLengths, METH_VARARGS | METH_KEYWORDS,
    ""},
    {"getVertexNeighbours",  getVertexNeighbours, METH_VARARGS | METH_KEYWORDS,
    ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION>=3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "edgeDB",     /* m_name */
        "major refactoring of the Analysis tree",  /* m_doc */
        -1,                  /* m_size */
        edgeDBMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_edgeDB(void)
{
	PyObject *m;
    // m = PyModule_Create("edgeDB", edgeDBMethods);
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}

#else
PyMODINIT_FUNC initedgeDB(void)
{
    PyObject *m;

    m = Py_InitModule("edgeDB", edgeDBMethods);
    import_array()

    //SpamError = PyErr_NewException("spam.error", NULL, NULL);
    //Py_INCREF(SpamError);
    //PyModule_AddObject(m, "error", SpamError);
}
#endif
