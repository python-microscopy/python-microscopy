#include <Python.h>
//#include "structmember.h"

#include "./axon/Common/axodefn.h"
#include "./axon/AxAbfFio32/abffiles.h"
#include "./axon/AxAbfFio32/abfheadr.h"

#include "numpy/arrayobject.h"

typedef struct
{
  PyObject_HEAD
  int hFile;
  ABFFileHeader FH;
  UINT uMaxSamples;
  DWORD dwMaxEpi;
} abfFile;

static void abfFile_dealloc(abfFile* self)
{
  int nError;

  if (self->hFile != NULL)
    {
      if (!ABF_Close(self->hFile,&nError)) 
	{
	  PyErr_SetString(PyExc_RuntimeError, "Failed to close abf file\n");
	}
      
      self->hFile = NULL;
    }
}

static PyObject * abfFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    abfFile *self;

    self = (abfFile *)type->tp_alloc(type, 0);

    self->hFile = NULL;

    return (PyObject *)self;
}

static int abfFile_init(abfFile *self, PyObject *args)
{
    const char *filename=NULL;
    int nError;
    
    if (! PyArg_ParseTuple(args, "s",&filename))
        return -1; 

    if (filename) {
       if (!ABF_ReadOpen(filename, &self->hFile, ABF_DATAFILE, &self->FH,
			 &self->uMaxSamples, &self->dwMaxEpi, &nError))
	 {
	   PyErr_SetString(PyExc_TypeError, 
		    "Error opening file");
	 }
    }

    return 0;
}

/* static PyMemberDef abfFile_members[] = { */
/*     {"number", T_INT, offsetof(abfFile, number), 0, */
/*      "noddy number"}, */
/*     {NULL}  /\* Sentinel *\/ */
/* }; */


static PyObject * abfFile_getNumEpisodes(abfFile *self, void *closure)
{
  return Py_BuildValue("i", self->FH.lActualEpisodes);
}


static PyObject * abfFile_getNumChannels(abfFile *self, void *closure)
{   
  return Py_BuildValue("i", self->FH.nADCNumChannels);
}



static PyGetSetDef abfFile_getseters[] = {
    {"NumEpisodes", 
     (getter)abfFile_getNumEpisodes, NULL,
     "Number of episodes",
     NULL},
    {"NumChannels", 
     (getter)abfFile_getNumChannels, NULL,
     "Number of channels",
     NULL},
    {NULL}  /* Sentinel */
};

static PyObject * abfFile_GetData(abfFile* self, PyObject * args)
{
  PyArrayObject* data;
  //PyObject *result;
  unsigned int EpisodeNum, ChanNum;
  unsigned int NumSamples, NumSamplesRead;
  int iNumSamp;
  int nError;

  if (! PyArg_ParseTuple(args, "ii",&EpisodeNum, &ChanNum))
        return NULL;

  //Episodes are 1-indexed in the abf API
  EpisodeNum += 1;

  //Check our arguments
  if (EpisodeNum > self->FH.lActualEpisodes || (DWORD)EpisodeNum < 1)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Episode Number");
      return NULL;
    }

  if (ChanNum >= self->FH.nADCNumChannels || ChanNum < 0)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Channel Number");
      return NULL;
    }



  if (!ABF_GetNumSamples(self->hFile,&self->FH, (DWORD)EpisodeNum,&NumSamples,&nError)) 
    {
      PyErr_SetString(PyExc_TypeError, "Error getting number of samples");
      return NULL;
    }

  iNumSamp = (int) NumSamples;
  
  //Allocate a new PyArray object
  data = (PyArrayObject*) PyArray_SimpleNew(1,&iNumSamp,PyArray_FLOAT);

  
  if (!ABF_ReadChannel(self->hFile,&self->FH,self->FH.nADCSamplingSeq[ChanNum],
		       EpisodeNum,(float*)PyArray_DATA(data),&NumSamplesRead,
		       &nError))
    {
      PyErr_SetString(PyExc_TypeError, "Error reading samples");
      return NULL;
    }

  if (NumSamples!=NumSamplesRead) 
    {
      PyErr_SetString(PyExc_TypeError, "Error reading samples");
      return NULL;
    }
    
  return (PyObject*) data;
}

/*
//Get the digital outputs
static PyObject * abfFile_GetDigitalWaveform(abfFile* self, PyObject * args)
{
  PyArrayObject* data;
  //PyObject *result;
  unsigned int EpisodeNum, ChanNum;
  unsigned int NumSamples, NumSamplesRead;
  int iNumSamp;
  int nError;

  if (! PyArg_ParseTuple(args, "ii",&EpisodeNum, &ChanNum))
        return NULL;

  //Check our arguments
  if (EpisodeNum > self->FH.lActualEpisodes || (DWORD)EpisodeNum < 1)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Episode Number");
      return NULL;
    }

  if (ChanNum >= self->FH.nADCNumChannels || ChanNum < 0)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Channel Number");
      return NULL;
    }



  if (!ABF_GetNumSamples(self->hFile,&self->FH, (DWORD)EpisodeNum,&NumSamples,&nError)) 
    {
      PyErr_SetString(PyExc_TypeError, "Error getting number of samples");
      return NULL;
    }

  iNumSamp = (int) NumSamples;
  
  //Allocate a new PyArray object
  data = (PyArrayObject*) PyArray_SimpleNew(1,&iNumSamp,PyArray_ULONG);

  
  if (!ABFH_GetDigitalWaveform(&self->FH,self->FH.nADCSamplingSeq[ChanNum],
		       EpisodeNum,(DWORD*)PyArray_DATA(data),
		       &nError))
    {
      PyErr_SetString(PyExc_TypeError, "Error reading samples");
      return NULL;
    }
    
  return (PyObject*) data;
  }*/


/*static PyObject * abfFile_GetAnalogWaveform(abfFile* self, PyObject * args)
{
  PyArrayObject* data;
  //PyObject *result;
  unsigned int EpisodeNum, ChanNum;
  unsigned int NumSamples, NumSamplesRead;
  int iNumSamp;
  int nError;

  if (! PyArg_ParseTuple(args, "ii",&EpisodeNum, &ChanNum))
        return NULL;

  //Check our arguments
  if (EpisodeNum > self->FH.lActualEpisodes || (DWORD)EpisodeNum < 1)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Episode Number");
      return NULL;
    }

  if (ChanNum >= ABF_WAVEFORMCOUNT || ChanNum < 0)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Channel Number");
      return NULL;
    }



  if (!ABF_GetNumSamples(self->hFile,&self->FH, (DWORD)EpisodeNum,&NumSamples,&nError)) 
    {
      PyErr_SetString(PyExc_TypeError, "Error getting number of samples");
      return NULL;
    }

  iNumSamp = (int) NumSamples;
  
  //Allocate a new PyArray object
  data = (PyArrayObject*) PyArray_SimpleNew(1,&iNumSamp,PyArray_FLOAT);

  
  if (!ABF_GetWaveform(self->hFile,&self->FH,ChanNum,
		       EpisodeNum,(float*)PyArray_DATA(data),
		       &nError))
    {
      PyErr_SetString(PyExc_TypeError, "Error getting waveform");
      return NULL;
    }
    
  return (PyObject*) data;
  }*/


static PyObject * abfFile_GetTimebase(abfFile* self)
{
  PyArrayObject* data;
  
  
  unsigned int NumSamples, NumSamplesRead;
  int iNumSamp;
  int nError;

  NumSamples = self->FH.lNumSamplesPerEpisode/self->FH.nADCNumChannels;

  iNumSamp = (int) NumSamples;
  
  //Allocate a new PyArray object
  data = (PyArrayObject*) PyArray_SimpleNew(1,&iNumSamp,PyArray_FLOAT);

  
  ABFH_GetTimebase(&self->FH,0.0,(float*)PyArray_DATA(data), NumSamples);
  /*  {
      PyErr_SetString(PyExc_TypeError, "Error getting waveform");
      return NULL;
      }*/
    
  return (PyObject*) data;
}

static PyObject * abfFile_GetStartTime(abfFile* self, PyObject * args)
{
  unsigned int EpisodeNum, ChanNum;
  double startTime;
  int nError;

  if (! PyArg_ParseTuple(args, "ii",&EpisodeNum, &ChanNum))
        return NULL;

  EpisodeNum += 1;

  //Check our arguments
  if (EpisodeNum > self->FH.lActualEpisodes || (DWORD)EpisodeNum < 1)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Episode Number");
      return NULL;
    }

  if (ChanNum >= self->FH.nADCNumChannels || ChanNum < 0)
    {
      PyErr_SetString(PyExc_TypeError, "Invalid Channel Number");
      return NULL;
    }

    
  if (!ABF_GetStartTime(self->hFile,&self->FH,ChanNum,
			(DWORD)EpisodeNum,&startTime,&nError))
    {
      PyErr_SetString(PyExc_TypeError, "Error getting start time");
      return NULL;
    }
    
  return  Py_BuildValue("f",startTime);
}


static PyMethodDef abfFile_methods[] = {
    {"GetData", (PyCFunction)abfFile_GetData, METH_VARARGS,
     "Get a chunk of data corresponding to a particular episode and channel \
      number.\n Note that when calling this function the episodes are 0-indexed\
      like the channel number\
      and the rest of Python, and not 1-indexed as in the abf API.\
\n\
     Parameters: episode number, channel number\
     "
    },
    /*{"GetDigitalWaveform", (PyCFunction)abfFile_GetDigitalWaveform, METH_VARARGS,
     "Get the digital output waveform corresponding to a particular channel # and episode.\n\
\n\
     Parameters: episode number, ADC channel number\
     "
     },*/
    /*{"GetAnalogWaveform", (PyCFunction)abfFile_GetAnalogWaveform, METH_VARARGS,
     "Get the analog autput waveform corresponding to a particular episode and channel number.\n\
\n\
     Parameters: episode number, DAC channel number\
     "
     },*/
    {"GetTimebase", (PyCFunction)abfFile_GetTimebase, METH_NOARGS,
     "Get the timebase for one episode\
     "
    },
    {"GetStartTime", (PyCFunction)abfFile_GetStartTime, METH_VARARGS,
     "Get the starting time of a particular episode.\n\
\n\
     Parameters: episode number, channel number\
     "
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject abfFileType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "abf.abfFile",             /*tp_name*/
    sizeof(abfFile),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)abfFile_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "an Axon abf file",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    abfFile_methods,             /* tp_methods */
    0,             /* tp_members */
    abfFile_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)abfFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    abfFile_new,                 /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initabf(void) 
{
    PyObject* m;

    if (PyType_Ready(&abfFileType) < 0)
        return;

    m = Py_InitModule3("abf", module_methods,
                       "Module containing a class for interfacing an Axon abf file.");

    if (m == NULL)
      return;

    import_array()

    Py_INCREF(&abfFileType);
    PyModule_AddObject(m, "abfFile", (PyObject *)&abfFileType);
}
