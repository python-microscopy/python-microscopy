//-*- c++ -*-
/*  cSMI extension module */
%module cSMI

%{
#include "DataStack.h"
#include "BaseRenderer.h"
#include "LUTRGBRenderer.h"
#include "DisplayParams.h"
#include "displayopts.h"
#include "LineProfile.h"
#include "excepts.h"
//#include "ShutterControl.h"
//#include "Cam.h"
//#include "PiezoOp.h"
//#include "StepOp.h"
//#include "SerialOp.h"
#include <string>

#include "numpy/arrayobject.h"
%}


    
// Language independent exception handler
%include exception.i

%exception {
	try {
 	$action
	} catch(IndexOutOfBounds) {
		SWIG_exception(SWIG_IndexError, "Index out of bounds");
	} catch(MemoryAllocError e) {
		SWIG_exception(SWIG_MemoryError, e.desc.c_str());
	} catch(FileIOError e) {
		SWIG_exception(SWIG_IOError, e.desc.c_str());
	} catch(...) {
		SWIG_exception(SWIG_RuntimeError,"Unknown Exception");
	}
}

%include "std_string.i"

/*%typemap(in) (int nchs, int *chans) 
{
	int buf[20];
	int len = PyTuple_Size($input);
	
	if (len > 20 || len < 1) 
	{
           PyErr_SetString(PyExc_ValueError,"Expected number of channels between 1 and 20");
           return NULL;
    }
    
    for (int i = 0; i < len; i++)
    {
    	PyObject * o = PyTuple_GetItem($input,i);	
    	buf[i] = (int)PyInt_AsLong(o);
    }
	

    $1 = len;
    $2 = buf;
};*/

%{
PyObject* _CDataStack_AsArray(PyObject *self, PyObject *args)
    {
		PyObject* obj0;
		PyObject* ret;
		CDataStack *ds;
		int dims[3];
		int nd = 3;
		int chnum = 0;
		int i;
		int strid;
		
		/*import_array();*/
		
		if(!PyArg_ParseTuple(args,(char *)"Oi:CDataStack_AsArray",&obj0, &chnum)) return NULL;
		if ((SWIG_ConvertPtr(obj0,(void **) &ds, SWIGTYPE_p_CDataStack,SWIG_POINTER_EXCEPTION | 0 )) == -1) return NULL;
		
		dims[0] = ds->getWidth();
		dims[1] = ds->getHeight();
		dims[2] = ds->getDepth();
		/*dims[3] = ds->numChannels;*/
		
		strid = sizeof(unsigned short);
		
		{
			try {
				ret = PyArray_FromDimsAndData(nd,dims,PyArray_USHORT, (char*)ds->getChannel(chnum));
				
				for (i = 0; i < nd; i++)
				{
					PyArray_STRIDES(ret)[i] = strid;
					strid *= dims[i];
				};
            
			} catch(IndexOutOfBounds) {
				SWIG_exception(SWIG_IndexError, "Index out of bounds");
			} catch(MemoryAllocError e) {
				SWIG_exception(SWIG_MemoryError, e.desc.c_str());
			} catch(FileIOError e) {
				SWIG_exception(SWIG_IOError, e.desc.c_str());
			} catch(...) {
				SWIG_exception(SWIG_RuntimeError,"Unknown Exception");
			}
		}
		
		Py_INCREF(obj0);
		PyArray_BASE(ret) = obj0;
		
		return ret;
		
		fail:
		return NULL;
	}
%}
 
%include "DataStack.h"
%include "BaseRenderer.h"
%include "LUTRGBRenderer.h"
%include "DisplayParams.h"
%include "displayopts.h"
//%include "ShutterControl.h"
//%include "Cam.h"
//%include "SerialOp.h"
//%include "PiezoOp.h"
//%include "StepOp.h"


%rename(__len__) LineProfile::length;
%rename(__getitem__) LineProfile::operator[];
%include "LineProfile.h"
    
%addmethods CLUT_RGBRenderer 
{
    void pyRender(PyObject *o, CDataStack * ds, int zpos=-1)
    {
    	unsigned char* buffer;
    	int size;

    	//bool blocked = wxPyBeginBlockThreads();
    	if (!PyArg_Parse(o, "t#", &buffer, &size))
    				return;

    	/*if (size != ds->getWidth() * ds->getHeight() * 3) {
    		PyErr_SetString(PyExc_TypeError, "Incorrect buffer size");
    		return;
    	}*/
    	//self->SetData(buffer);
    	self->Render(buffer, ds, zpos);

    	//wxPyEndBlockThreads(blocked);
        
    } 
};



%addmethods CDataStack 
{
    CDataStack(CDataStack &ds,int x1,int y1,int z1,int x2,int y2,int z2, PyObject *a)
    {
    	int* buffer;
    	int size;
    	PyObject *t;
    	//int nargs;
    	
    	if (!PyArg_Parse(a, "O", &t)) return 0;
    	
    	size = PyTuple_Size(t);
    	
    	if (size < 1)
    	{
    		PyErr_SetString(PyExc_TypeError, "Must take at least one channel");
    		return 0;
    	}
    	
    	buffer = new int[size];
    	
    	for (int i = 0; i < size; i++)
    	{
    		PyObject * o = PyTuple_GetItem(t,i);
    		
    		buffer[i] = (int)PyInt_AsLong(o);
    	}
    		
    	CDataStack * ds2;
    	ds2 = new CDataStack(ds,x1,y1,z1,x2,y2,z2, size, buffer);
    	
    	delete buffer;
    	
    	return ds2;
    } 
};
    
%native(CDataStack_AsArray) _CDataStack_AsArray;   
    
%init %{
import_array();
%}   
    
    
