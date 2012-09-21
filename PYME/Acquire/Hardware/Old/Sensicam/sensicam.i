//-*- c++ -*-
/*  sensicam extension module */
%module sensicam

%{
//#include "DataStack.h"
//#include "BaseRenderer.h"
//#include "LUTRGBRenderer.h"
//#include "DisplayParams.h"
//#include "displayopts.h"
//#include "LineProfile.h"
//#include "excepts.h"
//#include "ShutterControl.h"
#include "Cam.h"
//#include "PiezoOp.h"
//#include "StepOp.h"
//#include "SerialOp.h"
#include <string>

#include "numpy/arrayobject.h"
%}


    
// Language independent exception handler
//%include exception.i

// %exception {
// 	try {
//  	$action
// 	} catch(IndexOutOfBounds) {
// 		SWIG_exception(SWIG_IndexError, "Index out of bounds");
// 	} catch(MemoryAllocError e) {
// 		SWIG_exception(SWIG_MemoryError, e.desc.c_str());
// 	} catch(FileIOError e) {
// 		SWIG_exception(SWIG_IOError, e.desc.c_str());
// 	} catch(...) {
// 		SWIG_exception(SWIG_RuntimeError,"Unknown Exception");
// 	}
// }

%include "std_string.i"

 
//%include "DataStack.h"
//%include "BaseRenderer.h"
//%include "LUTRGBRenderer.h"
//%include "DisplayParams.h"
//%include "displayopts.h"
//%include "ShutterControl.h"
%include "Cam.h"
//%include "SerialOp.h"
//%include "PiezoOp.h"
//%include "StepOp.h"


    
%init %{
import_array();
%}   
    
    
