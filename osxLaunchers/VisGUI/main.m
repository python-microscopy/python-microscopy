//
//  main.m
//  VisGUI
//
//  Created by David Baddeley on 5/27/15.
//  Copyright (c) 2015 David Baddeley. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#include <Python/Python.h>

int main(int argc, const char * argv[]) {
    printf("argc = %d\n", argc);
    for (int i = 0; i < argc; i++){
        printf(argv[i]);
        printf("\n");
    }
    
    if (false) { //(argc > 1){
        NSLog(@"launching Python interpreter");
        
        //Py_SetProgramName("VisGUI");
        Py_Initialize();
        PySys_SetArgvEx(argc, argv, 0);
        
        PyRun_SimpleString("import site");
        PyRun_SimpleString("import sys\nprint sys.exec_prefix");
        
        printf(Py_GetPath());
        NSLog([NSString stringWithCString: (Py_GetPath()) encoding:NSUTF8StringEncoding]);
        //PyObject *pFilename = PyString_FromString(argv[1]);
        
        PyRun_SimpleString("import sys\nfrom PYME.LMVis import VisGUI\nVisGUI.main(sys.argv[1])\n");
        Py_Finalize();
        return 0;
    } else {
        NSLog(@"launching overseer");
        return NSApplicationMain(argc, argv);
    }
}
