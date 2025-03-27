//
//  main.m
//  t1
//
//  Created by David Baddeley on 5/27/15.
//  Copyright (c) 2015 David Baddeley. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <Python.h>

int main(int argc, const char * argv[]) {
    printf("argc = %d\n", argc);
    for (int i = 0; i < argc; i++){
        printf("%s", argv[i]);
        printf("\n");
    }
    
    if (false) { //(argc > 1){
        NSLog(@"launching Python interpreter");
        printf("%s", Py_GetPath());
        NSLog([NSString stringWithCString: (Py_GetPath()) encoding:NSUTF8StringEncoding]);
        Py_SetProgramName("dh5view");
        Py_Initialize();
        PySys_SetArgvEx(argc, argv, 0);
        //PyObject *pFilename = PyString_FromString(argv[1]);
        
        PyRun_SimpleString("import sys\nfrom PYME.LMVis import VisGUI\nVisGUI.main(sys.argv[1])\n");
        Py_Finalize();
        return 0;
    } else {
        NSLog(@"launching overseer");
        return NSApplicationMain(argc, argv);
    }
}