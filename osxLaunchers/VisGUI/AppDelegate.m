//
//  AppDelegate.m
//  VisGUI
//
//  Created by David Baddeley on 5/27/15.
//  Copyright (c) 2015 David Baddeley. All rights reserved.
//

#import "AppDelegate.h"

#include <Python/Python.h>

void __openFiles(NSArray* filenames) {
    NSLog(@"Opening Files");
    
    //int pid = [[NSProcessInfo processInfo] processIdentifier];
    NSPipe *pipe = [NSPipe pipe];
    //NSFileHandle *file = pipe.fileHandleForReading;
    
    NSArray *script = @[@"/Users/david/anaconda/bin/VisGUI"];
    
    NSTask *task = [[NSTask alloc] init];
    //task.launchPath = @"/Users/david/anaconda/bin/dh5view.py";
    task.launchPath = @"/usr/bin/nohup";
    //task.arguments = @[@"foo", @"bar.txt"];
    task.arguments = [script arrayByAddingObjectsFromArray:filenames];
    task.standardOutput = pipe;
    
    [task launch];
    
}

void _launchFiles(NSArray* filenames) {
    NSLog(@"Launching new instance");
    
    //[[NSBundle mainBundle] executablePath]
    
    //int pid = [[NSProcessInfo processInfo] processIdentifier];
    //NSPipe *pipe = [NSPipe pipe];
    //NSFileHandle *file = pipe.fileHandleForReading;
    
    //NSArray *script = @[@"/Users/david/anaconda/bin/VisGUI.py"];
    
    NSTask *task = [[NSTask alloc] init];
    //task.launchPath = @"/Users/david/anaconda/bin/dh5view.py";
    //task.launchPath = @"./VisGUI.app/Contents/MacOS/VisGUI";
    //task.launchPath = NSProcessInfo.processInfo.arguments[0];
    //task.arguments = @[@"foo", @"bar.txt"];
    task.launchPath = [[NSBundle mainBundle] executablePath];
    task.arguments = filenames;
    //task.arguments = [script arrayByAddingObjectsFromArray:filenames];
    //task.standardOutput = pipe;
    
    NSMutableDictionary * env = [NSMutableDictionary dictionaryWithDictionary:[[NSProcessInfo processInfo] environment]];
    
    //NSLog([[task environment] description]);
    
    NSString *newPath = [NSString stringWithFormat:@"/Users/david/anaconda/bin:%@", env[@"PATH"]];
    
    [env setObject:newPath forKey:@"PATH"];
    
    [task setEnvironment:env];
    
    NSLog([[task environment] description]);
    
    [task launch];
    
}

void _openFiles(NSArray* filenames) {
    printf(Py_GetPath());
    Py_SetProgramName("VisGUI");
    Py_Initialize();
    PyRun_SimpleString("from PYME.LMVis import VisGUI\nVisGUI.main()\n");
    Py_Finalize();
}

@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@end

@implementation AppDelegate

int nFiles = 0;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Insert code here to initialize your application
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Insert code here to tear down your application
}



- (void)application:(NSApplication *)sender openFiles:(NSArray *)filenames {
    printf("openFiles\n");
    //printf(filenames[0]);
    if (TRUE){
        __openFiles(filenames);
    } else {
        _openFiles(filenames);
    }
    
    nFiles += 1;
    
    //_exit(0);
    
    //[NSApp terminate:self];
    
    //foo
}

- (BOOL)applicationShouldOpenUntitledFile:(NSApplication *)sender{
    
    return (nFiles == 0);
}

- (BOOL)applicationOpenUntitledFile:(NSApplication *)theApplication {
    //    NSLog(@"Opening Files");
    //
    //    //int pid = [[NSProcessInfo processInfo] processIdentifier];
    //    NSPipe *pipe = [NSPipe pipe];
    //    //NSFileHandle *file = pipe.fileHandleForReading;
    //
    //    NSArray *script = @[@"/Users/david/anaconda/bin/dh5view.py"];
    //
    //    NSTask *task = [[NSTask alloc] init];
    //    //task.launchPath = @"/Users/david/anaconda/bin/dh5view.py";
    //    task.launchPath = @"/usr/bin/nohup";
    //    //task.arguments = @[@"foo", @"bar.txt"];
    //    task.arguments = script;
    //    task.standardOutput = pipe;
    //
    //    [task launch];
    //
    //    nFiles += 1;
    NSArray * args = @[@""];
    _launchFiles(args);
    nFiles += 1;
    
    //_exit(0);
    
    //[NSApp terminate:self];
    
    //foo
    return TRUE;
}


@end
