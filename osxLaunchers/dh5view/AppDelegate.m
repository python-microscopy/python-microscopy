//
//  AppDelegate.m
//  t1
//
//  Created by David Baddeley on 5/27/15.
//  Copyright (c) 2015 David Baddeley. All rights reserved.
//

#import "AppDelegate.h"
#include <stdlib.h>

void _openFiles(NSArray* filenames) {
    NSLog(@"Opening Files");
    
    //int pid = [[NSProcessInfo processInfo] processIdentifier];
    NSPipe *pipe = [NSPipe pipe];
    //NSFileHandle *file = pipe.fileHandleForReading;
    
    NSArray *script = @[@"/Users/david/anaconda/bin/dh5view"];
    
    NSTask *task = [[NSTask alloc] init];
    //task.launchPath = @"/Users/david/anaconda/bin/dh5view.py";
    task.launchPath = @"/usr/bin/nohup";
    //task.arguments = @[@"foo", @"bar.txt"];
    task.arguments = [script arrayByAddingObjectsFromArray:filenames];
    task.standardOutput = pipe;
    
    [task launch];
    
}


@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@end

@implementation AppDelegate

int nFiles = 0;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Insert code here to initialize your application
    NSLog(@"finished launching");
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Insert code here to tear down your application
}

- (void)application:(NSApplication *)sender openFiles:(NSArray *)filenames {
    _openFiles(filenames);
    
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
    NSArray * args = @[];
    _openFiles(args);
    nFiles += 1;
    
    //_exit(0);
    
    //[NSApp terminate:self];
    
    //foo
    return TRUE;
}


@end
