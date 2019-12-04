/************************************************************************************************
**
**    Copyright (c) 1993-1997 Axon Instruments.
**    All rights reserved.
**
*************************************************************************************************
** HEADER:  AXODEFN.H
** PURPOSE: Contains standard Axon definitions and prototypes. 
** AUTHOR:  BHI  Oct 1993
** NOTES:   * The following compilers are supported:
**            - Microsoft C/C++     (COMPILER = "Mircosoft")
**            - Borland C/C++       (COMPILER = "Borland")
**          * The following platforms are supported:
**            - MSDOS               (if _DOS #defined)
**            - WIN32               (if _WIN32 #defined)
*/

#ifndef INC_AXODEFN_H
#define INC_AXODEFN_H

#ifdef __cplusplus
   extern "C" {
#endif  /* __cplusplus */

#undef COMPILER

/*==============================================================================================
** Microsoft C compiler
*/
#ifdef _MSC_VER
#ifndef __STF__
#define COMPILER "Microsoft"

/* Pragmas to cutdown on MSVC warnings. */
#pragma warning(disable:4001)    // warning C4001: nonstandard extension 'single line comment' was used
#pragma warning(disable:4100)    // warning C4100: 'lParam' : unreferenced formal parameter
#pragma warning(disable:4206)    // warning C4206: nonstandard extension used : translation unit is empty
#pragma warning(disable:4505)    // warning C4505: 'GetFont' : unreferenced local function has been removed
#pragma warning(disable:4704)    // warning C4704: 'AXODBG_WriteProtectSelector' : in-line assembler precludes global optimizations
#pragma warning(disable:4705)    // warning C4705: statement has no effect
#pragma warning(disable:4710)    // warning C4710: function 'FunctionName(void)' not expanded

#if !defined(_M_IX86) && defined(_M_I86)
   #define _M_IX86 _M_I86  // Define the MSVC2x _M_IX86 symbol.
#endif
#if !defined(_M_I86) && defined(_M_IX86)
   #define _M_I86 _M_IX86  // Define the MSVC1x _M_I86 symbol.
#endif
#if !defined(_M_I86LM) && !defined(_WIN32)
   #error "ERROR: Large memory model must be used for 16 bit compiles."
#endif
#endif
#endif   /* _MSC_VER */

/*===============================================================================================
** Borland C/C++ compiler.
*/

#ifdef __BORLANDC__
#define COMPILER "Borland"

// Compatibility #defines for Borland C/C++ to behave like MSVC.
#define _M_I86       // Assume that Borland ONLY targets 80X86 processors.
#define _M_IX86 300  // Assume that Borland ONLY targets 80X86 processors.

#ifdef _Windows
   #define _WINDOWS
#endif

#ifdef __WIN32__
   #define _WIN32
#elif !defined(__LARGE__)
   #error "ERROR: Large memory model must be used for 16 bit compiles."
#endif

#ifdef __DLL__
   #ifdef _WINDOWS
      #define _WINDLL
   #else
      #define _DLL
   #endif
#endif

#endif  /* __BORLANDC__ */

/*===============================================================================================
** Microsoft Resource Compiler
*/

#ifdef RC_INVOKED
#define COMPILER "Resource Compiler"
#define _WINDOWS
#endif  /* RC_INVOKED */

/*===============================================================================================
** Add other compiler dependant code HERE!
*/

#if defined(__UNIX__) || defined(__STF__) 
#define COMPILER "gcc"

#include "unix.h"


#endif /*__UNIX__*/

//===============================================================================================
//
// If compiler is unknown, abort with an error.
//

#ifndef COMPILER
   #error "Compiler not recognised... check AXODEFN.H"
#endif

//===============================================================================================
// AXOAPI should be used in the declaration of all cross platform API functions.
// e.g. void AXOAPI ABF_Initialize(void);
#ifdef _WINDOWS
#define AXOAPI WINAPI
#else
#define AXOAPI PASCAL
#endif

//===============================================================================================
// MACRO:   ELEMENTS_IN
// PURPOSE: Returns the count of the elements in an array.
// NOTES:   *only* use this on an array. 
//          Do not use this on a pointer or it will not return the correct value.
//
#ifndef ELEMENTS_IN
#define ELEMENTS_IN(p)  (sizeof(p)/sizeof((p)[0]))
#endif

//===============================================================================================
//
// 16/32 bit compatibility #defines
//

#if defined(_WIN32)
#ifndef __STF__
   #define PLATFORM "Win32"
   #ifndef _WINDOWS
      #define _WINDOWS
   #endif
   #include "..\common\win32.h"
#endif
#elif defined(_DOS)
   #define PLATFORM "DOS"
   #include "..\common\msdos.h"
#elif defined(_WINDOWS)
   #error "ERROR: WIN16 is not supported any more."
#elif defined(__UNIX__)
   #define PLATFORM "Unix"
#else
   #error "Platform not recognised... check AXODEFN.H"
#endif

//=======================================================================================
// Macros used to stringize an argument.
//

// Helper macro for the stringize macro.
// Sometimes a nested call is necessary to get correct expansion by the preprocessor.
#ifndef AX_STRINGIZE_HELPER
#define AX_STRINGIZE_HELPER(a)   #a
#endif

#ifndef AX_STRINGIZE
#define AX_STRINGIZE(a)          AX_STRINGIZE_HELPER(a)  
#endif

// This macro formats the module and line number to prefix the message in the way that
// MSVC likes so that you can simply double click on the line to go to the location.
#ifndef AX_FILELINEMSG
#define AX_FILELINEMSG(msg)  __FILE__ "(" AX_STRINGIZE(__LINE__) ") : " msg
#endif

// Usage:
// #pragma message( __FILE__ "(" AX_STRINGIZE(__LINE__) ") : warning - MFC version change.")
// or
// #pragma message( AX_FILELINEMSG("warning - MFC version change.") )
//

//=======================================================================================
// Macros to declare and use string constants based on a symbol.
#if !defined(DECLARE_STR) && !defined(USE_STR)
#define DECLARE_STR(Name) static const char s__sz##Name[] = #Name
#define USE_STR(Name)     s__sz##Name
#endif

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif  /* __AXODEFN_H__ */
