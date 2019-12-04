//******************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//
//******************************************************************************
// HEADER:  AXODEBUG.H
// PURPOSE: Contains utility macros and defines for Axon development debugging.
// AUTHOR:  BHI  Oct 1993
//
// Debugging functions in WINDOWS.H
// BOLL WINAPI IsWindow(HWND hWnd);
// BOOL WINAPI IsGDIObject(HGDIOBJ hobj);
// BOOL WINAPI IsBadReadPtr(const void *lp, UINT cb);
// BOOL WINAPI IsBadWritePtr(void *lp, UINT cb);
// BOOL WINAPI IsBadHugeReadPtr(const void _huge* lp, DWORD cb);
// BOOL WINAPI IsBadHugeWritePtr(void _huge* lp, DWORD cb);
// BOOL WINAPI IsBadCodePtr(FARPROC lpfn);
// BOOL WINAPI IsBadStringPtr(const void *lpsz, UINT cchMax);
// void WINAPI FatalExit(int nCode);   // (nCode = -1 -> stack overflow)
// void WINAPI FatalAppExit(UINT, LPCSTR);
// void WINAPI DebugBreak(void);
// void WINAPI OutputDebugString(LPCSTR);


#ifndef __AXODEBUG_H__
#define __AXODEBUG_H__

#include <assert.h>

#ifndef __UNIX__
	#include <crtdbg.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Setup the debug reporting callback.
void AXODBG_Initialize(void);

// Protected call to DebugBreak() that only breaks if a debugger is running.
BOOL AXODBG_DebugBreak(void);

// Prints a printf formatted string to the debug context.
int cdecl AXODBG_printf( char *lpsz, ... );


// Set a prefix string used for all subsequent calls to AXODBG_printf()
void  AXODBG_SetTracePrefix(char const * szNewTracePrefix);

// You can set the prefix to anything, but usually you will want to set it to include the EXE and/or DLL name.

// This function sets the prefix to "{App=foo} {Mod=bar} "
// Call this function once from within your DllMain(), WinMain() or CWinApp::InitInstance()
void  AXODBG_SetTracePrefixFromModuleHandle(HMODULE mod_handle);


// Prints a textual description of a WIN32 system error to the debug context.
int  AXODBG_ShowSystemError(DWORD dwSystemError);
UINT AXODBG_GetSystemErrorText(DWORD dwSystemError, LPSTR pszBuf, UINT uMaxLen);

// Returns TRUE if the given pointer is aligned on a multiple of the passed data size.
BOOL AXODBG_IsAligned( void *pv, UINT uDataSize );

// Returns TRUE if the given pointer is not on the stack (rough check).
// Typically this function should be used in the constructor of large objects to check
// that they are not being allocated on the stack.
// e.g.
// CSomeObject::CSomeObject()
// {
//    MEMBERASSERT();
//    ASSERT_NOTONSTACK(this);
// }
BOOL AXODBG_NotOnStack( void *pv );

// Assertion processing.
void AXODBG_assert(LPCSTR psExp, LPCSTR psFile, int nLine);
void AXODBG_ErrorMsg(LPCSTR psFile, int nLine, LPCSTR pszFormat, ...);
void AXODBG_SystemErrorMsg(DWORD dwSystemError, LPCSTR psFile, int nLine);

// Define our own ASSERT macro that is only compiled into a _DEBUG build.
// This gives us more control over where the error output gets displayed
// than the default runtime version.

#if !defined(ASSERT)
#ifdef _STFDEBUG
/*   #define ASSERT(exp)               (void)( (exp) || (AXODBG_assert(#exp, __FILE__, __LINE__), 0) )*/
	#define ASSERT(exp) assert(exp)
#else
   #define ASSERT(exp)              ((void)0)
#endif
#endif   // ASSERT

//
// The ERRORMSG macros are like a combination of an ASSERT and a TRACE macro. 
// They are typically useful for marking a code path that should never get executed 
// (e.g. default clause of switch statement) with a debug time message and the 
// option to break to the debugger.
//

#if !defined(ERRORMSG)
#ifdef _STFDEBUG
   #define ERRORMSG(msg)            AXODBG_ErrorMsg(__FILE__, __LINE__, msg)
   #define ERRORMSG1(msg,a)         AXODBG_ErrorMsg(__FILE__, __LINE__, msg, a)
   #define ERRORMSG2(msg,a,b)       AXODBG_ErrorMsg(__FILE__, __LINE__, msg, a, b)
   #define ERRORMSG3(msg,a,b,c)     AXODBG_ErrorMsg(__FILE__, __LINE__, msg, a, b, c)
   #define ERRORMSG4(msg,a,b,c,d)   AXODBG_ErrorMsg(__FILE__, __LINE__, msg, a, b, c, d)
   #define ERRORMSG5(msg,a,b,c,d,e) AXODBG_ErrorMsg(__FILE__, __LINE__, msg, a, b, c, d, e)
#else
   #define ERRORMSG(msg)            ((void)0)
   #define ERRORMSG1(msg,a)         ((void)0)
   #define ERRORMSG2(msg,a,b)       ((void)0)
   #define ERRORMSG3(msg,a,b,c)     ((void)0)
   #define ERRORMSG4(msg,a,b,c,d)   ((void)0)
   #define ERRORMSG5(msg,a,b,c,d,e) ((void)0)
#endif
#endif   // ASSERT

#ifdef _WINDOWS
   #define HWNDASSERT(hWnd)      ASSERT(IsWindow(hWnd))
   #define IsBadPtr(p)           IsBadWritePtr((void *)(p), sizeof(*(p)))
   #define IsBadArray(p,n)       IsBadWritePtr((void *)(p), sizeof(*(p))*(n))
   #define RPTRASSERT(p)         ASSERT(!IsBadReadPtr((const void *)(p), sizeof(*(p))))
   #define FNPTRASSERT(p)        ASSERT(!IsBadCodePtr((FARPROC)(p)))
   #define LPSZASSERT(p)         ASSERT(!IsBadStringPtr(p, (UINT)(-1)))
#else
   #define IsBadPtr(p)           (p==NULL)
   #define IsBadArray(p,n)       IsBadPtr(p)
   #define RPTRASSERT(p)         ASSERT(!IsBadPtr(p))
   #define FNPTRASSERT(p)        ASSERT(!IsBadPtr(p))
   #define LPSZASSERT(p)         ASSERT(!IsBadPtr(p))
#endif

#define WPTRASSERT(p)            ASSERT(!IsBadPtr(p))
#define MEMBERASSERT()           WPTRASSERT(this)
#define ARRAYASSERT(p,n)         ASSERT(!IsBadArray(p, n))

//==================================================================================================================
// MACRO:
//    RARRAYASSERT(p,n)
//    WARRAYASSERT(p,n)
// PURPOSE:
//    Validate an array by checking the caller has read access to the array.
// PARAMETERS:
//    p     The array
//    n     The number of elements in the array (NOT the size in bytes)
// USAGE:
//    int   array[10];
//    RARRAYASSERT(array, 10);
#ifdef _WINDOWS
   #define RARRAYASSERT(p,n)     ASSERT( ((p) != NULL) && !IsBadReadPtr( (const void *)(p), sizeof(*(p))*(n) ) )
   #define WARRAYASSERT(p,n)     ASSERT( ((p) != NULL) && !IsBadWritePtr( (void *)(p), sizeof(*(p))*(n) ) )
#else
   #define RARRAYASSERT(p,n)     ASSERT(!IsBadPtr(p))
   #define WARRAYASSERT(p,n)     ASSERT(!IsBadPtr(p))
#endif


//==================================================================================================================
// VERIFY is an ASSERT macro that always evaluates the given expression, but
// only actually ASSERT's in a DEBUG build.
#if !defined(VERIFY)
#ifdef _STFDEBUG
   #define VERIFY(exp)           ASSERT(exp)
#else      
   #define VERIFY(exp)           (exp)
#endif
#endif

//==================================================================================================================
// The DEBUG_ONLY macro only expands out the expression in a DEBUG build.
#if !defined(DEBUG_ONLY)
#ifdef _STFDEBUG
   #define DEBUG_ONLY(exp)       (exp)
#else      
   #define DEBUG_ONLY(exp)       ((void)0)
#endif
#endif

//==================================================================================================================
// Trace functions that are only expanded for debug builds and when the AXODBG_TRACE flag is set.

#if !defined(TRACE)
#if (defined(_STFDEBUG) || defined(AXODBG_TRACE))
   #define TRACE(s)                    AXODBG_printf(s)
   #define DLLTRACE(s)                 AXODBG_printf(s)
   #define TRACE1(s, a)                AXODBG_printf(s, a)
   #define TRACE2(s, a, b)             AXODBG_printf(s, a, b)
   #define TRACE3(s, a, b, c)          AXODBG_printf(s, a, b, c)
   #define TRACE4(s, a, b, c, d)       AXODBG_printf(s, a, b, c, d)
   #define TRACE5(s, a, b, c, d, e)    AXODBG_printf(s, a, b, c, d, e)
#else      
   #define TRACE(s)                    ((void)0)
   #define DLLTRACE(s)                 ((void)0)
   #define TRACE1(s, a)                ((void)0)
   #define TRACE2(s, a, b)             ((void)0)
   #define TRACE3(s, a, b, c)          ((void)0)
   #define TRACE4(s, a, b, c, d)       ((void)0)
   #define TRACE5(s, a, b, c, d, e)    ((void)0)
#endif
#endif

#define TRACE_INT(e)    TRACE1(#e "=%d\n", e)
#define TRACE_FLOAT(e)  TRACE1(#e "=%g\n", e)
#define TRACE_STRING(e) TRACE1(#e "=%s\n", e)

//==================================================================================================================
// SHOW_SYSTEM_ERROR will only dump a text description of a system error in a DEBUG build.
//
#ifdef _STFDEBUG
   #define SHOW_SYSTEM_ERROR(dwError)  AXODBG_ShowSystemError(dwError)
#else
   #define SHOW_SYSTEM_ERROR(dwError)  ((void)0)
#endif

//==================================================================================================================
// VERIFY_SYSTEM_CALL will only dump a text description of a system error in a DEBUG build.
//
#ifdef _STFDEBUG
   #define VERIFY_SYSTEM_CALL(exp)     (exp || (AXODBG_SystemErrorMsg(0, __FILE__, __LINE__), 0))
#else
   #define VERIFY_SYSTEM_CALL(exp)     (exp)
#endif

//==================================================================================================================
// ASSERT_ISALIGNED will throw an assertion error if the pointer is not aligned on a boundary of its data type.
//
#ifdef _STFDEBUG
   #define ASSERT_ISALIGNED(p)  ASSERT(AXODBG_IsAligned(p, sizeof(*p)))
   #define ASSERT_NOTONSTACK(p) ASSERT(AXODBG_NotOnStack(p))
#else
   #define ASSERT_ISALIGNED(p)  ((void)0)
   #define ASSERT_NOTONSTACK(p) ((void)0)
#endif


//==================================================================================================================
// The following macros set and clear, respectively, given bits
// of the C runtime library debug flag, as specified by a bitmask.
//
// Valid flags include:      (default value)
// _CRTDBG_ALLOC_MEM_DF      (on)   
//       ON:  Enable debug heap allocations and use of memory block type identifiers, such as _CLIENT_BLOCK.
//       OFF: Add new allocations to heap's linked list, but set block type to _IGNORE_BLOCK.
// _CRTDBG_CHECK_ALWAYS_DF   (off)  
//       ON:  Call _CrtCheckMemory at every allocation and deallocation request.
//       OFF: _CrtCheckMemory must be called explicitly.
// _CRTDBG_CHECK_CRT_DF      (off)
//       ON:  Include _CRT_BLOCK types in leak detection and memory state difference operations.
//       OFF: Memory used internally by the run-time library is ignored by these operations.
// _CRTDBG_DELAY_FREE_MEM_DF (off)
//       ON:  Keep freed memory blocks in the heap's linked list, assign them the _FREE_BLOCK type, and fill them with the byte value 0xDD.
//       OFF: Do not keep freed blocks in the heap's linked list.
// _CRTDBG_LEAK_CHECK_DF     (off)
//       ON:  Perform automatic leak checking at program exit via a call to _CrtDumpMemoryLeaks and generate an error report if the application failed to free all the memory it allocated.
//       OFF: Do not automatically perform leak checking at program exit.
//
#ifdef   _STFDEBUG
   #define  SET_CRT_DEBUG_FIELD(a) \
               _CrtSetDbgFlag((a) | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))
   #define  CLEAR_CRT_DEBUG_FIELD(a) \
               _CrtSetDbgFlag(~(a) & _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))
#else
   #define  SET_CRT_DEBUG_FIELD(a)   ((void)0)
   #define  CLEAR_CRT_DEBUG_FIELD(a) ((void)0)
#endif


//==================================================================================================================
// HEAPASSERT checks the integrity of the heap managed by the run-time library.
// 
#ifdef   _STFDEBUG
   #define HEAPASSERT()              ASSERT(_CrtCheckMemory())
#else
   #define HEAPASSERT()              ((void)0)
#endif


//==================================================================================================================
// Flags supported by AXODBG_SetDebugFlag.
#define AXODBG_REPORT_FLAG       0x80000000  // If this flag is set the current flags are returned.
#define AXODBG_BREAK_ON_ASSERT   0x00000001  // If set, assertion failures will immediately break to the debugger.

// Function to set/get debugging flags modeled after _CrtSetDbgFlag().
DWORD AXODBG_SetDebugFlag(DWORD dwFlags);

#ifdef   _STFDEBUG
   #define  AXODBG_SET_DEBUG_FIELD(a) \
               AXODBG_SetDebugFlag((a) | AXODBG_SetDebugFlag(AXODBG_REPORT_FLAG))
   #define  AXODBG_CLEAR_DEBUG_FIELD(a) \
               AXODBG_SetDebugFlag(~(a) & AXODBG_SetDebugFlag(AXODBG_REPORT_FLAG))
#else
   #define  AXODBG_SET_DEBUG_FIELD(a)   ((void)0)
   #define  AXODBG_CLEAR_DEBUG_FIELD(a) ((void)0)
#endif

//==================================================================================================================
// Debug time only call to AXODBG_DebugBreak()
#ifdef   _STFDEBUG
   #define  AXODBG_DEBUGBREAK()  AXODBG_DebugBreak()
#else
   #define  AXODBG_DEBUGBREAK()  ((BOOL)FALSE)
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

//==================================================================================================================
// Compile time assert for static expressions 
// (e.g. STATIC_ASSERT(sizeof(MyStruct)==256); )
//
#define STATIC_ASSERT(expr) C_ASSERT(expr)

#endif

#endif   /* __AXODEBUG_H__ */
