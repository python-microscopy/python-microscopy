//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  FileIO.HPP
// PURPOSE: Contains the class definition for the CFileIO class, a simple wrapper
//          around the WIN32 file I/O API.
// 

#ifndef INC_FileIO_HPP
#define INC_FileIO_HPP

#include "axodefn.h"
#include "./../Common/wincpp.hpp"
#include "./../AxAbfFio32/abffiles.h"               // ABF file I/O API and error codes.
#include "./../AxAbfFio32/csynch.hpp"               // Virtual synch array object

class CFileIO
{
private:    // Member variables and constants.
   char         m_szFileName[_MAX_PATH]; // The complete filename of the file
   FILEHANDLE       m_hFileHandle;           // The DOS file handle for data file
	DWORD        m_dwLastError;           // Error number for last error.

private:    // Prevent default copy constructor and operator=()
   CFileIO(const CFileIO &FI);
   const CFileIO &operator=(const CFileIO &FI);

private:    // Internal functions.
   BOOL  SeekFailure(DWORD dwOffset);
   
public:   
   CFileIO();
   CFileIO(FILEHANDLE hFile);
   ~CFileIO();
   
   BOOL  Create(LPCSTR szFileName, BOOL bReadOnly, DWORD dwAttributes=FILE_ATTRIBUTE_NORMAL);
   BOOL  CreateEx(LPCSTR szFileName, DWORD dwDesiredAccess, DWORD dwShareMode,
                  DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes);
   BOOL  Close();
/*   BOOL  IsOpen() const;

   BOOL  Write(const void *pvBuffer, DWORD dwSizeInBytes, DWORD *pdwBytesWritten=NULL);
*/   BOOL  Read(void *pvBuffer, DWORD dwSizeInBytes, DWORD *pdwBytesRead=NULL);
   BOOL  Seek(LONGLONG lOffset, UINT uFlag=FILE_BEGIN, LONGLONG *plNewOffset=NULL);
/*   BOOL  GetCurrentPosition(LONGLONG *plCurrentPosition);
   BOOL  Flush();

   BOOL     SetEndOfFile();
*/   LONGLONG GetFileSize();
/*
   BOOL   GetFileTime( LPFILETIME pCreationTime, LPFILETIME pLastAccessTime=NULL, 
                       LPFILETIME pLastWriteTime=NULL);
   BOOL   SetFileTime( LPFILETIME pCreationTime, LPFILETIME pLastAccessTime=NULL, 
                       LPFILETIME pLastWriteTime=NULL);

   BOOL   GetFileInformation(LPBY_HANDLE_FILE_INFORMATION lpFileInformation);

*/   void   SetFileHandle(FILEHANDLE hFile);
   FILEHANDLE GetFileHandle() const;   
   LPCSTR GetFileName() const;
   FILEHANDLE Release();   
/*
   BOOL   Duplicate(CFileIO *pNewFile, BOOL bInheritable=TRUE);
   BOOL   SetInheritable(BOOL bInheritable=TRUE);
*/   
   BOOL   SetLastError();
   BOOL   SetLastError(DWORD nError);
   DWORD  GetLastError() const;

};

//===============================================================================================
// FUNCTION: GetFileName
// PURPOSE:  Get the name of the file.
//
inline LPCSTR CFileIO::GetFileName() const
{
//   MEMBERASSERT();
   return m_szFileName;
}

//===============================================================================================
// FUNCTION: GetFileHandle
// PURPOSE:  Returns the file handle opened in the object.
//
inline FILEHANDLE CFileIO::GetFileHandle() const
{
//   MEMBERASSERT();
   return m_hFileHandle;
}
/*

//===============================================================================================
// FUNCTION: IsOpen
// PURPOSE:  Returns TRUE if an open file handle is held.
//
inline BOOL CFileIO::IsOpen() const
{
   MEMBERASSERT();
   return (m_hFileHandle != INVALID_HANDLE_VALUE);
}

//===============================================================================================
// FUNCTION: SetEndOfFile
// PURPOSE:  Truncates the file to the current position.
//
inline BOOL CFileIO::SetEndOfFile()
{
   MEMBERASSERT();
   return ::SetEndOfFile(m_hFileHandle) ? TRUE : SetLastError();
}


//===============================================================================================
// CLASS:   CFileIO_NoClose
// PURPOSE: Derivation of CFileIO that does not close the file when destroyed.
// NOTES:   N.B. Not polymorphic -- do NOT use through base class pointer.
//
class CFileIO_NoClose : public CFileIO
{
private:    // Prevent default copy constructor and operator=()
   CFileIO_NoClose(const CFileIO_NoClose &FI);
   const CFileIO_NoClose &operator=(const CFileIO_NoClose &FI);

public:   
   CFileIO_NoClose(HANDLE hFile);
   ~CFileIO_NoClose();
};

//===============================================================================================
// CLASS:   CFileIO_Pipe
// PURPOSE: Class wrapper around a Win32 pipe.
// NOTES:   Poor encapsulation as internal CFileIO objects are returned.
//
class CFileIO_Pipe
{
private:
   CFileIO  m_ReadPipe;
   CFileIO  m_WritePipe;

private:    // Prevent default copy constructor and operator=()
   CFileIO_Pipe(const CFileIO_Pipe &);
   const CFileIO_Pipe &operator=(const CFileIO_Pipe &);

public:
   CFileIO_Pipe();
   ~CFileIO_Pipe();

   BOOL Create(BOOL bInheritable);
   CFileIO *GetReadPipe();
   CFileIO *GetWritePipe();
};
*/
#endif   // INC_FileIO_HPP
