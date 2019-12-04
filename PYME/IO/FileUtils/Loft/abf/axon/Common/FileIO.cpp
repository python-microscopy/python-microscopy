//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  FileIO.CPP
// PURPOSE: Contains the member functions for the CFileIO class.
// NOTES:
// 
// Overlapped file I/O is not available in WIN95 -- PSS ID Number: Q125717
// Windows 95 does not support overlapped operations on files, disk, pipes, or
// mail slots, but does support overlapped operations on serial and parallel
// communication ports. Non-overlapped file write operations are similar to
// overlapped file writes, because Windows 95 uses a lazy-write disk cache.
// Overlapped I/O can be implemented in a Win32-based application by creating
// multiple threads to handle I/O.
//  
// Asynchronous I/O on files, disk, pipes and mail slots is not implemented in
// Windows 95. If a Win32-based application running on Windows 95 attempts to
// perform asynchronous file I/O (such as ReadFile() with any value other than
// NULL in the lpOverlapped field) on any of these objects, the ReadFile()  or
// WriteFile fails and GetLastError() returns ERROR_INVALID_PARAMETER (87).
//  

#include "wincpp.hpp"
#include "FileIO.hpp"

//===============================================================================================
// FUNCTION: Constructor
// PURPOSE:  Initialize the object
//
CFileIO::CFileIO()
{
   //MEMBERASSERT();
   m_hFileHandle   = NULL;
   m_szFileName[0] = '\0';
   m_dwLastError   = 0;
}

//===============================================================================================
// FUNCTION: Copy Constructor
// PURPOSE:  Creates a CFileIO object from a Win32 HANDLE.
//
CFileIO::CFileIO(FILEHANDLE hFile)
{
   //MEMBERASSERT();
   m_hFileHandle   = hFile;
   m_szFileName[0] = '\0';
   m_dwLastError   = 0;
}

//===============================================================================================
// FUNCTION: Destructor
// PURPOSE:  Cleanup the object when it is deleted.
//
CFileIO::~CFileIO()
{
   //MEMBERASSERT();
   Close();
}

//===============================================================================================
// FUNCTION: Create
// PURPOSE:  Opens a file and stores the filename if successful.
//
BOOL CFileIO::Create(LPCSTR szFileName, BOOL bReadOnly, DWORD dwAttributes)
{
   //MEMBERASSERT();
//   LPSZASSERT(szFileName);
   ASSERT(m_hFileHandle == FILE_NULL);

   DWORD dwFlags    = GENERIC_READ;
   DWORD dwCreation = OPEN_EXISTING;
   if (!bReadOnly)
   {
      dwFlags   |= GENERIC_WRITE;
      dwCreation = CREATE_ALWAYS;
   }
   return CreateEx(szFileName, dwFlags, FILE_SHARE_READ, dwCreation, dwAttributes);
}

//===============================================================================================
// FUNCTION: CreateEx
// PURPOSE:  Opens a file and stores the filename if successful.
//
BOOL CFileIO::CreateEx(LPCSTR szFileName, DWORD dwDesiredAccess, DWORD dwShareMode,
                       DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes)
{
   //MEMBERASSERT();
//   LPSZASSERT(szFileName);
   ASSERT(m_hFileHandle == FILE_NULL);
   m_hFileHandle = ::c_CreateFileA(szFileName, dwDesiredAccess, dwShareMode, NULL, 
                                dwCreationDisposition, dwFlagsAndAttributes, NULL);
   if (m_hFileHandle == FILE_NULL)
      return SetLastError();
      
// TRACE1("Create(%s)\n", szFileName);
   strncpy(m_szFileName, szFileName, _MAX_PATH-1);
   m_szFileName[_MAX_PATH-1] = '\0';
   return TRUE;
}

//===============================================================================================
// FUNCTION: SetFileHandle
// PURPOSE:  Initializes the object from a Win32 HANDLE.
//
void CFileIO::SetFileHandle(FILEHANDLE hFile)
{
   //MEMBERASSERT();
   ASSERT(m_hFileHandle == FILE_NULL);
   m_hFileHandle   = hFile;
   m_szFileName[0] = '\0';
   m_dwLastError   = 0;
}

/*
//===============================================================================================
// FUNCTION: Write
// PURPOSE:  Write a buffer of data to the file.
//
BOOL CFileIO::Write(const void *pvBuffer, DWORD dwSizeInBytes, DWORD *pdwBytesWritten)
{
   //MEMBERASSERT();
//   ASSERT(m_hFileHandle != INVALID_HANDLE_VALUE);

   DWORD dwBytesWritten = 0;
   BOOL bRval = ::WriteFile(m_hFileHandle, pvBuffer, dwSizeInBytes, &dwBytesWritten, NULL);
   if (pdwBytesWritten)
      *pdwBytesWritten = dwBytesWritten;
   if (!bRval)
      SetLastError();
   return bRval;
}


//===============================================================================================
// FUNCTION: Flush
// PURPOSE:  Flush the current file to disk.
//
BOOL CFileIO::Flush()
{
   //MEMBERASSERT();
//   ASSERT(m_hFileHandle != INVALID_HANDLE_VALUE);
   return ::FlushFileBuffers(m_hFileHandle) ? TRUE : SetLastError();
}

*/
//===============================================================================================
// PROCEDURE: Read
// PURPOSE:   Reads a block and returnd FALSE on ERROR.
//
BOOL CFileIO::Read(LPVOID lpBuf, DWORD dwBytesToRead, DWORD *pdwBytesRead)
{
   //MEMBERASSERT();
   ASSERT(m_hFileHandle != FILE_NULL);

   DWORD dwBytesRead = 0;
   BOOL bRval = ::c_ReadFile(m_hFileHandle, lpBuf, dwBytesToRead, &dwBytesRead, NULL);
   if (pdwBytesRead)
      *pdwBytesRead = dwBytesRead;
   if (!bRval)
      return SetLastError();
   if (dwBytesRead!=dwBytesToRead)
      return SetLastError(ERROR_HANDLE_EOF);
   return TRUE;
}


//===============================================================================================
// FUNCTION: Close
// PURPOSE:  Closes a previously opened file.
//
BOOL CFileIO::Close()
{
   //MEMBERASSERT();
   if (m_hFileHandle != NULL)
   {
      if (!::c_CloseHandle(m_hFileHandle))
         return SetLastError();

      // TRACE1("Close(%s)\n", m_szFileName);
      m_hFileHandle = NULL;
   }
   m_szFileName[0] = '\0';
   return TRUE;
}
      

//===============================================================================================
// FUNCTION: Release
// PURPOSE:  Releases the file handle held in the object.
//
FILEHANDLE CFileIO::Release()
{
   //MEMBERASSERT();
   FILEHANDLE hRval    = m_hFileHandle;
   m_hFileHandle   = NULL;
   m_szFileName[0] = '\0';
   return hRval;
}


//===============================================================================================
// FUNCTION: SeekFailure
// PURPOSE:  Returns TRUE if an open file handle is held.
//
BOOL CFileIO::SeekFailure(DWORD dwOffset)
{
   //MEMBERASSERT();
   if (dwOffset == 0xFFFFFFFF)
   { 
      DWORD dwError = GetLastError();
      if (dwError != NO_ERROR )
      {
         SetLastError(dwError);
         return TRUE;
      }
   }
   return FALSE;
}

//===============================================================================================
// FUNCTION: Seek
// PURPOSE:  Change the current position of the file pointer.
// NOTES:    Valid flags are FILE_CURRENT, FILE_BEGIN, & FILE_END.
//
BOOL CFileIO::Seek(LONGLONG lOffset, UINT uFlag, LONGLONG *plNewOffset)
{
#ifndef _WINDOWS
	/*MEMBERASSERT();*/
    short    origin = 0;

    switch (uFlag)
    {
            case FILE_BEGIN : origin = SEEK_SET;                    /* start of file */
                     break;
            case FILE_CURRENT : origin = SEEK_CUR; /* current position of file pointer */
                     break;
            case FILE_END : origin = SEEK_END;                      /* end of file */
                     break;
    }
    return (!fseek (m_hFileHandle, (long)lOffset, origin)); /* stdio read */
#else
   MEMBERASSERT();
   LARGE_INTEGER Offset;
   Offset.QuadPart = lOffset;
   Offset.LowPart = ::SetFilePointer(m_hFileHandle, Offset.LowPart, &Offset.HighPart, uFlag);
   if (SeekFailure(Offset.LowPart))
      return FALSE;
   if (plNewOffset)
      *plNewOffset = Offset.QuadPart;
   return TRUE;
#endif
}
/*
//===============================================================================================
// FUNCTION: GetCurrentPosition
// PURPOSE:  Get the current position of the file pointer.
//
BOOL CFileIO::GetCurrentPosition(LONGLONG *plCurrentPosition)
{
   //MEMBERASSERT();
   return Seek(0, FILE_CURRENT, plCurrentPosition);
}
*/
//===============================================================================================
// FUNCTION: GetFileSize
// PURPOSE:  Returns the size of a file in bytes.
//
LONGLONG CFileIO::GetFileSize()
{
   /*MEMBERASSERT();*/
   ASSERT(m_hFileHandle != FILE_NULL);
#ifndef _WINDOWS
   return c_GetFileSize(m_hFileHandle,NULL);
#else
   LARGE_INTEGER FileSize;
   FileSize.QuadPart = 0;

   FileSize.u.LowPart = ::GetFileSize(m_hFileHandle, LPDWORD(&FileSize.u.HighPart));
   if (SeekFailure(FileSize.u.LowPart))
      return 0;

   return FileSize.QuadPart;
#endif
}
/*      
//===============================================================================================
// FUNCTION: GetFileTime
// PURPOSE:  Gets time values for the file.
//
BOOL CFileIO::GetFileTime( LPFILETIME pCreationTime, LPFILETIME pLastAccessTime, LPFILETIME pLastWriteTime)
{
   //MEMBERASSERT();
   //ASSERT(m_hFileHandle != INVALID_HANDLE_VALUE);
   return ::GetFileTime(m_hFileHandle, pCreationTime, pLastAccessTime, pLastWriteTime);
}

//===============================================================================================
// FUNCTION: SetFileTime
// PURPOSE:  Sets time values for the file.
//
BOOL CFileIO::SetFileTime( LPFILETIME pCreationTime, LPFILETIME pLastAccessTime, LPFILETIME pLastWriteTime)
{
   //MEMBERASSERT();
   //ASSERT(m_hFileHandle != INVALID_HANDLE_VALUE);
   return ::SetFileTime(m_hFileHandle, pCreationTime, pLastAccessTime, pLastWriteTime);
}

//===============================================================================================
// FUNCTION: GetFileInformation
// PURPOSE:  .
//
BOOL CFileIO::GetFileInformation(LPBY_HANDLE_FILE_INFORMATION pFileInformation)
{
   //MEMBERASSERT();
   return ::GetFileInformationByHandle(m_hFileHandle, pFileInformation) ? TRUE : SetLastError();
}

//===============================================================================================
// FUNCTION: Duplicate
// PURPOSE:  Makes a copy of the handle into another CFileIO object.
//
BOOL CFileIO::Duplicate(CFileIO *pNewFile, BOOL bInheritable)
{
   //MEMBERASSERT();
   HANDLE hNewHandle  = NULL;
   HANDLE *pNewHandle = NULL;
   if (pNewFile)
   {
//      WPTRASSERT(pNewFile);
      pNewHandle = &hNewHandle;
   }

   if (!DuplicateHandle(GetCurrentProcess(), // source process
                        m_hFileHandle,       // handle to duplicate
                        GetCurrentProcess(), // destination process
                        pNewHandle,          // new handle
                        0,                   // new access flags - ignored since DUPLICATE_SAME_ACCESS
                        bInheritable,        // it's inheritable
                        DUPLICATE_SAME_ACCESS))
      return SetLastError();

   if (pNewFile)
      pNewFile->SetFileHandle(hNewHandle);

   return TRUE;
}

//===============================================================================================
// FUNCTION: SetInheritable
// PURPOSE:  Sets the "inheritability" of the file handle.
//
BOOL CFileIO::SetInheritable(BOOL bInheritable)
{
   //MEMBERASSERT();
   return Duplicate(NULL, bInheritable);
}
*/
//===============================================================================================
// FUNCTION: SetLastError
// PURPOSE:  Sets the last error value and always returns FALSE for convenience.
//
BOOL CFileIO::SetLastError()
{
   /*MEMBERASSERT();*/
   return SetLastError(GetLastError());
}


//===============================================================================================
// FUNCTION: SetLastError
// PURPOSE:  Sets the last error value and always returns FALSE for convenience.
//
BOOL CFileIO::SetLastError(DWORD dwError)
{
   /*MEMBERASSERT();*/
   m_dwLastError = dwError;
//   TRACE1("CFileIO::SetLastError(%u)\n", dwError);
//   SHOW_SYSTEM_ERROR(dwError);
   return FALSE;           // convenience.
}

//===============================================================================================
// FUNCTION: GetLastError
// PURPOSE:  Gets the last error condition provoked by this object.
//
DWORD CFileIO::GetLastError() const
{
   /*MEMBERASSERT();*/
   return m_dwLastError;
}
/*
//###############################################################################################
//###############################################################################################
//###
//###    CFileIO_NoClose is a non-polymorphic derivation of CFileIO that does not close the
//###    file on destruction. Typically used with the copy constructor to wrap a HANDLE.
//###
//###############################################################################################
//###############################################################################################

//===============================================================================================
// FUNCTION: Constructor
// PURPOSE:  Object initialization.
//
CFileIO_NoClose::CFileIO_NoClose(HANDLE hFile)
   : CFileIO(hFile)
{
}

//===============================================================================================
// FUNCTION: Destructor
// PURPOSE:  Object cleanup.
//
CFileIO_NoClose::~CFileIO_NoClose()
{
   Release();
}

//###############################################################################################
//###############################################################################################
//###
//###    CFileIO_Pipe is a thin wrapper around a Win32 pipe.
//###
//###############################################################################################
//###############################################################################################

CFileIO_Pipe::CFileIO_Pipe()
{
}

CFileIO_Pipe::~CFileIO_Pipe()
{
}

BOOL CFileIO_Pipe::Create(BOOL bInheritable)
{
   // set up the security attributes for the anonymous pipe
   SECURITY_ATTRIBUTES saPipe  = { 0 };
   saPipe.nLength              = sizeof(SECURITY_ATTRIBUTES);
   saPipe.lpSecurityDescriptor = NULL;
   saPipe.bInheritHandle       = bInheritable;
   
   // handles to the anonymous pipe
   HANDLE hReadPipe=NULL, hWritePipe=NULL;

   // create the anonymous pipe
   if (!CreatePipe(&hReadPipe, &hWritePipe, &saPipe, 0))
   {
      SHOW_SYSTEM_ERROR(0);
      return FALSE;
   }

   m_ReadPipe.SetFileHandle(hReadPipe);
   m_WritePipe.SetFileHandle(hWritePipe);
   return TRUE;
}

CFileIO *CFileIO_Pipe::GetReadPipe()
{
   //MEMBERASSERT();
   return &m_ReadPipe;
}

CFileIO *CFileIO_Pipe::GetWritePipe()
{
   //MEMBERASSERT();
   return &m_WritePipe;
}

#ifdef TESTBED

#define WRITETEST 1
#define READTEST  1

int main(int argc, char **argv)
{
   const char szFilename[] = "D:\\Bigfile.tmp";
   const LONGLONG llFileLen = LONGLONG(1024 * 1024 * 1024) * 3;
   const char szWriteText[] = "Bruce's big file writing test.";
   const int nTextLen = strlen(szWriteText);

#if WRITETEST
   {
      CFileIO File;
      //VERIFY(File.Create(szFilename, FALSE));
   
      LONGLONG llOffset = 0;

//      VERIFY(File.Seek(llFileLen, FILE_BEGIN, &llOffset));
//      ASSERT(llOffset == llFileLen);
      printf("llOffset = %d MB\n", int(llOffset / (1024 * 1024)));

      //VERIFY(File.Write(szWriteText, nTextLen));

      LONGLONG llFileSize = File.GetFileSize();
//      ASSERT(llFileSize > 0);
//      ASSERT(llFileSize == llFileLen + nTextLen);
      printf("llFileSize = %d MB\n", int(llFileSize / (1024 * 1024)));

//      VERIFY(File.Seek(-nTextLen, FILE_CURRENT, &llOffset));

      char szReadText[128] = "";
//      VERIFY(File.Read(szReadText, nTextLen));
      szReadText[nTextLen] = '\0';

//      ASSERT(strcmp(szWriteText, szReadText)==0);
      printf("szReadText = %s\n", szReadText);
//      VERIFY(File.Close());
   }
#endif

#if READTEST
   {
      CFileIO File;
      //VERIFY(File.Create(szFilename, TRUE));

      LONGLONG llOffset = 0;
      //VERIFY(File.Seek(llFileLen, FILE_BEGIN, &llOffset));
      printf("llOffset = %d MB\n", int(llOffset / (1024 * 1024)));

      LONGLONG llFileSize = File.GetFileSize();
      ASSERT(llFileSize > 0);
      ASSERT(llFileSize == llFileLen + nTextLen);
      printf("llFileSize = %d MB\n", int(llFileSize / (1024 * 1024)));

      char szReadText[128] = "";
      //VERIFY(File.Read(szReadText, nTextLen));
      szReadText[nTextLen] = '\0';

      ASSERT(strcmp(szWriteText, szReadText)==0);
      printf("szReadText = %s\n", szReadText);
      //VERIFY(File.Close());
   }
#endif
   return 0;
}

#endif  // TESTBED
*/
