//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//
//***********************************************************************************************
// HEADER:  CSYNCH.HPP
// PURPOSE: Contains cclass definition for CSynch for creating and maintaining a Synch array.
// AUTHOR:  BHI  May 1994
//

#ifndef INC_CSYNCH_HPP
#define INC_CSYNCH_HPP

#include "../Common/axodefn.h"
#include "../Common/axodebug.h"

//-----------------------------------------------------------------------------------------------
// Local constants:

#define SYNCH_BUFFER_SIZE  100         // buffer 100 synch entries at a time.

// Synch structure definition.
struct Synch
{
   DWORD dwStart;
   DWORD dwLength;
   DWORD dwFileOffset;
};

//-----------------------------------------------------------------------------------------------
// CSynch class definition

class CSynch
{
public:
   enum eMODE { eWRITEMODE, eREADMODE };

private:    // Member variables.
#ifdef _WINDOWS
   char   m_szFileName[_MAX_PATH];            // Filename for array virtualization
#endif

   FILEHANDLE  m_hfSynchFile;                      // Handle to temporary file.
   eMODE  m_eMode;                            // Mode flag for buffering algorithm.

   UINT   m_uSynchCount;                     // Total count of entries in the synch array
   UINT   m_uCacheCount;                      // Count of entries in the cache
   UINT   m_uCacheStart;                     // Number of first entry in the cache.

   Synch  m_SynchBuffer[SYNCH_BUFFER_SIZE];   // Buffer for caching synch entries.
   Synch  m_LastEntry;                        // Last entry written (write only).

private:    // Declare but don't define copy constructor to prevent use of default
   CSynch(const CSynch &CS);
   const CSynch &operator=(const CSynch &CS);

private:    // Private member functions.
   void _Initialize();
   BOOL _Flush();
/*   BOOL _PackBuffer(UINT uAcquiredSamples, UINT &uEntries, UINT uSampleSize);
//   BOOL _Read(LPVOID lpBuf, DWORD dwBytesToRead);
*/   
   BOOL Read(LPVOID lpBuf, DWORD dwFilePos, DWORD dwEntriesToRead);
   BOOL _GetReadMode( UINT uFirstEntry, Synch *pSynch, UINT uEntries );
   BOOL _GetWriteMode( UINT uFirstEntry, Synch *pSynch, UINT uEntries );
   BOOL _IsFileOpen();
   
public:     // Public member functions
   CSynch();
   ~CSynch();
   void Clone(CSynch *pCS);

   BOOL  OpenFile();
   void  CloseFile();

   void  SetMode(eMODE eMode);
   
   BOOL  Put( UINT uStart, UINT uLength, UINT uOffset=0 );
/*   void  UpdateLength( DWORD dwLength );
   void  IncreaseLastLength( DWORD dwIncrease );
   BOOL  GetLastEntry(Synch *pSynch);
   
*/   BOOL  Get( UINT uFirstEntry, Synch *pSynch, UINT uEntries );
/*   BOOL  Update( UINT uEntry, const Synch *pSynch );
*/   UINT  GetCount() const;
/*   
   BOOL  Write( HANDLE hDataFile, UINT uAcquiredSamples, UINT *puSynchCount, UINT uSampleSize );
*/
};


//===============================================================================================
// PROCEDURE: GetCount
// PURPOSE:   Returns the current count of synch elements.
//
inline UINT CSynch::GetCount() const
{
//   MEMBERASSERT();
   return m_uSynchCount;
}

//===============================================================================================
// PROCEDURE: Read
// PURPOSE:   Reads a block and returns FALSE on ERROR.
//
inline BOOL CSynch::Read(LPVOID lpBuf, DWORD dwFirstEntry, DWORD dwEntriesToRead)
{
//   MEMBERASSERT();

   DWORD dwBytesToRead= dwEntriesToRead * sizeof(Synch);
//   ARRAYASSERT(LPSTR(lpBuf), dwBytesToRead);

   // Save the current file position and seek to the desired position.
   DWORD dwCurrentPos = c_SetFilePointer( m_hfSynchFile, 0, NULL, FILE_CURRENT );

   if (dwCurrentPos == INVALID_SEEK_VALUE)
      return FALSE;

   c_SetFilePointer( m_hfSynchFile, dwFirstEntry * sizeof(Synch), NULL, FILE_BEGIN );

   DWORD dwBytesRead = 0;
   BOOL bOK = c_ReadFile(m_hfSynchFile, lpBuf, dwBytesToRead, &dwBytesRead, NULL);

   // Restore the original file.
   c_SetFilePointer( m_hfSynchFile, dwCurrentPos, NULL, FILE_BEGIN );

   // Debug error messages.
   if( !bOK )
   {
      TRACE( "CSynch::Read - ReadFile failed");// with the following error:\n" );
//      SHOW_SYSTEM_ERROR( GetLastError() );
   }

   if( dwBytesRead != dwBytesToRead ) {
//      TRACE2( "CSynch::Read - error reading from file:  dwBytesToRead %d, dwBytesRead %d.\n",
//               dwBytesToRead, dwBytesRead );
   }
   return (bOK && (dwBytesRead==dwBytesToRead));
}

//===============================================================================================
// PROCEDURE: Get
// PURPOSE:   Retrieves synch entries from the virtualized array.
//
inline BOOL CSynch::Get( UINT uFirstEntry, Synch *pSynch, UINT uEntries )
{
//   MEMBERASSERT();
//   ASSERT(uEntries > 0);
//   ARRAYASSERT(pSynch, uEntries);
//   ASSERT(uFirstEntry+uEntries <= m_uSynchCount);
   if (m_eMode == eREADMODE)
      return _GetReadMode( uFirstEntry, pSynch, uEntries );
   else
      return _GetWriteMode( uFirstEntry, pSynch, uEntries );
}

//===============================================================================================
// PROCEDURE: _IsFileOpen
// PURPOSE:   Returns TRUE if the temp file was opened ok.
//
inline BOOL CSynch::_IsFileOpen()
{
   return (m_hfSynchFile != FILE_NULL);
}

#endif      // INC_CSYNCH_HPP
