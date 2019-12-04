//***********************************************************************************************
//
//    Copyright (c) 1993-2002 Axon Instruments, Inc.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// MODULE:  CSYNCH.CPP
// PURPOSE: Contains member function code for the CSynch class.
// AUTHOR:  BHI  May 1994
// NOTES:
//

#include "../Common/wincpp.hpp"
#include "./csynch.hpp"

#include "../Common/FileIO.hpp"
#include "./abfutil.h"

#if defined(__UNIX__) || defined(__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

//===============================================================================================
// PROCEDURE: _Initialize
// PURPOSE:   Internal initialization routine.
//
void CSynch::_Initialize()
{
#ifdef _WINDOWS
   m_szFileName[0] = '\0';                // Filename for array virtualization
#endif
   m_hfSynchFile = NULL;  // Handle to temporary file.
   m_eMode       = eWRITEMODE;            // Mode flag for buffering algorithm.

   m_uSynchCount = 0;                     // Total count of entries in the synch array
   m_uCacheCount = 0;                     // Count of entries in the cache
   m_uCacheStart = 0;                     // Number of first entry in the cache.

   memset(m_SynchBuffer, 0, sizeof(m_SynchBuffer));  // Buffer for caching synch entries.
   memset(&m_LastEntry, 0, sizeof(m_LastEntry));     // Last entry written (write only).
}

//===============================================================================================
// PROCEDURE: CSynch
// PURPOSE:   Constructor.
//
CSynch::CSynch()
{
     /*MEMBERASSERT();*/    
   _Initialize();
}

//===============================================================================================
// PROCEDURE: ~CSynch
// PURPOSE:   Destructor. Closes the temporary file and deletes it.
//
CSynch::~CSynch()
{
     /*MEMBERASSERT();*/    
   CloseFile();
}


//===============================================================================================
// PROCEDURE: Clone
// PURPOSE:   Clone a passed CSynch array. Ownership is transfered of the temp file etc.
//
void CSynch::Clone(CSynch *pCS)
{
     /*MEMBERASSERT();*/    
   CloseFile();

   // Clone the settings.
//   strcpy(m_szFileName, pCS->m_szFileName);
   m_hfSynchFile = pCS->m_hfSynchFile;
   m_eMode       = pCS->m_eMode;
   m_uSynchCount = pCS->m_uSynchCount;
   m_uCacheCount = pCS->m_uCacheCount;
   m_uCacheStart = pCS->m_uCacheStart;
   m_LastEntry   = pCS->m_LastEntry;

   // Clone the data.
   memcpy(m_SynchBuffer, pCS->m_SynchBuffer, sizeof(m_SynchBuffer));

   // Initialize the source CSynch object so that it doesn't delete the backing file.
   pCS->_Initialize();
}

//===============================================================================================
// PROCEDURE: CreateFile
// PURPOSE:   Gets a unique filename and opens it as a temporary file.
//
BOOL CSynch::OpenFile()
{
	/*MEMBERASSERT();*/
   _Initialize();
#ifndef _WINDOWS
   // Create the temporary file.
   m_hfSynchFile = tmpfile();
   ASSERT(m_hfSynchFile != FILE_NULL);
   return (m_hfSynchFile != NULL);
#else
   // Get a unique temporary file name.
//   AXU_GetTempFileName("synch", 0, m_szFileName);
   ABFU_GetTempFileName("synch", 0, m_szFileName);

   // Create the temporary file.
   m_hfSynchFile = CreateFileA(m_szFileName, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, NULL);
   ASSERT(m_hfSynchFile != INVALID_HANDLE_VALUE);
   return (m_hfSynchFile != INVALID_HANDLE_VALUE);
#endif
}

//===============================================================================================
// PROCEDURE: CloseFile
// PURPOSE:   Closes the file if it was opened previously.
//
void CSynch::CloseFile()
{
     /*MEMBERASSERT();*/    
   if (m_hfSynchFile != NULL)
   {
#ifndef _WINDOWS
	   c_CloseHandle(m_hfSynchFile);
#else
	   CloseHandle(m_hfSynchFile);
#endif
	   m_hfSynchFile = NULL;
   }
   _Initialize();
}

//===============================================================================================
// PROCEDURE: SetMode
// PURPOSE:   Sets the buffering mode of the CSynch object.
//
void CSynch::SetMode(eMODE eMode)
{
     /*MEMBERASSERT();*/    
   if ((m_eMode==eMode) || !_IsFileOpen())
      return;

   // If the old mode was for writing, flush the cache to disk.
   if (m_eMode==eWRITEMODE)
      _Flush();

   // Set the new mode.
   m_eMode = eMode;
   m_uCacheStart = m_uSynchCount;

   // If the new mode is for writing, preload the cache with the last n entries.
   if (m_eMode==eWRITEMODE)
   {
      UINT uCount = SYNCH_BUFFER_SIZE;

      if (m_uSynchCount < SYNCH_BUFFER_SIZE)
      {
         m_uCacheStart = 0;
         uCount = m_uSynchCount;
      }
      else
         m_uCacheStart = m_uSynchCount - SYNCH_BUFFER_SIZE;

      // Read the data out of the file.
      Read( m_SynchBuffer, m_uCacheStart, uCount );

      // Set the current position to the start of the bit we last read, and truncate the file here.
      c_SetFilePointer(m_hfSynchFile, m_uCacheStart * sizeof(Synch), NULL, FILE_BEGIN);
      //TRACE1( "CSynch::SetMode file position is %d.\n",
      //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
//      VERIFY(SetEndOfFile(m_hfSynchFile));

      m_uCacheCount = uCount;
      m_LastEntry   = m_SynchBuffer[uCount-1];
   }
   else
   {
      // Set the start index of the cache to the count of items in the file to
      // cause the cache to be invalid and filled on the next get.
      m_uCacheStart = m_uSynchCount;
   }
}


//===============================================================================================
// PROCEDURE: _GetWriteMode
// PURPOSE:   Retrieves synch entries from the virtualized array.
//
BOOL CSynch::_GetWriteMode( UINT uFirstEntry, Synch *pSynch, UINT uEntries )
{
     /*MEMBERASSERT();*/    
   ASSERT(uFirstEntry+uEntries <= m_uSynchCount);
   ASSERT(uEntries > 0);
   /*ARRAYASSERT(pSynch, uEntries);*/
   ASSERT(m_eMode == eWRITEMODE);

   // If just the last entry is required, return it and get out.
   if (uFirstEntry == m_uSynchCount-1)
   {
      *pSynch = m_LastEntry;
      return TRUE;
   }

   // If the block requested is not contained completely in the cache, 
   // read the file for it, reading straight into the passed buffer.
   if (m_uSynchCount - uFirstEntry > SYNCH_BUFFER_SIZE)
   {
      // Rather than checking whether the file has been opened in this case
      // we will just assert that this is so here. This means that it is the
      // responsibility of the caller to ensure that synch entries are not
      // requested outside the synch buffer if the cache is not backed by a file.
      ASSERT(m_hfSynchFile != FILE_NULL);

      // Calculate how many entries there are from the requested first entry until
      // we hit the ones currently in the cache.
      UINT uCount = m_uSynchCount - uFirstEntry - SYNCH_BUFFER_SIZE;
      
      // Limit the count to no greater than the requested amount.
      if (uCount > uEntries)
         uCount = uEntries;
      
      // Read the data out of the file.
      if( !Read( pSynch, uFirstEntry, uCount) )
         return FALSE;

      // Update pointers and counters.
      pSynch      += uCount;
      uFirstEntry += uCount;
      uEntries    -= uCount;

      if (uEntries == 0)
         return TRUE;
   }
   
   // Transfer the part of the buffer that is "invalidated", i.e. about to be overwritten.
   if (uFirstEntry < m_uCacheStart)
   {
      UINT uCount = m_uCacheStart - uFirstEntry;
      ASSERT(uCount <= SYNCH_BUFFER_SIZE - m_uCacheCount);
      Synch *pS = m_SynchBuffer + SYNCH_BUFFER_SIZE - uCount;
      if (uCount > uEntries)
         uCount = uEntries;
      memcpy(pSynch, pS, uCount*sizeof(Synch));
      pSynch      += uCount;
      uFirstEntry += uCount;
      uEntries    -= uCount;
      if (uEntries == 0)
         return TRUE;
   }

   // Transfer the more recently written part of the cache.
   ASSERT(uFirstEntry >= m_uCacheStart);
   ASSERT(uFirstEntry - m_uCacheStart + uEntries <= m_uCacheCount);
   memcpy(pSynch, m_SynchBuffer + uFirstEntry - m_uCacheStart, uEntries*sizeof(Synch));
   return TRUE;
}


//===============================================================================================
// PROCEDURE: _GetReadMode
// PURPOSE:   Retrieves synch entries from the virtualized array.
//
BOOL CSynch::_GetReadMode( UINT uFirstEntry, Synch *pSynch, UINT uEntries )
{
     /*MEMBERASSERT();*/    
   ASSERT(m_hfSynchFile != FILE_NULL);
   ASSERT(uEntries > 0);
   /*ARRAYASSERT(pSynch, uEntries);*/
   ASSERT(uFirstEntry+uEntries <= m_uSynchCount);
   ASSERT(m_eMode == eREADMODE);

   // Loop until the get() has been satisfied.
   while (uEntries)
   {
      // If the first entry is not in the cache, reload the cache.
      if ((uFirstEntry < m_uCacheStart) || (uFirstEntry >= m_uCacheStart + m_uCacheCount))
      {
         m_uCacheStart = uFirstEntry - (uFirstEntry % SYNCH_BUFFER_SIZE);
         m_uCacheCount  = m_uSynchCount - m_uCacheStart;
         if (m_uCacheCount > SYNCH_BUFFER_SIZE)
            m_uCacheCount = SYNCH_BUFFER_SIZE;
         Read( m_SynchBuffer, m_uCacheStart, m_uCacheCount );
      }

      // Calculate how many entries intersect the cache.

      UINT uCount = min(uEntries, m_uCacheCount);

      // Copy the entries out of the cache.
      memcpy(pSynch, m_SynchBuffer+uFirstEntry-m_uCacheStart, uCount*sizeof(Synch));
      uFirstEntry += uCount;
      pSynch      += uCount;
      uEntries    -= uCount;
   }
   return TRUE;
}


//===============================================================================================
// PROCEDURE: _Flush
// PURPOSE:   Flushes the Synch cache to disk.
//
BOOL CSynch::_Flush()
{
     /*MEMBERASSERT();*/    
   ASSERT(m_eMode==eWRITEMODE);
   if (m_uCacheCount == 0)
      return TRUE;

   BOOL bRval = TRUE;
   DWORD dwBytesWritten = 0;
   if (_IsFileOpen())
   {
      //TRACE1( "CSynch::_Flush current file pointer is %d on entry.\n",
      //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );

      // Write out the current contents of the cache.
      UINT  uBytesToWrite = m_uCacheCount * sizeof(Synch);
      bRval = c_WriteFile(m_hfSynchFile, m_SynchBuffer, uBytesToWrite, &dwBytesWritten, NULL);
      
      //TRACE1( "CSynch::_Flush current file pointer is %d after WriteFile.\n",
      //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
   }
   // If a disk full error occurs, save what was actually written and rotate the buffer
   // ready for the next attempt(?).
   if (!bRval)
   {
      UINT uEntriesWritten   = dwBytesWritten/sizeof(Synch);
      UINT uEntriesUnwritten = m_uCacheCount - uEntriesWritten;

      Synch *pTemp = new Synch[uEntriesWritten];
      if (pTemp)
         memcpy(pTemp, m_SynchBuffer, uEntriesWritten*sizeof(Synch));
      for (UINT i=0; i<uEntriesUnwritten; i++)
         m_SynchBuffer[i] = m_SynchBuffer[uEntriesWritten+i];
      if (pTemp)
      {
         memcpy(m_SynchBuffer+uEntriesUnwritten, pTemp, uEntriesWritten*sizeof(Synch));
         delete[] pTemp;
      }
      m_uCacheCount = uEntriesUnwritten;
      m_uCacheStart += uEntriesWritten;
   }
   else
   {
      m_uCacheCount = 0;
      m_uCacheStart = m_uSynchCount;
   }
   return bRval;
}

/*
//===============================================================================================
// PROCEDURE: _PackBuffer
// PURPOSE:   Pack the entries in the buffer down, removing the dwFileOffset member.
//
BOOL CSynch::_PackBuffer(UINT uAcquiredSamples, UINT &uEntries, UINT uSampleSize)
{
   MEMBERASSERT();
   ASSERT(uEntries <= SYNCH_BUFFER_SIZE);
   DWORD *pdwDest = (DWORD *)m_SynchBuffer;
   Synch *pS = m_SynchBuffer;
   for (UINT i=0; i<uEntries; i++)
   {
      // Check whether the synch entry refers to data outside that which was actually saved.
      if (pS->dwFileOffset/uSampleSize + pS->dwLength > uAcquiredSamples)
      {
         uEntries = i;
         return FALSE;
      }
      *pdwDest++ = pS->dwStart;
      *pdwDest++ = pS->dwLength;
      pS++;
   }
   return TRUE;
}


//===============================================================================================
// PROCEDURE: Write
// PURPOSE:   Copies the complete synch array to another file, packing out the file-offset entry.
//
BOOL CSynch::Write( HANDLE hDataFile, UINT uAcquiredSamples, UINT *puSynchCount, UINT uSampleSize )
{
   MEMBERASSERT();
   ASSERT( hDataFile != INVALID_HANDLE_VALUE );
   WPTRASSERT(puSynchCount);

   // Flush any cached Synch entries to the temp file. This should not fail as the reserve file
   // will have been released just prior to calling this function. If it does fail, we will
   // still be able to work with the Synch entries that were saved ok.
   if (m_uCacheCount)
      _Flush();

   // Set the return value for the number of synch entries. If none exist, return.
   *puSynchCount = 0;
   if (m_uSynchCount == 0)
      return TRUE;

   // Seek to the end of the passed file. This will only fail for invalid file handles.
   CFileIO_NoClose OutFile(hDataFile);
   LONGLONG llCurrentPos = 0;
   if (!OutFile.Seek(0, FILE_END, &llCurrentPos))
      return FALSE;

   // Seek to the start of the temporary file.
   SetFilePointer(m_hfSynchFile, 0L, NULL, FILE_BEGIN);

   // Read the Synch data in a buffer at a time and write it out to the passed file.
   UINT uEntries = m_uSynchCount;
   UINT uWritten = 0;
   UINT uCount = 0;
   while ( uEntries > 0 )
   {
      uCount = min(uEntries, SYNCH_BUFFER_SIZE);
   
      // Read in a buffer from the temp file.
      VERIFY(Read( m_SynchBuffer, uWritten, uCount));      

      // Pack the buffer, removing the dwFileOffset members and checking for invalid synch entries.
      // If an invalid entry is found, the count is truncated at the last valid entry.
      if (!_PackBuffer(uAcquiredSamples, uCount, uSampleSize))
         uEntries = uCount;
      
      // Write the packed buffer out to the temp file.
      if ( !OutFile.Write( m_SynchBuffer, uCount * 2 * sizeof(DWORD) ))
      {
         // If an error occurs, go back to the start of the block and truncate the file
         // ready for the next attempt after the user has freed up some disk space.
         VERIFY(OutFile.Seek(llCurrentPos, FILE_BEGIN));
         VERIFY(OutFile.SetEndOfFile());
         return FALSE;
      }
      
      uEntries -= uCount;
      uWritten += uCount;
   }
   
   // Seek back to end of the temporary file.
   SetFilePointer(m_hfSynchFile, 0L, NULL, FILE_END);
   //TRACE1( "CSynch::Write current file pointer is %d after seek to end.\n",
   //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
   *puSynchCount = uWritten;
   return TRUE;
}
*/
//===============================================================================================
// PROCEDURE: Put
// PURPOSE:   Puts a new Synch entry into the synch array, flushing to disk if full.
//
BOOL CSynch::Put( UINT uStart, UINT uLength, UINT uOffset )
{
     /*MEMBERASSERT();*/    
   ASSERT(m_eMode==eWRITEMODE);
   ASSERT((m_uSynchCount == 0) || (m_LastEntry.dwStart <= uStart));

   // Flush the cache if it is full.
   if ((m_uCacheCount >= SYNCH_BUFFER_SIZE) && (!_Flush()))
      return FALSE;

   // If a value of zero is passed as the file offset, the file offset for this
   // entry is derived from the previous one.      
   if (uOffset == 0)
      m_LastEntry.dwFileOffset += m_LastEntry.dwLength * 2;
   else
      m_LastEntry.dwFileOffset = uOffset;
      
   m_LastEntry.dwStart  = uStart;
   m_LastEntry.dwLength = uLength;
   m_SynchBuffer[m_uCacheCount++] = m_LastEntry;
   m_uSynchCount++;
   return TRUE;
}
/*
//===============================================================================================
// PROCEDURE: Update
// PURPOSE:   Updates an entry in the SynchArray.
//
BOOL CSynch::Update( UINT uEntry, const Synch *pSynch )
{
   ASSERT(uEntry < m_uSynchCount);
   ASSERT(m_eMode!=eREADMODE);
   ASSERT(m_hfSynchFile != INVALID_HANDLE_VALUE);

   // Save current file position.
   long lCurrentPos = SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT);
   //TRACE1( "CSynch::Update current file pointer is %d on entry.\n",
   //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
   if (lCurrentPos == INVALID_SEEK_VALUE)
      return FALSE;

   // Seek to the entry.
   SetFilePointer(m_hfSynchFile, uEntry * sizeof(Synch), NULL, FILE_BEGIN);
   //TRACE1( "CSynch::Update current file pointer is %d after seek to current entry.\n",
   //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
   
   // Write the data out of the file.
   DWORD dwBytesWritten = 0;
   BOOL bOK = WriteFile(m_hfSynchFile, pSynch, sizeof(pSynch), &dwBytesWritten, NULL);

   // Reset the pointer to the original position.
   SetFilePointer(m_hfSynchFile, lCurrentPos, NULL, FILE_BEGIN);
   //TRACE1( "CSynch::Update current file pointer is %d after seek to original position.\n",
   //         SetFilePointer(m_hfSynchFile, 0, NULL, FILE_CURRENT) );
   
   if (!bOK)
      return FALSE;
   
   // If the entry requested is contained in the cache, update the cache.
   if (m_uSynchCount - uEntry <= SYNCH_BUFFER_SIZE)
   {
      int nIndex = int(uEntry - m_uCacheStart);
      if (nIndex < 0)
         nIndex += SYNCH_BUFFER_SIZE;

      ASSERT(nIndex >= 0);
      ASSERT(nIndex < SYNCH_BUFFER_SIZE);
      m_SynchBuffer[nIndex] = *pSynch;

      if (uEntry==m_uSynchCount-1)
         m_LastEntry = *pSynch;
   }
   return TRUE;
}

//===============================================================================================
// PROCEDURE: UpdateLength
// PURPOSE:   Updates the length of the last entry put in the SynchArray.
//
void CSynch::UpdateLength(DWORD dwLength)
{
   MEMBERASSERT();
   ASSERT(m_eMode==eWRITEMODE);
   ASSERT(m_uCacheCount > 0);
   m_LastEntry.dwLength = dwLength;
   m_SynchBuffer[m_uCacheCount-1] = m_LastEntry;
}

//===============================================================================================
// PROCEDURE: IncreaseLastLength
// PURPOSE:   Increases the length of the last entry put in the SynchArray.
//
void CSynch::IncreaseLastLength( DWORD dwIncrease )
{
   MEMBERASSERT();
   ASSERT(m_eMode==eWRITEMODE);
   ASSERT(m_uCacheCount > 0);
   m_LastEntry.dwLength += dwIncrease;
   m_SynchBuffer[m_uCacheCount-1] = m_LastEntry;
}

//===============================================================================================
// PROCEDURE: GetLastEntry
// PURPOSE:   Returns the last Synch structure added to the synch array.
//
BOOL CSynch::GetLastEntry(Synch *pSynch)
{
   MEMBERASSERT();
   WPTRASSERT(pSynch);
   if (!m_uSynchCount)
      return FALSE;

   *pSynch = m_LastEntry;
   return TRUE;
}
*/
