//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  FILEDESC.CPP
// PURPOSE: Contains the member functions for the CFileDescriptor class.
// 

#include "../Common/wincpp.hpp"
#include "filedesc.hpp"

//===============================================================================================
// FUNCTION: Constructor
// PURPOSE:  Initialize the object
//
CFileDescriptor::CFileDescriptor() 
{
   //MEMBERASSERT();
   m_uLastEpiSize       = 0;
   m_uFlags             = 0;
   m_pvReadBuffer       = NULL;
   m_uCachedEpisode     = UINT(-1);
   m_uCachedEpisodeSize = 0;
   m_uAcquiredEpisodes  = 0;
   m_uAcquiredSamples   = 0;
   m_bHasOverlappedData = FALSE;
   m_nLastError         = 0;
   m_szFileName[0]      = '\0';
}

//===============================================================================================
// FUNCTION: Destructor
// PURPOSE:  Cleanup the object when it is deleted.
//
CFileDescriptor::~CFileDescriptor()
{
   //MEMBERASSERT();
   FreeReadBuffer();
}

//===============================================================================================
// FUNCTION: SetFlag
// PURPOSE:  Set the given flags.
//
void CFileDescriptor::SetFlag(UINT uFlag)
{
   //MEMBERASSERT();
   m_uFlags |= uFlag;
}

//===============================================================================================
// FUNCTION: TestFlag
// PURPOSE:  Test if the given flags are set.
//
BOOL CFileDescriptor::TestFlag(UINT uFlag)
{
   //MEMBERASSERT();
   return (m_uFlags & uFlag) != 0;
}

//===============================================================================================
// FUNCTION: Open
// PURPOSE:  Opens an existing file for read or write access.
//
BOOL CFileDescriptor::Open(const char *szFileName, BOOL bReadOnly)
{
#ifdef _WINDOWS
   MEMBERASSERT();
//   LPSZASSERT(szFileName);
#endif
   if (!m_File.Create(szFileName, bReadOnly))
   {
      BOOL bTooManyFiles = (m_File.GetLastError()==ERROR_TOO_MANY_OPEN_FILES);
      return SetLastError(bTooManyFiles ? ABF_NODOSFILEHANDLES : ABF_EOPENFILE);
   }
   m_uFlags = bReadOnly ? FI_READONLY : FI_WRITEONLY;
   strncpy(m_szFileName, szFileName, _MAX_PATH-1);
   m_szFileName[_MAX_PATH-1] = '\0';

   if (!m_VSynch.OpenFile())
      return SetLastError(ABF_BADTEMPFILE);

/*   if (!bReadOnly)
   {
      if (!m_Tags.Initialize(sizeof(ABFTag), CACHE_SIZE))
         return SetLastError(ABF_BADTEMPFILE);
      if (!m_Deltas.Initialize(sizeof(ABFDelta), CACHE_SIZE))
         return SetLastError(ABF_BADTEMPFILE);
      for( int i=0; i<ABF_WAVEFORMCOUNT; i++ )
         if (!m_DACFile[i].OpenFile())
            return SetLastError(ABF_BADTEMPFILE);
   }
*/
   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: Reopen
// PURPOSE:  Reopen an existing file for read or write access.
//
BOOL CFileDescriptor::Reopen(BOOL bReadOnly)
{
   //MEMBERASSERT();
   BOOL bIsReadOnly = ((m_uFlags & FI_READONLY) != 0);
   if (bIsReadOnly == bReadOnly)
      return TRUE;

   m_File.Close();
   if (!m_File.Create(m_szFileName, bReadOnly))
   {
      BOOL bTooManyFiles = (m_File.GetLastError()==ERROR_TOO_MANY_OPEN_FILES);

      m_File.Create(m_szFileName, !bReadOnly);
      return SetLastError(bTooManyFiles ? ABF_NODOSFILEHANDLES : ABF_EOPENFILE);
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: GetFileName
// PURPOSE:  Return the name of the file.
//
LPCSTR CFileDescriptor::GetFileName() const
{
   //MEMBERASSERT();
   return m_szFileName;
}


//===============================================================================================
// FUNCTION: FillToNextBlock
// PURPOSE:  Pads the data file out to the next ABF_BLOCKSIZE byte boundary.
//
BOOL CFileDescriptor::FillToNextBlock( long *plBlockNum )
{
   WPTRASSERT(plBlockNum);
   
   LONGLONG llOffset = 0;
   VERIFY(m_File.Seek(0L, FILE_END, &llOffset));
   UINT uFillLastBlock = ABF_BLOCKSIZE - UINT(llOffset % ABF_BLOCKSIZE);

   if (uFillLastBlock != ABF_BLOCKSIZE)
   {
      BYTE cBuffer[ABF_BLOCKSIZE] = {0};
      if (!Write( cBuffer, uFillLastBlock ))
         return FALSE;
      llOffset += uFillLastBlock;
   }
   *plBlockNum = long(llOffset / ABF_BLOCKSIZE);
   return TRUE;
}


//===============================================================================================
// FUNCTION: Write
// PURPOSE:  Write a buffer of data to the file.
//
BOOL CFileDescriptor::Write(const void *pvBuffer, UINT uSizeInBytes)
{
   //MEMBERASSERT();
   DWORD dwBytesWritten;
   while (!m_File.Write( pvBuffer, uSizeInBytes, &dwBytesWritten ))
   {
      // If the write failed, step the file pointer back to the start of the write.
      VERIFY(m_File.Seek( -long(dwBytesWritten), FILE_CURRENT));
      m_File.SetEndOfFile();
      SetLastError(ABF_EDISKFULL);

      // Notify the user through the callback function. If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: SetEndOfFile
// PURPOSE:  Truncates the file at the current position.
//
BOOL CFileDescriptor::SetEndOfFile()
{
   //MEMBERASSERT();
   return m_File.SetEndOfFile();
}
*/
//===============================================================================================
// FUNCTION: EpisodeStart
// PURPOSE:  Gets the episode start value for a particular episode.
//
UINT CFileDescriptor::EpisodeStart(UINT uEpisode)
{
   //MEMBERASSERT();
   ASSERT(uEpisode > 0);
   Synch SynchEntry;
   m_VSynch.Get(uEpisode-1, &SynchEntry, 1);
   return SynchEntry.dwStart;
}
/*
//===============================================================================================
// FUNCTION: SetEpisodeStart
// PURPOSE:  Sets the episode start time for a particular episode.
//
void CFileDescriptor::SetEpisodeStart(UINT uEpisode, UINT uSynchTime)
{
   //MEMBERASSERT();
   ASSERT(uEpisode > 0);
   Synch SynchEntry;
   VERIFY(m_VSynch.Get(uEpisode-1, &SynchEntry, 1));
   SynchEntry.dwStart = uSynchTime;
   VERIFY(m_VSynch.Update(uEpisode-1, &SynchEntry));
}
*/
//===============================================================================================
// FUNCTION: EpisodeLength
// PURPOSE:  Gets the episode length value for a particular episode.
//
UINT CFileDescriptor::EpisodeLength(UINT uEpisode)
{
   //MEMBERASSERT();
   ASSERT(uEpisode > 0);
   Synch SynchEntry;
   m_VSynch.Get(uEpisode-1, &SynchEntry, 1);
   return SynchEntry.dwLength;
}

//===============================================================================================
// FUNCTION: CheckEpisodeNumber
// PURPOSE:  Checks that the given episode number is valid.
//
BOOL CFileDescriptor::CheckEpisodeNumber(UINT uEpisode)
{
   //MEMBERASSERT();
   return (uEpisode > 0) && (uEpisode <= m_uAcquiredEpisodes);
}

//===============================================================================================
// FUNCTION: SetAcquiredEpisodes
// PURPOSE:  Sets the count of acquired episodes.
//
void CFileDescriptor::SetAcquiredEpisodes(UINT uEpisodes)
{
   //MEMBERASSERT();
   m_uAcquiredEpisodes = uEpisodes;
}

//===============================================================================================
// FUNCTION: GetAcquiredEpisodes
// PURPOSE:  Gets the count of acquired episodes.
//
UINT CFileDescriptor::GetAcquiredEpisodes() const
{
   //MEMBERASSERT();
   return m_uAcquiredEpisodes;
}

//===============================================================================================
// FUNCTION: SetAcquiredSamples
// PURPOSE:  Sets the count of acquired samples.
//
void CFileDescriptor::SetAcquiredSamples(UINT uSamples)
{
   //MEMBERASSERT();
   m_uAcquiredSamples = uSamples;
}
/*
//===============================================================================================
// FUNCTION: GetAcquiredSamples
// PURPOSE:  Gets the count of acquired samples.
//
UINT CFileDescriptor::GetAcquiredSamples() const
{
   //MEMBERASSERT();
   return m_uAcquiredSamples;
}

*/
//===============================================================================================
// FUNCTION: AllocReadBuffer
// PURPOSE:  Allocates the read buffer and resets the cached episode.
//
BOOL CFileDescriptor::AllocReadBuffer(UINT uBytes)
{
   //MEMBERASSERT();
   FreeReadBuffer();
   m_pvReadBuffer = malloc(uBytes);
   return (m_pvReadBuffer!=NULL);
}

//===============================================================================================
// FUNCTION: FreeReadBuffer
// PURPOSE:  Frees the read buffer and resets the cached episode.
//
void CFileDescriptor::FreeReadBuffer()
{
   //MEMBERASSERT();
   if (m_pvReadBuffer != NULL)
      free(m_pvReadBuffer);
   m_pvReadBuffer = NULL;
   m_uCachedEpisode = UINT(-1);
   m_uCachedEpisodeSize = 0;
}

//===============================================================================================
// FUNCTION: SetCachedEpisode
// PURPOSE:  Sets the count and size of the cached episode.
//
void CFileDescriptor::SetCachedEpisode(UINT uEpisode, UINT uEpisodeSize)
{
   //MEMBERASSERT();
   m_uCachedEpisode     = uEpisode;
   m_uCachedEpisodeSize = uEpisodeSize;
}

//===============================================================================================
// FUNCTION: GetCachedEpisode
// PURPOSE:  Gets the count of the cached episode.
//
UINT CFileDescriptor::GetCachedEpisode() const
{
   //MEMBERASSERT();
   return m_uCachedEpisode;
}

//===============================================================================================
// FUNCTION: GetCachedEpisodeSize
// PURPOSE:  Gets the size of the cached episode.
//
UINT CFileDescriptor::GetCachedEpisodeSize() const
{
   //MEMBERASSERT();
   return m_uCachedEpisodeSize;
}

//===============================================================================================
// FUNCTION: SetLastEpiSize
// PURPOSE:  Sets the size of the last episode.
//
void CFileDescriptor::SetLastEpiSize(UINT uEpiSize)
{
   //MEMBERASSERT();
   m_uLastEpiSize = uEpiSize;
}

//===============================================================================================
// FUNCTION: GetLastEpiSize
// PURPOSE:  Gets the size of the last episode.
//
UINT CFileDescriptor::GetLastEpiSize() const
{
   //MEMBERASSERT();
   return m_uLastEpiSize;
}
/*
//===============================================================================================
// FUNCTION: FileOffset
// PURPOSE:  Gets the file offset for a particular episode.
//
UINT CFileDescriptor::FileOffset( UINT uEpisode)
{
   //MEMBERASSERT();
   ASSERT(uEpisode > 0);
   Synch SynchEntry;
   m_VSynch.Get(uEpisode-1, &SynchEntry, 1);
   return SynchEntry.dwFileOffset;
}

//===============================================================================================
// FUNCTION: WriteSynchArray
// PURPOSE:  Writes the synch array out to disk.
//
BOOL CFileDescriptor::WriteSynchArray( long *plBlockNum, long *plCount, UINT uSampleSize )
{
   //MEMBERASSERT();
   *plBlockNum = 0;
   *plCount    = GetSynchCount();
   if (*plCount==0)
      return TRUE;

   // Try to write the synch array out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   UINT uCount = 0;
   while (!FillToNextBlock( plBlockNum ) || 
          !m_VSynch.Write( GetFileHandle(), m_uAcquiredSamples, &uCount, uSampleSize ))
   {
      // Notify the user through the callback function. If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }

   *plCount            = uCount;
   m_uAcquiredEpisodes = uCount;
   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: SynchCount
// PURPOSE:  Gets the count of items in the synch array.
//
UINT CFileDescriptor::GetSynchCount() const
{
   //MEMBERASSERT();
   return m_VSynch.GetCount();
}

//===============================================================================================
// FUNCTION: SetSynchMode
// PURPOSE:  Sets the buffering mode of the virtualized synch array.
//
void CFileDescriptor::SetSynchMode(CSynch::eMODE eMode)
{
   //MEMBERASSERT();
   m_VSynch.SetMode(eMode);
}

/*
//===============================================================================================
// FUNCTION: GetSynchObject
// PURPOSE:  Gets a pointer to the synch object (violating encapsulation!). 
//
CSynch *CFileDescriptor::GetSynchObject()
{
   //MEMBERASSERT();
   return &m_VSynch;
}
*/
//===============================================================================================
// FUNCTION: ChangeSynchArray
// PURPOSE:  Swaps in an alternate synch array. USE WITH CARE!
//
void CFileDescriptor::ChangeSynchArray(CSynch *pNewSynch)
{
   //MEMBERASSERT();
   m_VSynch.Clone(pNewSynch);
}
/*
//===============================================================================================
// FUNCTION: PutTag
// PURPOSE:  Puts a tag entry into the virtual tag array.
//
BOOL CFileDescriptor::PutTag( const ABFTag *pTag )
{
   //MEMBERASSERT();
   if (!m_Tags.Put( pTag ))
      return SetLastError(ABF_EDISKFULL);
   return TRUE;
}

//===============================================================================================
// FUNCTION: TagCount
// PURPOSE:  Returns the number of tags in the tag array.
//
UINT CFileDescriptor::GetTagCount() const
{
   //MEMBERASSERT();
   return m_Tags.GetCount();
}

//===============================================================================================
// FUNCTION: WriteTags
// PURPOSE:  Writes the tag array out to disk.
//
BOOL CFileDescriptor::WriteTags( long *plBlockNum, long *plCount )
{
   //MEMBERASSERT();
   *plBlockNum = 0;
   *plCount    = m_Tags.GetCount();
   if (*plCount==0)
      return TRUE;

   // Try to write the tags out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   while (!FillToNextBlock( plBlockNum ) || 
          !m_Tags.Write( GetFileHandle() ))
   {
      // Notify the user through the callback function. 
      // If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: UpdateTag
// PURPOSE:  Updates an entry in the tag array.
//
BOOL CFileDescriptor::UpdateTag( UINT uTag, const ABFTag *pTag)
{
   //MEMBERASSERT();
   WPTRASSERT( pTag );
   return m_Tags.Update( uTag, pTag );
}

//===============================================================================================
// FUNCTION: ReadTags
// PURPOSE:  Reads a sub section of the tag array.
//
BOOL CFileDescriptor::ReadTags( UINT uFirstTag, ABFTag *pTagArray, UINT uNumTags)
{
   //MEMBERASSERT();
   ARRAYASSERT( pTagArray, uNumTags );
   return m_Tags.Get( uFirstTag, pTagArray, uNumTags );
}

//===============================================================================================
// FUNCTION: SaveVoiceTag
// PURPOSE:  Adds a voice tag descriptor to the pending list.
//
BOOL CFileDescriptor::SaveVoiceTag( LPCSTR pszFileName, long lDataOffset, ABFVoiceTagInfo *pVTI)
{
   CVoiceTag *pVoiceTag = new CVoiceTag( pszFileName, lDataOffset, pVTI );
   if (!pVoiceTag)
      return SetLastError(ABF_OUTOFMEMORY);
   m_VoiceTagList.Put(pVoiceTag);
   return TRUE;
}


//===============================================================================================
// FUNCTION: WriteVoiceTags
// PURPOSE:  Create a new catalog for ABFVoiceTagInfo structures and save the tags from the list.
//
BOOL CFileDescriptor::WriteVoiceTags( long *plBlockNum, long *plCount )
{
   //MEMBERASSERT();
   *plBlockNum = 0;
   *plCount    = m_VoiceTagList.GetCount();
   if (*plCount==0)
      return TRUE;

   // Try to write the tags out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   while (!FillToNextBlock( plBlockNum ) || 
          !m_VoiceTagList.Write(m_File, *plBlockNum, &m_Notify))
   {
      // Notify the user through the callback function. 
      // If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }

   return TRUE;
}


//===============================================================================================
// FUNCTION: GetVoiceTag
// PURPOSE:  Retrieves a voice tag into a new file, leaving space for a header.
//
BOOL CFileDescriptor::GetVoiceTag( UINT uTag, LPCSTR pszFileName, long lDataOffset, 
                                  ABFVoiceTagInfo *pVTI, long lVoiceTagPtr)
{
   //MEMBERASSERT();
   VERIFY(Seek(lVoiceTagPtr*ABF_BLOCKSIZE + uTag*sizeof(ABFVoiceTagInfo), FILE_BEGIN));
   if (!Read(pVTI, sizeof(ABFVoiceTagInfo)))
   {
      SetLastError(ABF_EREADDATA);
      return NULL;
   }   
   VERIFY(Seek(pVTI->lFileOffset, FILE_BEGIN));
   
   CVoiceTag VoiceTag( pszFileName, lDataOffset, pVTI);
   if (!VoiceTag.ReadTag( m_File ))
      return SetLastError(VoiceTag.GetLastError());
   return TRUE;
}


//===============================================================================================
// FUNCTION: PutDelta
// PURPOSE:  Puts a tag entry into the virtual Delta array.
//
BOOL CFileDescriptor::PutDelta( const ABFDelta *pDelta )
{
   //MEMBERASSERT();
   if (!m_Deltas.Put( pDelta ))
      return SetLastError(ABF_EDISKFULL);
   return TRUE;
}

//===============================================================================================
// FUNCTION: DeltaCount
// PURPOSE:  Returns the number of Deltas in the Delta array.
//
UINT CFileDescriptor::GetDeltaCount() const
{
   //MEMBERASSERT();
   return m_Deltas.GetCount();
}

//===============================================================================================
// FUNCTION: WriteDeltas
// PURPOSE:  Writes the Delta array out to disk.
//
BOOL CFileDescriptor::WriteDeltas( long *plBlockNum, long *plCount )
{
   //MEMBERASSERT();
   *plBlockNum = 0;
   *plCount    = m_Deltas.GetCount();
   if (*plCount==0)
      return TRUE;

   // Try to write the Deltas out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   while (!FillToNextBlock( plBlockNum ) || 
          !m_Deltas.Write( GetFileHandle() ))
   {
      // Notify the user through the callback function. 
      // If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadDeltas
// PURPOSE:  Reads a sub section of the Delta array.
//
BOOL CFileDescriptor::ReadDeltas( UINT uFirstDelta, ABFDelta *pDeltaArray, UINT uNumDeltas)
{
   //MEMBERASSERT();
   ARRAYASSERT( pDeltaArray, uNumDeltas );
   return m_Deltas.Get( uFirstDelta, pDeltaArray, uNumDeltas );
}

//===============================================================================================
// FUNCTION: PutDACFileSweep
// PURPOSE:  Puts a tag entry into the virtual Delta array.
//
BOOL CFileDescriptor::PutDACFileSweep( UINT uDACChannel, UINT uSweep, const DAC_VALUE *pnData, UINT uLength )
{
   //MEMBERASSERT();
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   if (!m_DACFile[uDACChannel].PutSweep( uSweep, pnData, uLength ))
      return SetLastError(ABF_EDISKFULL);
   return TRUE;
}

//===============================================================================================
// FUNCTION: GetDACFileSweepCount
// PURPOSE:  Returns the number of DAC file sweeps in the array.
//
UINT CFileDescriptor::GetDACFileSweepCount( UINT uDACChannel ) const
{
   //MEMBERASSERT();
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   return m_DACFile[uDACChannel].GetCount();
}

//===============================================================================================
// FUNCTION: WriteDACFileSweeps
// PURPOSE:  Writes the DAC file data out to disk.
//
BOOL CFileDescriptor::WriteDACFileSweeps( UINT uDACChannel, long *plBlockNum, long *plCount )
{
   //MEMBERASSERT();
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Coerce to safe value.
   if( uDACChannel >= ABF_WAVEFORMCOUNT )
   {
      TRACE1( "WriteDACFileSweeps: uDACChannel changed from %d to 0.\n", uDACChannel );
      uDACChannel = 0;
   }

   *plBlockNum = 0;
   *plCount    = m_DACFile[uDACChannel].GetCount();
   if (*plCount==0)
      return TRUE;

   // Try to write the Deltas out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   while (!FillToNextBlock( plBlockNum ) || 
          !m_DACFile[uDACChannel].Write( GetFileHandle() ))
   {
      // Notify the user through the callback function. 
      // If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: GetDACFileSweep
// PURPOSE:  Reads a sweep from the DAC file array.
//
BOOL CFileDescriptor::GetDACFileSweep(UINT uDACChannel, UINT uSweep, DAC_VALUE *pnData, UINT uMaxLength)
{
   //MEMBERASSERT();
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   ARRAYASSERT( pnData, uMaxLength );

   return m_DACFile[uDACChannel].GetSweep( uSweep, pnData, uMaxLength );
}

//===============================================================================================
// FUNCTION: SetErrorCallback
// PURPOSE:  Sets a function to be called if a file I/O error occurs while writing to disk.
//
BOOL CFileDescriptor::SetErrorCallback(ABFCallback fnCallback, void *pvThisPointer)
{
   //MEMBERASSERT();
   m_Notify.RegisterCallback(fnCallback, pvThisPointer);
   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: SetOverlappedFlag
// PURPOSE:  Sets the state of the overlapped flag.
//
void CFileDescriptor::SetOverlappedFlag(BOOL bOverlapped)
{
   //MEMBERASSERT();
   m_bHasOverlappedData = bOverlapped;
}

//===============================================================================================
// FUNCTION: GetOverlappedFlag
// PURPOSE:  Returns the current state of the overlapped flag.
//
BOOL CFileDescriptor::GetOverlappedFlag() const
{
   //MEMBERASSERT();
   return m_bHasOverlappedData;
}

//===============================================================================================
// FUNCTION: SetLastError
// PURPOSE:  Sets the last error value and always returns FALSE for convenience.
//
BOOL CFileDescriptor::SetLastError(int nError)
{
   //MEMBERASSERT();
   m_nLastError = nError;
   return FALSE;           // convenience.
}


//===============================================================================================
// FUNCTION: SetErrorCallback
// PURPOSE:  Sets a function to be called if a file I/O error occurs while writing to disk.
//
int CFileDescriptor::GetLastError() const
{
   //MEMBERASSERT();
   return m_nLastError;
}


/*
//===============================================================================================
// FUNCTION: PutAnnotation
// PURPOSE:  Adds a new annotation to the cache.
//
BOOL CFileDescriptor::PutAnnotation( LPCSTR pszText )
{
   //MEMBERASSERT();
   
   UINT uTags = m_Annotations.Add( pszText );
   if( uTags )
   {
      return TRUE;
   }

   return SetLastError( ABF_EWRITEANNOTATION );
}

//===============================================================================================
// FUNCTION: GetAnnotationCount
// PURPOSE:  Returns the number of annotations in the file.
//
UINT CFileDescriptor::GetAnnotationCount() const
{
   //MEMBERASSERT();
   
   return m_Annotations.GetNumStrings();
}

//===============================================================================================
// FUNCTION: WriteAnnotations
// PURPOSE:  Write the annotations from the cache to the file.
//
BOOL CFileDescriptor::WriteAnnotations( long *plBlockNum, long *plCount )
{
   MEMBERASSERT();
   
   *plBlockNum = 0;
   *plCount    = GetAnnotationCount();
   if (*plCount==0)
      return TRUE;

   UINT uOffset = 0;

   // Try to write the Annotations out to the file
   // If a write fails the user is notified and given the chance
   // to free up disk space and try again.
   while (!FillToNextBlock( plBlockNum ) || 
          !m_Annotations.Write( GetFileHandle(), uOffset ))
   {
      // Notify the user through the callback function. 
      // If the callback returns TRUE, go 'round again.
      if (!m_Notify.Notify(ABF_EDISKFULL))
         return FALSE;
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadAnnotation
// PURPOSE:  Read a single annotation into the buffer.
// NOTES:    At present, this ignores the index and always returns the next annotation.
//
BOOL CFileDescriptor::ReadAnnotation( UINT uIndex, LPSTR pszText, UINT uBufSize )
{
   //MEMBERASSERT();
   LPSZASSERT( pszText );
   WARRAYASSERT( pszText, uBufSize );
   
   memset( pszText, 0, uBufSize );
   LPCSTR pszAnnotation = m_Annotations.Get( uIndex );
   if( pszAnnotation )
   {
      UINT uLen = strlen( pszAnnotation );
      if( uLen > uBufSize )
         return SetLastError( ABF_OUTOFMEMORY );

      strncpy( pszText, pszAnnotation, uLen );
      return TRUE;
   }

   return SetLastError( ABF_ENOANNOTATIONS );
}

//===============================================================================================
// FUNCTION: ReadAllAnnotations
// PURPOSE:  Read the annotations from the file into the cache.
//
BOOL CFileDescriptor::ReadAllAnnotations( long lBlockNum )
{
   //MEMBERASSERT();
   
   // Seek to the start of the first segment.
   UINT uSeekPos = lBlockNum * ABF_BLOCKSIZE;

   // Read the annotations in to the cache.
   if( !m_Annotations.Read( GetFileHandle(), uSeekPos ) )
      return SetLastError( ABF_ENOANNOTATIONS );

   return TRUE;
}

//===============================================================================================
// FUNCTION: GetMaxAnnotationSize
// PURPOSE:  Return the size of the largest annotation.
//
UINT CFileDescriptor::GetMaxAnnotationSize() const
{
   //MEMBERASSERT();
   
   return m_Annotations.GetMaxSize() + 1;
}
*/
