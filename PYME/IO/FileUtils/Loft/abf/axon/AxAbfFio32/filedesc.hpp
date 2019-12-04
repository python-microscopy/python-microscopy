//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  FILEDESC.HPP
// PURPOSE: Contains the class definition for the CFileDescriptor class.
// 

#ifndef INC_FILEDESC_HPP
#define INC_FILEDESC_HPP
/*
#include "./../Common/BufferedArray.hpp"  // Virtual item array objects
*/
#include "abffiles.h"               // ABF file I/O API and error codes.
#include "./../Common/FileIO.hpp"     // Low-level file I/O services
#include "csynch.hpp"               // Virtual synch array object
/*
//#include "DACFile.hpp"              // Virtual DACFile array object
//#include "voicetag.hpp"             // Array of voice tag descriptors.
#include "notify.hpp"               // CABFNotify class -- wraps ABFCallback function.
#include "SimpleStringCache.hpp"    // Virtual annotations object
*/
#define FI_PARAMFILE  0x0001
#define FI_READONLY   0x0002
#define FI_WRITEONLY  0x0004

class CFileDescriptor
{
private:    // Member variables.
   CFileIO        m_File;               // The low level File object.
   CSynch         m_VSynch;             // The virtual synch array
   
   enum { CACHE_SIZE=10 };
//   CBufferedArray m_Tags;               // The tag writing object.
//   CBufferedArray m_Deltas;             // The delta writing object.
//   CVoiceTagList  m_VoiceTagList;       // List of voice tags waiting to be saved in ABF file.
//   CDACFile       m_DACFile[ABF_WAVEFORMCOUNT]; // DAC file sweeps.
   
   UINT           m_uFlags;             // Various flags governing the file opened.
//   CABFNotify     m_Notify;             // Client notification object.
   int            m_nLastError;         // Error number for last error.

   UINT           m_uAcquiredEpisodes;  // The number of episodes written to this file.
   UINT           m_uAcquiredSamples;   // The total number of samples written to this file.
   void          *m_pvReadBuffer;       // Buffer to be used for de-multiplexing data
   UINT           m_uCachedEpisode;     // Episode number for the episode cached in the read buffer.
   UINT           m_uCachedEpisodeSize; // Size of the cached episode.
   UINT           m_uLastEpiSize;       // The size of the last episode for continuous files
   BOOL           m_bHasOverlappedData; // TRUE if the file contains overlapped data.
   char           m_szFileName[_MAX_PATH];
   
//   CSimpleStringCache   m_Annotations;        // The annotations writing object.
   
private:
   CFileDescriptor(const CFileDescriptor &FI);
   const CFileDescriptor &operator=(const CFileDescriptor &FI);
   
public:   
   CFileDescriptor();
   ~CFileDescriptor();
   
   void  SetFlag(UINT uFlag);
   BOOL  TestFlag(UINT uFlag);
   
   BOOL  IsOK();
   BOOL  Open(const char *szFileName, BOOL bReadOnly);
/*   BOOL  Reopen(BOOL bReadOnly);
   
   BOOL  FillToNextBlock( long *plBlockNum );
   BOOL  Write(const void *pvBuffer, UINT uSizeInBytes);
*/   BOOL  Read(void *pvBuffer, UINT uSizeInBytes);
   BOOL  Seek(LONGLONG llOffset, UINT uFlag, LONGLONG *pllOffset=NULL);
/*   BOOL  SetEndOfFile();

*/   LONGLONG GetFileSize();
/*   LPCSTR GetFileName() const;

*/   BOOL  CheckEpisodeNumber(UINT uEpisode);
   void  SetAcquiredEpisodes(UINT uEpisodes);
   UINT  GetAcquiredEpisodes() const; 
   void  SetAcquiredSamples(UINT uSamples);  
/*   UINT  GetAcquiredSamples() const;  

*/   BOOL  AllocReadBuffer(UINT uBytes);
   void  FreeReadBuffer();
   void *GetReadBuffer();       // Buffer to be used for de-multiplexing data

   void  SetCachedEpisode(UINT uEpisode, UINT uEpisodeSize);
   UINT  GetCachedEpisode() const;
   UINT  GetCachedEpisodeSize() const; // Size of the cached episode.

   void  SetLastEpiSize(UINT uEpiSize);
   UINT  GetLastEpiSize() const;
   
   // Synch array functions.   
   BOOL  PutSynchEntry( UINT uStart, UINT uLength, UINT uOffset=0 );
/*   void  IncreaseEventLength( UINT dwIncrease );
*/   BOOL  GetSynchEntry( UINT uEpisode, Synch *pSynch );
   UINT  EpisodeStart( UINT uEpisode);
  /*   void  SetEpisodeStart(UINT uEpisode, UINT uSynchTime);
*/   UINT  EpisodeLength( UINT uEpisode);
/*   UINT  FileOffset( UINT uEpisode);
   BOOL  WriteSynchArray( long *plBlockNum, long *plCount, UINT uSampleSize );
*/   UINT  GetSynchCount() const;
   void  SetSynchMode(CSynch::eMODE eMode);
/*   CSynch *GetSynchObject();
*/   void  ChangeSynchArray(CSynch *pNewSynch);
/*   
   // Tag array functions.
   BOOL  PutTag( const ABFTag *pTag );
   UINT  GetTagCount() const;
   BOOL  WriteTags( long *plBlockNum, long *plCount );
   BOOL  UpdateTag(UINT uTag, const ABFTag *pTag);
   BOOL  ReadTags( UINT uFirstTag, ABFTag *pTagArray, UINT uNumTags);
   
   // Voice tag functions.
   BOOL  SaveVoiceTag( LPCSTR pszFileName, long lDataOffset, ABFVoiceTagInfo *pVTI);
   BOOL  WriteVoiceTags( long *plBlockNum, long *plCount );
   BOOL  GetVoiceTag( UINT uTag, LPCSTR pszFileName, long lDataOffset, 
                      ABFVoiceTagInfo *pVTI, long lVoiceTagPtr);

   // Delta functions.
   BOOL  PutDelta(const ABFDelta *pDelta);
   UINT  GetDeltaCount() const;
   BOOL  WriteDeltas( long *plBlockNum, long *plCount );
   BOOL  ReadDeltas(UINT uFirstDelta, ABFDelta *pDeltaArray, UINT uNumDeltas);

   // DAC file functions.
   BOOL  PutDACFileSweep( UINT uDACChannel, UINT uSweep, const DAC_VALUE *pnData, UINT uLength );

   BOOL  GetDACFileSweep(UINT uDACChannel, UINT uSweep, DAC_VALUE *pnData, UINT uMaxLength);
   UINT  GetDACFileSweepCount( UINT uDACChannel ) const;
   BOOL  WriteDACFileSweeps( UINT uDACChannel, long *plBlockNum, long *plCount );

   // Annotations functions.
   BOOL  PutAnnotation( LPCSTR pszText );
   UINT  GetAnnotationCount() const;
   BOOL  WriteAnnotations( long *plBlockNum, long *plCount );
   BOOL  ReadAnnotation( UINT uIndex, LPSTR pszText, UINT uBufSize );
   BOOL  ReadAllAnnotations( long lBlockNum );
   UINT  GetMaxAnnotationSize() const;
*/
   FILEHANDLE GetFileHandle();   
/*   BOOL  SetErrorCallback(ABFCallback fnCallback, void *pvThisPointer);

*/   void  SetOverlappedFlag(BOOL bOverlapped);
   BOOL  GetOverlappedFlag() const;
   
   BOOL  SetLastError(int nError);
   int   GetLastError() const;

};


//===============================================================================================
// FUNCTION: GetFileHandle
// PURPOSE:  Returns the file handle opened in the object.
//
inline FILEHANDLE CFileDescriptor::GetFileHandle()
{
//   MEMBERASSERT();
   return m_File.GetFileHandle();
}


//===============================================================================================
// FUNCTION: Seek
// PURPOSE:  Change the current position of the file pointer.
//
inline BOOL CFileDescriptor::Seek(LONGLONG llOffset, UINT uFlag, LONGLONG *pllNewOffset)
{
//   MEMBERASSERT();
   return m_File.Seek(llOffset, uFlag, pllNewOffset);
}


//===============================================================================================
// FUNCTION: GetFileSize
// PURPOSE:  Return the length of the file in bytes.
//
inline LONGLONG CFileDescriptor::GetFileSize()
{
//   MEMBERASSERT();
   return m_File.GetFileSize();
}


//===============================================================================================
// PROCEDURE: Read
// PURPOSE:   Reads a block and returnd FALSE on ERROR.
//
inline BOOL CFileDescriptor::Read(LPVOID lpBuf, UINT uBytesToRead)
{
//   MEMBERASSERT();
   return m_File.Read(lpBuf, uBytesToRead) ? TRUE : SetLastError(ABF_EREADDATA);
}

//===============================================================================================
// FUNCTION: IsOK
// PURPOSE:  Checks to see whether the object was created OK.
//
inline BOOL CFileDescriptor::IsOK()
{
//   MEMBERASSERT();
   return (m_nLastError==0);
}

//===============================================================================================
// FUNCTION: PutSynchEntry
// PURPOSE:  Puts a new entry into the Synch array.
//
inline BOOL CFileDescriptor::PutSynchEntry( UINT uStart, UINT uLength, UINT uOffset )
{
//   MEMBERASSERT();
   return m_VSynch.Put( uStart, uLength, uOffset );
}
/*
//===============================================================================================
// FUNCTION: IncreaseEventLength
// PURPOSE:  Increases the length of the last entry in the synch array.
//
inline void CFileDescriptor::IncreaseEventLength( UINT uIncrease )
{
   MEMBERASSERT();
   m_VSynch.IncreaseLastLength( uIncrease );
}
*/   
//===============================================================================================
// FUNCTION: GetSynchEntry
// PURPOSE:  Gets the Synch information for a particular episode.
//
inline BOOL CFileDescriptor::GetSynchEntry( UINT uEpisode, Synch *pSynch )
{
//   MEMBERASSERT();
//   ASSERT(uEpisode > 0);
   return m_VSynch.Get( uEpisode-1, pSynch, 1 );
}

//===============================================================================================
// FUNCTION: GetReadBuffer
// PURPOSE:  returns a pointer to the read buffer.
//
inline void *CFileDescriptor::GetReadBuffer()
{
//   MEMBERASSERT();
   return m_pvReadBuffer;
}

#endif   // INC_FILEDESC_HPP
