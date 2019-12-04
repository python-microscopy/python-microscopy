//***********************************************************************************************
//
//    Copyright (c) 1993-2000 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  ABFFILES.CPP
// PURPOSE: Contains the low level multi data file I/O package for ABF data files.
// 
// An ANSI C++ compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL ABFFILES.C)


#include "../Common/wincpp.hpp"
#include "abffiles.h"

#include "abfutil.h"                // Large memory allocation/free
/*
#include "StringResource.h"         // Access to string resources.
#include "oldheadr.h"               // old header conversion prototypes
*/
#include "csynch.hpp"               // Virtual synch array object
#include "filedesc.hpp"             // File descriptors for ABF files.
#include "./../Common/ArrayPtr.hpp"   // Smart array pointer template class.
#include "./../Common/FileReadCache.hpp"
/*
#include "./../AxoUtils32/AxoUtils32.h"     // for AXU_* functions
#include "./../Common/crc.h"

#ifndef __UNIX__
	#include <objbase.h>                 // UuidCreate
#endif

//
// Set the maximum number of files that can be open simultaneously.
// This can be overridden from the compiler command line.
//
*/
#ifndef ABF_MAXFILES
   #define ABF_MAXFILES 64
#endif

#define ABF_DEFAULTCHUNKSIZE  8192     // Default chunk size for reading gap-free amd var-len files.

#if defined(__UNIX__) || defined(__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

/*
// Set USE_DACFILE_FIX to 1 to use the fix (incomplete) for DAC File channels.
#define USE_DACFILE_FIX 0

//------------------------------------ Shared Variables -----------------------------------------
*/
static CFileDescriptor *g_FileData[ABF_MAXFILES];

HINSTANCE g_hInstance = NULL;

//===============================================================================================

static BOOL ReadEDVarLenSynch(CFileDescriptor *pFI, ABFFileHeader *pFH, 
                              DWORD *pdwMaxEpi, UINT *puMaxSamples, int *pnError);

static BOOL ReadEDFixLenSynch(CFileDescriptor *pFI, const ABFFileHeader *pFH, DWORD *pdwMaxEpi, 
                              BOOL bAllowOverlap, int *pnError);

static BOOL ReadOldSynchArray(CFileDescriptor *pFI, ABFFileHeader *pFH, DWORD *pdwMaxEpi, 
                              int *pnError);

//===============================================================================================
// Macros and functions to deal with returning error return codes through a pointer if given.

#define ERRORRETURN(p, e)  return ErrorReturn(p, e);
static BOOL ErrorReturn(int *pnError, int nErrorNum)
{
   if (pnError)
      *pnError = nErrorNum;
   return FALSE;
}

//===============================================================================================
// FUNCTION: GetNewFileDescriptor
// PURPOSE:  Allocate a new file descriptor and return it.
//
static BOOL GetNewFileDescriptor(CFileDescriptor **ppFI, int *pnFile, int *pnError)
{
//   WPTRASSERT(ppFI);
//   WPTRASSERT(pnFile);
   int nFile;
   
   // Find an empty slot.
   for (nFile=0; nFile < ABF_MAXFILES; nFile++)
      if (g_FileData[nFile] == NULL)
         break;
   
   // Return an error if no space left.   
   if (nFile == ABF_MAXFILES)
      ERRORRETURN(pnError, ABF_TOOMANYFILESOPEN);

   // Allocate a new descriptor.
   CFileDescriptor *pFI = new CFileDescriptor;
   if (pFI == NULL)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);
      
   if (!pFI->IsOK())
   {
      delete pFI;
      ERRORRETURN(pnError, ABF_BADTEMPFILE);
   }
      
   *ppFI = g_FileData[nFile] = pFI;
   *pnFile = nFile;
   return TRUE;
}

//-----------------------------------------------------------------------------------------------
// FUNCTION: GetFileDescriptor
// PURPOSE:  Retreive an existing file descriptor.
//
static BOOL GetFileDescriptor(CFileDescriptor **ppFI, int nFile, int *pnError)
{
//   WPTRASSERT(ppFI);

   // Check that index is within range.
   if ((nFile < 0) || (nFile >= ABF_MAXFILES))
      ERRORRETURN(pnError, ABF_EBADFILEINDEX);

   // Get a pointer to the descriptor.
   CFileDescriptor *pFI = g_FileData[nFile];
   if (pFI == NULL)
      ERRORRETURN(pnError, ABF_EBADFILEINDEX);

   // Return the descriptor.
   *ppFI = pFI;
   return TRUE;
}

//-----------------------------------------------------------------------------------------------
// FUNCTION: ReleaseFileDescriptor
// PURPOSE:  Release an existing file descriptor.
//
static void ReleaseFileDescriptor(int nFile)
{
   delete g_FileData[nFile];
   g_FileData[nFile] = NULL;
}

//===============================================================================================
// FUNCTION: SampleSize
// PURPOSE:  Get the sample size used in the data described by the header.
//
static UINT SampleSize(const ABFFileHeader *pFH)
{
//   ABFH_ASSERT(pFH);
   return (pFH->nDataFormat != ABF_INTEGERDATA) ? sizeof(float) : sizeof(short);
}

//===============================================================================================
// FUNCTION: GetDataOffset
// PURPOSE:  Get the file offset to the data allowing for "ignored" points from old AxoLab files.
//
static long GetDataOffset(const ABFFileHeader *pFH)
{
//   ABFH_ASSERT(pFH);
   long lDataOffset = pFH->lDataSectionPtr * ABF_BLOCKSIZE;
   
   // Adjust the data pointer for any garbage data words at the start of
   // the data portion of the file. (Created by AxoLab in continuous
   // files only)
   if (pFH->nOperationMode == ABF_GAPFREEFILE)
      lDataOffset += pFH->nNumPointsIgnored * SampleSize(pFH);
      
   return lDataOffset;
}
/*
//==============================================================================================
// FUNCTION: CalculateCRC
// PURPOSE:  Return checksum Cyclic Redundancy Code CRC.
//
unsigned long CalculateCRC( CFileDescriptor *pFI )
{
   WPTRASSERT( pFI);

   LONGLONG llReadPointer         = 0L;
   BOOL bReadOk                   = FALSE;
   char acBuffer[ ABF_BLOCKSIZE ] = {0};
   CRC crc(CRC::CRC_32);

   // Get the total length of the file.
   LONGLONG llFileLength = pFI->GetFileSize();
   ASSERT(llFileLength >= sizeof(ABFFileHeader));

   VERIFY(pFI->Seek( 0L, FILE_BEGIN));
   
   while( llReadPointer < llFileLength )
   {
      // Read a file block into the buffer.
      bReadOk = pFI->Read( acBuffer, ABF_BLOCKSIZE );
      ASSERT( bReadOk );

      // Update the CRC of the buffer
      crc.Update(acBuffer, ABF_BLOCKSIZE );
      llReadPointer += ABF_BLOCKSIZE;
   }

//#ifdef _DEBUG     
//   TRACE1("Calculate CRC Value %X\n", crc.Value() ); 
//#endif

   // Set pointer at the beggining.
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));

   return crc.Value();
}

//==============================================================================================
// FUNCTION: ValidateFileCRC
// PURPOSE:  
//
static BOOL ValidateFileCRC( CFileDescriptor *pFI, ABFFileHeader *pFH, int nSizeOfHeader )
{
   WPTRASSERT( pFI );
   WPTRASSERT( pFH );

   // Validate CRC for files that support it.
   // The versions of ABF 1.82 and higher support CRC checksum.
   if( pFH->fFileVersionNumber < ABF_V182 )
      return TRUE; // Valid and no checking.

   unsigned long ulExpectedCRC    = 0L;
   LONGLONG llReadPointer         = 0L;
   BOOL bReadOk                   = FALSE;
   char acBuffer[ ABF_BLOCKSIZE ] = {0};
   CRC crc(CRC::CRC_32);

   // Keep expected CRC value from header.
   ulExpectedCRC = pFH->ulFileCRC;
   
   // Zero the lFileCRC. The CRC is generated with this field as zero.
   pFH->ulFileCRC = 0;

   // Get the total length of the file.
   LONGLONG llFileLength = pFI->GetFileSize();
   ASSERT(llFileLength >= nSizeOfHeader );
   
   crc.Update( pFH, nSizeOfHeader );

   llReadPointer = nSizeOfHeader;

   VERIFY(pFI->Seek( llReadPointer, FILE_BEGIN));
   
   while( llReadPointer < llFileLength )
   {
      // Read a file block into the buffer.
      bReadOk = pFI->Read( acBuffer, ABF_BLOCKSIZE );
      ASSERT( bReadOk );
      
      // Update the CRC of the buffer
      crc.Update(acBuffer, ABF_BLOCKSIZE );
      llReadPointer += ABF_BLOCKSIZE;
   }

#ifdef _DEBUG   
   TRACE1("Validate CRC Value %X\n", crc.Value() ); 
#endif
   // Set pointer at the beggining.
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));

   unsigned long ulFileCRC = crc.Value();

   // Compare expected CRC with file CRC.
   if ( ulFileCRC != ulExpectedCRC )
   {
#ifdef _DEBUG   
      TRACE( "File CRC Validation Failed\n" );
#endif
      return FALSE;
   }

#ifdef _DEBUG   
   TRACE( "File CRC Validation OK\n" );
#endif
   return TRUE;
} 
*/
//===============================================================================================
// FUNCTION:   ABF_Initialize()
// PARAMETERS:
//   hInstance - Instance handle from which resources will be taken.
// RETURNS:
//   BOOL       - TRUE = Initialization was successful.
//
// PURPOSE:    This function should be called before any of the other API functions.
// NOTES:      This function is not exported as it is called from the DLL startup code. If the
//             API is bound into an executable rather than a DLL it will need to be called
//             explicitly.
//
BOOL ABF_Initialize()
{
   // Protect against multiple calls.
/*   if (g_hInstance != NULL)
      return TRUE;

   // Save the DLL instance handle.
   g_hInstance = hDLL;
*/
   for (int i=0; i<ABF_MAXFILES; i++)
      g_FileData[i] = NULL;
      
#if (ABF_MAXFILES > 15)      
//   UINT uAvailableFiles = SetHandleCount(ABF_MAXFILES);  uAvailableFiles = uAvailableFiles;
#endif
   return TRUE;
}
/*

//===============================================================================================
// FUNCTION: ABF_Cleanup
// PURPOSE:  Cleanup function, only applicable to DOS & Windows programs, not DLLs.
// NOTES:    This function is not exported as it is called from the DLL startup code. If the
//           API is bound into an executable rather than a DLL it will need to be called
//           explicitly.
//
void ABF_Cleanup(void)
{
   for (int i=0; i<ABF_MAXFILES; i++)
   {
      if (g_FileData[i])
      {
         WPTRASSERT(g_FileData[i]);
         TRACE1("ABF file '%s' was not closed.\n", g_FileData[i]->GetFileName());
         ABF_Close(i, NULL);
      }
   }
}
*/
//===============================================================================================
// FUNCTION: ABF_ReadOpen
// PURPOSE:  This routine opens an existing data file for reading. It reads the acquisition 
//           parameters, ADC/DAC unit strings and comment string from the file header.
// INPUT:
//   szFileName     the name of the data file that will be opened
//   fFlags         Flag for whether the file is a parameter file
//   puMaxSamples   points to the requested size of data blocks to be returned.
//                  This is only used in the case of GAPFREE and EVENT-DETECTED-
//                  VARIABLE-LENGTH acquisitions. Otherwise the size of the
//                  Episode is used. 80x86 limitations require this to be 
//                  less than or equal to 64k.
//   pdwMaxEpi      The maximum number of episodes to be read.
// OUTPUT:
//   pFH            the acquisition parameters that were read from the data file
//   phFile         pointer to the ABF file number of this file (NOT the DOS handle)
//   puMaxSamples   the maximum number of samples that can be read contiguously
//                  from the data file.
//   pdwMaxEpi      the number of episodes of puMaxSamples points that exist
//                  in the data file.
// 
BOOL WINAPI ABF_ReadOpen(LPCSTR szFileName, int *phFile, UINT fFlags, ABFFileHeader *pFH, 
                         UINT *puMaxSamples, DWORD *pdwMaxEpi, int *pnError)
{
// CSH  LPSZASSERT(szFileName);
// CSH  WPTRASSERT(phFile);
// CSH  ABFH_WASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   int nError = 0;
   CFileDescriptor *pFI = NULL;
   UINT uDAC = 0;

   // Get a new file descriptor if available.
   if (!GetNewFileDescriptor(&pFI, phFile, pnError))
      return FALSE;
      
   // Now open the file for reading.
   if (!pFI->Open(szFileName, TRUE))
   {
      nError = pFI->GetLastError();
      goto RCloseAndAbort;
   }

   // Read the data file parameters.
   if (!ABFH_ParamReader(pFI->GetFileHandle(), &NewFH, &nError))
   {
      nError = (nError == ABFH_EUNKNOWNFILETYPE) ? ABF_EUNKNOWNFILETYPE : ABF_EBADPARAMETERS;
      goto RCloseAndAbort;
   }

   if (NewFH.lFileSignature == ABF_REVERSESIGNATURE)
   {
      nError = ABF_EBADPARAMETERS;
      goto RCloseAndAbort;
   }

   // if we are reading a parameter file, we are done.
   if (fFlags & ABF_PARAMFILE)
   {
      // If it is an old (pre-ABF file), update file version and file type.
      if( (NewFH.nFileType == ABF_CLAMPEX) ||
          (NewFH.nFileType == ABF_FETCHEX) )
      {
          NewFH.nFileType          = ABF_ABFFILE;
          NewFH.fFileVersionNumber = ABF_CURRENTVERSION;
      }

      pFI->SetFlag(FI_PARAMFILE);

      // Restore the original header.
      ABFH_DemoteHeader( pFH, &NewFH );
   
      return TRUE;
   }

   // Check for valid parameters.
//   WPTRASSERT(puMaxSamples);
//   WPTRASSERT(pdwMaxEpi);

   // Check that the data file actually contains data.
   if ((NewFH.lActualAcqLength <= 0) || (NewFH.nADCNumChannels <= 0))
   {
      nError = ABF_EBADPARAMETERS;
      goto RCloseAndAbort;
   }

   // Disable stimulus file output if data file does not include any 
   // stimulus file sweeps. This is to prevent problems later when 
   // looking for a non-existent DACFile section.
   for( uDAC=0; uDAC<ABF_WAVEFORMCOUNT; uDAC++ )
   {
      if( (NewFH.lDACFileNumEpisodes[uDAC] <= 0) || (NewFH.lDACFilePtr[uDAC] <= 0) )
      {
	      NewFH.lDACFileNumEpisodes[uDAC] = 0;
	      NewFH.lDACFilePtr[uDAC]         = 0;
	      if( NewFH.nWaveformSource[uDAC] == ABF_DACFILEWAVEFORM )
		      NewFH.nWaveformSource[uDAC] = ABF_WAVEFORMDISABLED;
      }
   }

   if (NewFH.nOperationMode == ABF_GAPFREEFILE)
   {
      // If the gap-free file has a synch array, read and expand it.
      if (!ReadEDVarLenSynch(pFI, &NewFH, pdwMaxEpi, puMaxSamples, &nError))
         goto RCloseAndAbort;
   }
   else if (NewFH.nFileType != ABF_ABFFILE)
   {
      // Read Synch array for discontinuous FETCHEX/AXOTAPE files.
      if (!ReadOldSynchArray(pFI, &NewFH, pdwMaxEpi, &nError))
         goto RCloseAndAbort;
   }
   else if (NewFH.nOperationMode == ABF_VARLENEVENTS)
   {
      // Read the synch array and split it into smaller chunks if necessary.
      if (!ReadEDVarLenSynch(pFI, &NewFH, pdwMaxEpi, puMaxSamples, &nError))
         goto RCloseAndAbort;
   }
   else    // must be event detected fixed length data, or Waveform file.
   {
      BOOL bAllowOverlap = ((fFlags & ABF_ALLOWOVERLAP) != 0);
      if (!ReadEDFixLenSynch (pFI, &NewFH, pdwMaxEpi, bAllowOverlap, &nError))
         goto RCloseAndAbort;
   }

   // Set the return value for the read chunk size.
   *puMaxSamples = (UINT)(NewFH.lNumSamplesPerEpisode / NewFH.nADCNumChannels);

   // Set header variable for the number of episodes in the file.
   NewFH.lActualEpisodes = *pdwMaxEpi;
   pFI->SetAcquiredEpisodes(*pdwMaxEpi);
   pFI->SetAcquiredSamples(NewFH.lActualAcqLength);

   // Seek to start of Data section
//   VERIFY(pFI->Seek(GetDataOffset(&NewFH), FILE_BEGIN));

   // Restore the original header.
   ABFH_DemoteHeader( pFH, &NewFH );
   
   return TRUE;

RCloseAndAbort:
   ASSERT(nError!=0);
   ReleaseFileDescriptor(*phFile);
   phFile = ABF_INVALID_HANDLE;
   ERRORRETURN(pnError, nError);
}
/*
//===============================================================================================
// FUNCTION: ABF_IsABFFile
// PURPOSE:  This routine opens a file and determines if it is an ABF file or not.
// RETURNS:  TRUE if the file is an ABF file. The type of ABF file is returned in
//           *pnDataFormat:  ABF_INTEGERDATA or ABF_FLOATDATA.
//
BOOL WINAPI ABF_IsABFFile(const char *szFileName, int *pnDataFormat, int *pnError)
{
   LPSZASSERT(szFileName);
   int nError = 0;
   
   // Now open the file for reading.
   HANDLE hHandle = CreateFile(szFileName, GENERIC_READ, FILE_SHARE_READ, NULL, 
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
   if (hHandle == INVALID_HANDLE_VALUE)
   {
      if (GetLastError()==ERROR_TOO_MANY_OPEN_FILES)
         ERRORRETURN(pnError, ABF_NODOSFILEHANDLES);
      ERRORRETURN(pnError, ABF_EOPENFILE);
   }

   // Read the data file parameters.
   ABFFileHeader FH;
   if (!ABFH_ParamReader(hHandle, &FH, &nError))
   {
      if (nError == ABFH_EUNKNOWNFILETYPE)
         nError = ABF_EUNKNOWNFILETYPE;
      else
         nError = ABF_EBADPARAMETERS;
   }
   CloseHandle(hHandle);
   if (nError)
      ERRORRETURN(pnError, nError);

   if (pnDataFormat)
      *pnDataFormat = FH.nDataFormat;
      
   return TRUE;
}


//===============================================================================================
// FUNCTION: ABF_WriteOpen
// PURPOSE:  This routine opens an existing data file for writing.
//           It writes all the acquisition parameters to the file header.
// INPUT:
//   szFileName     the name of the data file that will be opened
//   phFile         pointer to the ABF file number of this file (NOT the DOS handle)
//   fFlags         Flag for whether the file is a parameter file
//   pFH            the acquisition parameters to be written to the data file
// 
// OUTPUT:
//   NONE.
// 
BOOL WINAPI ABF_WriteOpen(LPCSTR szFileName, int *phFile, UINT fFlags, ABFFileHeader *pFH, int *pnError)
{
   LPSZASSERT(szFileName);
   WPTRASSERT(phFile);
   ABFH_WASSERT(pFH);
   
   // Get a new file descriptor if available.
   CFileDescriptor *pFI = NULL;
   if (!GetNewFileDescriptor(&pFI, phFile, pnError))
      return FALSE;
      
   // Now create and open the file for writing
   if (!pFI->Open(szFileName, FALSE))
   {
      // An error has occurred, cleanup and return the error.   
      int nError = pFI->GetLastError();
      ReleaseFileDescriptor(*phFile);
      *phFile = ABF_INVALID_HANDLE;
      ERRORRETURN(pnError, nError);
   }

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   NewFH.lDataSectionPtr      = sizeof(ABFFileHeader) / ABF_BLOCKSIZE;
   NewFH.lScopeConfigPtr      = 0;
   NewFH.lStatisticsConfigPtr = 0;
   NewFH.lNumScopes           = 0;
   NewFH.lActualAcqLength     = 0;
   NewFH.lActualEpisodes      = 0;
   NewFH.nNumPointsIgnored    = 0;
   NewFH.lTagSectionPtr       = 0;
   NewFH.lNumTagEntries       = 0;
   NewFH._lDACFilePtr         = 0;
   NewFH._lDACFileNumEpisodes = 0;
   NewFH.lDeltaArrayPtr       = 0;
   NewFH.lNumDeltas           = 0;
   NewFH.lSynchArrayPtr       = 0;
   NewFH.lSynchArraySize      = 0;
   NewFH.lVoiceTagPtr         = 0;
   NewFH.lVoiceTagEntries     = 0;
   NewFH.lAnnotationSectionPtr= 0;
   NewFH.lNumAnnotations      = 0;
   NewFH.ulFileCRC             = 0;

   for( UINT i=0; i<ABF_WAVEFORMCOUNT; i++ )
   {
      NewFH.lDACFilePtr[i]         = 0;
      NewFH.lDACFileNumEpisodes[i] = 0;
   }

   if (fFlags & ABF_PARAMFILE)
      pFI->SetFlag(FI_PARAMFILE);
      
   // Create a GUID for this file.
	NewFH.FileGUID = GUID_NULL;
	::CoCreateGuid(&NewFH.FileGUID);

   // Write the data file parameters, returning if successful.
   if (!ABFH_ParamWriter(pFI->GetFileHandle(), &NewFH, NULL))
   {      
      // An error has occurred, cleanup and return the error.   
      ReleaseFileDescriptor(*phFile);
      *phFile = ABF_INVALID_HANDLE;
      remove(szFileName);
      ERRORRETURN(pnError, ABF_EDISKFULL);
   }

   // Calculate CRC in current file descriptor.
   // lFileCRC needs to be zero during the calculation.
   NewFH.ulFileCRC = CalculateCRC( pFI );

   // Update the header in the file descriptor with CRC in place.
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));
   if (!ABFH_ParamWriter(pFI->GetFileHandle(), &NewFH, NULL))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   // Restore the original header.
   ABFH_DemoteHeader( pFH, &NewFH );
   
   return TRUE;
}

//===============================================================================================
// FUNCTION: WriteSynchArray
// PURPOSE:  This routine writes the Synch array out to disk
//
static BOOL WriteSynchArray(CFileDescriptor *pFI, ABFFileHeader *pFH, int *pnError)
{
   WPTRASSERT(pFI);
   ABFH_WASSERT(pFH);

   // Return if no write is required.
   if (pFI->TestFlag(FI_PARAMFILE))
      return TRUE;

   // Transfer the synch array to the ABF file, checking to see that the synch array
   // only refers to data that was actually saved.
   if (!pFI->WriteSynchArray( &pFH->lSynchArrayPtr, &pFH->lSynchArraySize, SampleSize(pFH) ))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   pFH->lActualEpisodes  = pFH->lSynchArraySize;
   pFH->lActualAcqLength = pFI->GetAcquiredSamples();
   return TRUE;
}

//===============================================================================================
// FUNCTION: WriteTags
// PURPOSE:  This routine writes the accumulated tags out to disk
//
static BOOL WriteTags(CFileDescriptor *pFI, ABFFileHeader *pFH, int *pnError)
{
   WPTRASSERT(pFI);
   ABFH_WASSERT(pFH);
   if (!pFI->WriteTags( &pFH->lTagSectionPtr, &pFH->lNumTagEntries ))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   if (!pFI->WriteVoiceTags( &pFH->lVoiceTagPtr, &pFH->lVoiceTagEntries ))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: ABF_HasData
// PURPOSE:  This routine returns TRUE if data has been written to the file since opening it.
//
BOOL WINAPI ABF_HasData(int nFile, const ABFFileHeader *pFH)
{
//   ABFH_ASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, NULL))
      return FALSE;

   // Take a copy of the passed in header to ensure it is 5k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   if (NewFH.lDataSectionPtr==0)
      return FALSE;

   // Assume that only data has been written to the file at this point.
   ASSERT(NewFH.lSynchArrayPtr==0);
   ASSERT(NewFH.lTagSectionPtr==0);
   ASSERT(NewFH.lVoiceTagPtr==0);
   ASSERT(NewFH.lDeltaArrayPtr==0);
   ASSERT(NewFH.lAnnotationSectionPtr==0);
   ASSERT(NewFH.lDACFilePtr[0]==0);
   ASSERT(NewFH.lDACFilePtr[1]==0);

   return (pFI->GetFileSize() > NewFH.lDataSectionPtr * ABF_BLOCKSIZE);
}
/*   
//===============================================================================================
// FUNCTION: ABF_UpdateHeader
// PURPOSE:  This routine should always be called before closing a file opened with
//           ABF_WriteOpen. It updates the file header and writes the synch array out
//           to disk if required.
//
BOOL WINAPI ABF_UpdateHeader(int nFile, ABFFileHeader *pFH, int *pnError)
{
   ABFH_WASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (pFI->TestFlag(FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);
      
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   // Assume that only data has been written to the file at this point.
   //GR 12-Jan-2001:  These used to be asserts to assert that these values 
   //were all equal to 0.  This was causing assertion failures when 
   //attempting to copy part or all of an analysis window file as an 
   //atf file for the second time.
   //The logic is that if these variables are no longer 0, then we have 
   //already written the file header and don't need to write it again.
   //And what if the header settings have changed since last time we 
   //wrote them?  Then the temporary file should have been re-written 
   //and these variables will be 0 again.  I think.
   if( (NewFH.lSynchArrayPtr!=0) ||
       (NewFH.lTagSectionPtr!=0) ||
       (NewFH.lVoiceTagPtr!=0) ||
       (NewFH.lAnnotationSectionPtr!=0) ||
       (NewFH.lDeltaArrayPtr!=0) )
   {
      return TRUE;
   }

   for( UINT i=0; i<ABF_WAVEFORMCOUNT; i++ )
      ASSERT(NewFH.lDACFilePtr[i] == 0);

   // Get the total length of the file.
   LONGLONG llFileLength = pFI->GetFileSize();
   ASSERT(llFileLength >= sizeof(ABFFileHeader));

   // Calculate the number of data samples in the file.
   UINT uAcquiredSamples = UINT((llFileLength - NewFH.lDataSectionPtr * ABF_BLOCKSIZE)/SampleSize(&NewFH));
   pFI->SetAcquiredSamples(uAcquiredSamples);
   
   if (pFI->GetSynchCount() != 0)
   {   
      // Write the synch array out to disk, storing the size and location
      // of the Synch array in the header in the process.
      if (!WriteSynchArray(pFI, &NewFH, pnError))
      {
         // Truncate the file at the end of the data section.
         VERIFY(pFI->Seek( llFileLength, FILE_END));
         VERIFY(pFI->SetEndOfFile());
         return FALSE;
      }

      // Switch the synch array to read mode to allow access to synch array in Save As (PRC 6/99)
      pFI->SetSynchMode( CSynch::eREADMODE );
   }
   else if (NewFH.nOperationMode == ABF_GAPFREEFILE)
   {
      NewFH.lEpisodesPerRun  = 1;
      NewFH.lRunsPerTrial    = 1;
      NewFH.lActualAcqLength = uAcquiredSamples;
      NewFH.lActualEpisodes  = uAcquiredSamples / NewFH.lNumSamplesPerEpisode;
      // Allow for the last (possibly incomplete) episode.
      if( uAcquiredSamples % NewFH.lNumSamplesPerEpisode )
         NewFH.lActualEpisodes++;
   }
   else if (NewFH.nOperationMode == ABF_WAVEFORMFILE)
   {
      UINT uAcquiredEpisodes = uAcquiredSamples / NewFH.lNumSamplesPerEpisode;
      uAcquiredSamples = uAcquiredEpisodes * NewFH.lNumSamplesPerEpisode;

      pFI->SetAcquiredEpisodes(uAcquiredEpisodes);
      pFI->SetAcquiredSamples(uAcquiredSamples);

      NewFH.lActualEpisodes  = uAcquiredEpisodes;
      NewFH.lActualAcqLength = uAcquiredSamples;
   }
   
   if (pFI->GetTagCount() > 0)
   {
      // Write the tags out to disk, storing the size and location 
      // of the tag block in the header in the process.
      if (!WriteTags(pFI, &NewFH, pnError))
      {
         // Truncate the file at the end of the data section.
         VERIFY(pFI->Seek( llFileLength, FILE_END));
         VERIFY(pFI->SetEndOfFile());
         return FALSE;
      }
   }

   if (pFI->GetDeltaCount() > 0)
   {
      // Write the deltas out to disk, storing the size and location 
      // of the delta array in the header in the process.
      if (!pFI->WriteDeltas( &NewFH.lDeltaArrayPtr, &NewFH.lNumDeltas ))
      {
         // Truncate the file at the end of the data section.
         VERIFY(pFI->Seek( llFileLength, FILE_END));
         VERIFY(pFI->SetEndOfFile());
         ERRORRETURN(pnError, ABF_EDISKFULL);
      }
   }

   if (pFI->GetAnnotationCount() > 0)
   {
      // Write the annotations out to disk, storing the size and location 
      // of the annotations section in the header in the process.
      if(!pFI->WriteAnnotations( &NewFH.lAnnotationSectionPtr, &NewFH.lNumAnnotations ) )
      {
         // Truncate the file at the end of the data section.
         VERIFY(pFI->Seek( llFileLength, FILE_END));
         VERIFY(pFI->SetEndOfFile());
         ERRORRETURN(pnError, ABF_EDISKFULL);
      }
   }

   for(int i=0; i<ABF_WAVEFORMCOUNT; i++ )
   {
      if (pFI->GetDACFileSweepCount(i) > 0)
      {
         // Write the deltas out to disk, storing the size and location 
         // of the delta array in the header in the process.
         if (!pFI->WriteDACFileSweeps( i, &NewFH.lDACFilePtr[i], &NewFH.lDACFileNumEpisodes[i] ))
         {
            // Truncate the file at the end of the data section.
            VERIFY(pFI->Seek( llFileLength, FILE_END));
            VERIFY(pFI->SetEndOfFile());
            ERRORRETURN(pnError, ABF_EDISKFULL);
         }
      }
   }

   // Read back the header image that was written when the file was opened.
   ABFFileHeader OldHeader;
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));
   
   UINT uHeaderSize = ABF_OLDHEADERSIZE;
   if( ABFH_IsNewHeader(&NewFH) )
      uHeaderSize = ABF_HEADERSIZE;

   VERIFY(pFI->Read(&OldHeader, uHeaderSize));

   // Create a copy of the header as it stands now (post acquisition).
   ABFFileHeader NewHeader = NewFH;

   // Copy the original values of items that can be subject to deltas into the new header.
   // This ensures the copy on disk is the original values (with delta info) and the client
   // has the current (updated with deltas) settings.

   // This list should be maintained along with the enumeration for delta types in ABFHEADR.H

   // Delta type: ABF_DELTA_HOLDING0 .. ABF_DELTA_HOLDING3
   for (int i=0; i<ABF_DACCOUNT; i++)
      NewHeader.fDACHoldingLevel[i] = OldHeader.fDACHoldingLevel[i];

   // Delta type: ABF_DELTA_DIGITALOUTS
   NewHeader.nDigitalHolding = OldHeader.nDigitalHolding;

   // Delta type: ABF_DELTA_THRESHOLD
   NewHeader.fTriggerThreshold = OldHeader.fTriggerThreshold;

   // Delta type: ABF_DELTA_PRETRIGGER
   NewHeader.lPreTriggerSamples = OldHeader.lPreTriggerSamples;

   // Delta type: ABF_DELTA_AUTOSAMPLE_GAIN + nAutosampleADCNum
// FIX FIX FIX PRC DEBUG Telegraph changes - check !
   for (int i=0; i<ABF_ADCCOUNT; i++)
      NewHeader.fTelegraphAdditGain[i] = OldHeader.fTelegraphAdditGain[i];

   // NewHeader.lFileCRC needs to be zero for the CRC calculation.
   NewHeader.ulFileCRC = 0L;

   // Update all of header
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));
   if (!ABFH_ParamWriter(pFI->GetFileHandle(), &NewHeader, NULL))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   // Update the current file length. 
   long lCurrentFileSize = (long)pFI->GetFileSize();
   
   // Pad with zeroes to the nearest block boundary. 
   pFI->FillToNextBlock( &lCurrentFileSize );

   // Calculate CRC in current file descriptor. 
   NewHeader.ulFileCRC = CalculateCRC( pFI );

   // Update the header in the file descriptor with CRC in place.
   VERIFY(pFI->Seek( 0L, FILE_BEGIN));
   if (!ABFH_ParamWriter(pFI->GetFileHandle(), &NewHeader, NULL))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   ABFH_DemoteHeader( pFH, &NewFH );

   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: ABF_Close
// PURPOSE:  This routine closes the current data file and cleans up any work buffers that
//           were allocated for processing the data.
// 
BOOL WINAPI ABF_Close(int nFile, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
   {
//      TRACE("ABF_Close failed.\n");
      return FALSE;
   }

   ReleaseFileDescriptor(nFile);
   return TRUE;
}

//===============================================================================================
// FUNCTION: SamplesToSynchCounts
// PURPOSE:  Converts a value in multiplexed samples to Synch Time Units.
//
static UINT SamplesToSynchCounts(const ABFFileHeader *pFH, UINT uSamples)
{
   DWORD dwLengthInSynchUnits = uSamples;
   if( pFH->fSynchTimeUnit != 0.0F ) 
   {
      double dLen = dwLengthInSynchUnits * ABFH_GetFirstSampleInterval(pFH)  * pFH->nADCNumChannels / 1E3;
      dLen = floor( dLen + 0.5 );
      dwLengthInSynchUnits = DWORD( dLen );
   }

   return dwLengthInSynchUnits;
}

//===============================================================================================
// FUNCTION: ExpandSynchEntry
// PURPOSE:  Unpacks a synch entry into one or more chunks no greater than the max chunk size.
//
static void ExpandSynchEntry(const ABFFileHeader *pFH, CSynch &SynchArray, Synch *pItem, UINT uChunkSize, UINT uSampleSize)
{
   UINT uStart      = pItem->dwStart;
   UINT uLength     = pItem->dwLength;
   UINT uFileOffset = pItem->dwFileOffset;

   while (uLength > uChunkSize)
   {
      SynchArray.Put(uStart, uChunkSize, uFileOffset);
      uStart      += SamplesToSynchCounts( pFH, uChunkSize );
      uFileOffset += uChunkSize * uSampleSize;
      uLength     -= uChunkSize;
   }
   SynchArray.Put(uStart, uLength, uFileOffset);
}

//===============================================================================================
// FUNCTION: _SetChunkSize
// PURPOSE:  This routine can be called on files of type ABF_GAPFREEFILE or ABF_VARLENEVENTS to change
//           the size of the data chunks returned by the read routines.
// INPUT:
//   hFile          ABF file number of this file (NOT the DOS handle)
//   pFH            the current acquisition parameters for the data file
//   puMaxSamples   points to the requested size of data blocks to be returned.
//                  This is only used in the case of GAPFREE and EVENT-DETECTED-
//                  VARIABLE-LENGTH acquisitions. Otherwise the size of the
//                  Episode is used. 80x86 limitations require this to be 
//                  less than or equal to 64k.
//   pdwMaxEpi      The maximum number of episodes to be read.
// OUTPUT:
//   pFH            the acquisition parameters that were read from the data file
//   puMaxSamples   the maximum number of samples that can be read contiguously
//                  from the data file.
//   pdwMaxEpi      the number of episodes of puMaxSamples points that exist
//                  in the data file.
// 
static BOOL _SetChunkSize( CFileDescriptor *pFI, ABFFileHeader *pFH, 
                           UINT *puMaxSamples, DWORD *pdwMaxEpi, int *pnError )
{
   // Check for valid parameters.
//   WPTRASSERT(puMaxSamples);
//   WPTRASSERT(pdwMaxEpi);

   // Check that requested chunk size is reasonable.
   // If chunk-size is zero, it is treated as a request for ABF to set a reasonable 
   // chunk-size. An error is returned if chunk-size is given but too small. 
   // If the size given is too big, it is set to the largest possible size.

   UINT uLimSamples = PCLAMP7_MAXSWEEPLEN_PERCHAN;
   UINT uMaxSamples = *puMaxSamples;
   
   // if uMaxSamples == -1, restore the chunk size to the "raw" value (i.e. from disk).
   if ((int)uMaxSamples != -1 )
   {
      if (uMaxSamples == 0)
         uMaxSamples = ABF_DEFAULTCHUNKSIZE / pFH->nADCNumChannels;
      else if (uMaxSamples > uLimSamples)
         uMaxSamples = uLimSamples;
   }

   UINT uAcqLenPerChannel = UINT(pFH->lActualAcqLength / pFH->nADCNumChannels);
   if (uMaxSamples > uAcqLenPerChannel)
      uMaxSamples = uAcqLenPerChannel;

   pFH->lNumSamplesPerEpisode = long(uMaxSamples * pFH->nADCNumChannels);
   
   // Set the return value for the read chunk size.
   *puMaxSamples = (UINT)(pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels);

   // Scan through the synch array building up into full event sizes and then
   // subdividing down into multiples of the chunk size.
   if (pFI->GetSynchCount() <= 0)
   {
      // Only ABF_GAPFREEFILEs and ABF_WAVEFORMFILEs can optionally have synch arrays.
      ASSERT((pFH->nOperationMode == ABF_GAPFREEFILE) ||
             (pFH->nOperationMode == ABF_WAVEFORMFILE));

      // Gap-free files only have synch arrays if they have been paused during recording
      // If there is no synch array, work out how many chunks we have etc.
      //
      // Gapfree files without synch arrays need to know the size of the last episode 
      // (this can be less than the episode size for gap-free data)

      UINT uMaxEpi      = uAcqLenPerChannel / uMaxSamples;
      UINT uLastEpiSize = uAcqLenPerChannel % uMaxSamples;

      if (uLastEpiSize > 0)
      {
         uMaxEpi++;
         ASSERT(pFH->nOperationMode == ABF_GAPFREEFILE);
      }
      else
         uLastEpiSize = uMaxSamples;

      *pdwMaxEpi = uMaxEpi;
      pFI->SetLastEpiSize(uLastEpiSize * pFH->nADCNumChannels);
   }
   else if ((pFH->nOperationMode == ABF_GAPFREEFILE) || (pFH->nOperationMode == ABF_VARLENEVENTS))
   {
      // Create a new synch array that we can build from the old one.
      CSynch NewSynchArray;
      if (!NewSynchArray.OpenFile())
         ERRORRETURN(pnError, ABF_BADTEMPFILE);

      // Cache some useful constants
      const UINT uSampleSize   = SampleSize(pFH);      
      const UINT uSynchCount   = pFI->GetSynchCount();
      const UINT uMaxChunkSize = *puMaxSamples * UINT(pFH->nADCNumChannels);

      // Get the first entry.
      Synch LastItem = { 0 };
      pFI->GetSynchEntry(1, &LastItem);

      // Loop through the rest of the entries.
      for (UINT i=2; i<=uSynchCount; i++)
      {
         // For event detected variable length data files, episodes may be larger then 
         // wFullEpisodeSize. These will be broken up into multiple units of length 
         // uMaxChunkSize or less, and the Synch array adjusted accordingly.

         // Calculate file offsets and expand out any episodes longer than
         // uMaxChunkSize to span multiple Synch entries.
   
         Synch SynchItem;
         pFI->GetSynchEntry(i, &SynchItem);

         // if there are no missing samples, add this length to the previous entry.
         if( SynchItem.dwStart == LastItem.dwStart + SamplesToSynchCounts(pFH, LastItem.dwLength) )
            LastItem.dwLength += SynchItem.dwLength;
         else
         {
            ExpandSynchEntry(pFH, NewSynchArray, &LastItem, uMaxChunkSize, uSampleSize);
            LastItem = SynchItem;
         }
      }

      ExpandSynchEntry(pFH, NewSynchArray, &LastItem, uMaxChunkSize, uSampleSize);

      if (pFI->TestFlag(FI_READONLY))
         NewSynchArray.SetMode(CSynch::eREADMODE);
      
      pFI->ChangeSynchArray(&NewSynchArray);

      *pdwMaxEpi = pFI->GetSynchCount();
   }
   else
   {
//      ERRORMSG("ABF_SetChunkSize should only be used on ABF_GAPFREEFILE or ABF_VARLENEVENTS ABF files");
   }

   // Set header variable for the number of episodes in the file.
   pFH->lActualEpisodes = *pdwMaxEpi;
   pFI->SetAcquiredEpisodes(*pdwMaxEpi);
   pFI->FreeReadBuffer();

   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadEDVarLenSynch
// PURPOSE:  This function shifts and expands the temporary Synch buffer to the Synch
//           array for a Variable-Length-Event-Detected file.
//
static BOOL ReadEDVarLenSynch(CFileDescriptor *pFI, ABFFileHeader *pFH, 
                              DWORD *pdwMaxEpi, UINT *puMaxSamples, int *pnError)
{
   WPTRASSERT(pFI);
//   ABFH_WASSERT(pFH);
   WPTRASSERT(pdwMaxEpi);

   // If a synch array exists, read it into the virtual synch array as is.
   if ((pFH->lSynchArraySize > 0) && (pFH->lSynchArrayPtr > 0))
   {   
      // All variable length and gapfree ABF files use use samples as synch time counts.
      // However, statistics ATF files read in via ATF2ABF32 use synch time units which
      // are NOT samples, so we need to handle that situation.

      // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
      // Read the synch array in chunks, writing it out to the virtual synch array.
      // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

      CFileReadCache SynchFile;
      if (!SynchFile.Initialize(sizeof(ABFSynch), SYNCH_BUFFER_SIZE, pFI->GetFileHandle(), 
                                LONGLONG(pFH->lSynchArrayPtr) * ABF_BLOCKSIZE, 
                                pFH->lSynchArraySize))
         ERRORRETURN(pnError, ABF_OUTOFMEMORY);

      UINT  uSampleSize   = SampleSize(pFH);      
      UINT  uAcqLength    = UINT(pFH->lActualAcqLength);
      UINT  uFileOffset   = 0;
      UINT  uLastStart    = 0;

      for (UINT i=0; i<UINT(pFH->lSynchArraySize); i++)
      {
         ABFSynch *pS = (ABFSynch *)SynchFile.Get(i);
         if (!pS)
            ERRORRETURN(pnError, ABF_EBADSYNCH);

         UINT uStart  = pS->lStart;
         UINT uLength = pS->lLength;
      
         // Check synch entry length.
         if (uLength > uAcqLength)
            ERRORRETURN(pnError, ABF_EBADSYNCH);
      
         // check that entries are consecutive
         if( uStart < uLastStart )
            ERRORRETURN(pnError, ABF_EBADSYNCH);

         uLastStart = uStart;

         pFI->PutSynchEntry(uStart, uLength, uFileOffset);
         uFileOffset += uLength * uSampleSize;
         uAcqLength  -= uLength;
      }
   
      pFI->SetSynchMode( CSynch::eREADMODE );
   }
   return _SetChunkSize( pFI, pFH, puMaxSamples, pdwMaxEpi, pnError );
}


//===============================================================================================
// FUNCTION: _SetOverlap
// PURPOSE:  Changes the overlap flag and processes the synch array to edit redundant data out if no overlap.
//
static BOOL _SetOverlap(CFileDescriptor *pFI, const ABFFileHeader *pFH, BOOL bAllowOverlap, int *pnError)
{
//   ABFH_ASSERT(pFH);

   // Only fixed length events files have overlapping events.
   if (pFH->nOperationMode != ABF_FIXLENEVENTS)
      return TRUE;

   // Fixed length events files always use samples for synch time units.
   ASSERT(pFH->fSynchTimeUnit==0.0F);

   // If none of the sweeps overlap there is nothing to do, get out now.
   if (!pFI->GetOverlappedFlag())
      return TRUE;

   // Create a new synch array that we can build from the old one.
   CSynch NewSynchArray;
   if (!NewSynchArray.OpenFile())
      ERRORRETURN(pnError, ABF_BADTEMPFILE);

   // Cache some useful constants
   const UINT uSynchCount = pFI->GetSynchCount();

   if (bAllowOverlap)
   {
      Synch Item = { 0 };
      UINT uSweepLength = UINT(pFH->lNumSamplesPerEpisode);

      // Loop through entries setting them all to the sweep length.
      for (UINT i=1; i<=uSynchCount; i++)
      {
         pFI->GetSynchEntry(i, &Item);
         NewSynchArray.Put(Item.dwStart, uSweepLength, Item.dwFileOffset);
      }
   }
   else
   {
      // Get the first entry.
      Synch LastItem = { 0 };
      pFI->GetSynchEntry(1, &LastItem);

      // Loop through the rest of the entries.
      for (UINT i=2; i<=uSynchCount; i++)
      {
         Synch SynchItem;
         pFI->GetSynchEntry(i, &SynchItem);

         if ((SynchItem.dwStart != ABF_AVERAGESWEEPSTART) &&
             (LastItem.dwStart != ABF_AVERAGESWEEPSTART))
         {
            // If redundant data is found, then truncate this episode if
            // overlapped data is not to be allowed.
            if (LastItem.dwStart + LastItem.dwLength > SynchItem.dwStart)
               LastItem.dwLength = SynchItem.dwStart - LastItem.dwStart;
         }

         NewSynchArray.Put(LastItem.dwStart, LastItem.dwLength, LastItem.dwFileOffset);
         LastItem = SynchItem;
      }
      NewSynchArray.Put(LastItem.dwStart, LastItem.dwLength, LastItem.dwFileOffset);
   }

   if (pFI->TestFlag(FI_READONLY))
      NewSynchArray.SetMode(CSynch::eREADMODE);

   pFI->ChangeSynchArray(&NewSynchArray);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadEDFixLenSynch
// PURPOSE:  Reads a fixed length synch array off disk and stores it away in a synch buffer.
//           Overlapping episodes are truncated so that the user is only returned data once.
//
static BOOL ReadEDFixLenSynch(CFileDescriptor *pFI, const ABFFileHeader *pFH, DWORD *pdwMaxEpi, 
                              BOOL bAllowOverlap, int *pnError)
{
//   WPTRASSERT(pFI);
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(pdwMaxEpi);
   if ((pFH->lSynchArraySize <= 0) || (pFH->lSynchArrayPtr <= 0))
   {
      // Only waveform files can optionally have a synch array.
      if (pFH->nOperationMode!=ABF_WAVEFORMFILE)
         ERRORRETURN(pnError, ABF_ENOSYNCHPRESENT);

      *pdwMaxEpi = pFH->lActualEpisodes;
      return TRUE;
   }
   
   // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
   // read the synch array in chunks, writing it out to the virtual synch array.
   // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
   
   CFileReadCache SynchFile;
   if (!SynchFile.Initialize(sizeof(ABFSynch), SYNCH_BUFFER_SIZE, pFI->GetFileHandle(), 
                             LONGLONG(pFH->lSynchArrayPtr) * ABF_BLOCKSIZE, pFH->lSynchArraySize))
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   BOOL  bOverlapFound = FALSE;
   UINT  uFileOffset   = 0;
   UINT  uSampleSize   = SampleSize(pFH);      
   UINT  uAcqLength    = UINT(pFH->lActualAcqLength);

   // Get the first entry.
   ABFSynch *pS = (ABFSynch *)SynchFile.Get(0);
   if (!pS)
      ERRORRETURN(pnError, ABF_EBADSYNCH);

   UINT uStart  = pS->lStart;
   UINT uLength = pS->lLength;

   // Loop n-1 times checking the entry against the following one.
   for (UINT i=1; i<UINT(pFH->lSynchArraySize); i++)
   {
      // All episodes should be of the same length.
      ASSERT(uLength == UINT(pFH->lNumSamplesPerEpisode));

      // Check synch entry range
      if (uLength > uAcqLength)
         ERRORRETURN(pnError, ABF_EBADSYNCH);

      // Event detected modes are described by a Synch array that specifies each
      // episode's position and length in the data file.
      pS = (ABFSynch *)SynchFile.Get(i);
      if (!pS)
         ERRORRETURN(pnError, ABF_EBADSYNCH);

      if ((pFH->nOperationMode!=ABF_WAVEFORMFILE) && (uStart!=ABF_AVERAGESWEEPSTART))
      {
         // Only fix-len event detected files can have overlapping sweeps, and these
         // always use samples as synch time counts -- this simplifies comparisons.
         ASSERT(pFH->fSynchTimeUnit==0.0);

         // Some versions of AxoTape produced negative entries in the synch array.
         // DEMOTAPE (Axotape-for-DOS demo version) creates corrupted synch arrays...
         if (long(uStart) < 0)
            ERRORRETURN(pnError, ABF_EBADSYNCH);
            
         // Check for redundant data in following episodes
         if (pS->lStart > -1)
         {
            // check that entries are consecutive
            if (UINT(pS->lStart) <= uStart) 
               ERRORRETURN(pnError, ABF_EBADSYNCH);
         
            // If redundant data is found, then truncate this episode if
            // overlapped data is not to be allowed.
            if (uStart + uLength > UINT(pS->lStart))
               bOverlapFound = TRUE;
         }
      }

      pFI->PutSynchEntry(uStart, uLength, uFileOffset);
      uFileOffset += uLength * uSampleSize;
      uAcqLength  -= uLength;

      uStart  = pS->lStart;
      uLength = pS->lLength;
   }

   // Put the last entry into the synch array.
   pFI->PutSynchEntry(uStart, uLength, uFileOffset);

   *pdwMaxEpi = UINT(pFH->lSynchArraySize);

   pFI->SetSynchMode( CSynch::eREADMODE );
   pFI->SetOverlappedFlag(bOverlapFound);

   return _SetOverlap(pFI, pFH, bAllowOverlap, pnError);
}

//===============================================================================================
// FUNCTION: ReadOldSynchArray
// PURPOSE:  Reads a synch array from an old (pre ABF) data file and stores it away in the synch
//           buffer. Copes with the complexities of old synch arrays (non-trivial).
//
static BOOL ReadOldSynchArray(CFileDescriptor *pFI, ABFFileHeader *pFH, 
                              DWORD *pdwMaxEpi, int *pnError)
{
//   WPTRASSERT(pFI);
//   ABFH_WASSERT(pFH);
//   WPTRASSERT(pdwMaxEpi);
   if ((pFH->lSynchArraySize <= 0) || (pFH->lSynchArrayPtr <= 0))
   {
      if (pFH->nOperationMode != ABF_WAVEFORMFILE)
         ERRORRETURN(pnError, ABF_ENOSYNCHPRESENT);

      *pdwMaxEpi = UINT(pFH->lActualAcqLength / pFH->lNumSamplesPerEpisode);
      return TRUE;
   }

   // Get the length of the file.
   long lFileLength = long(pFI->GetFileSize());
   ASSERT(lFileLength > 0);

   // Old Csynch arrays must be converted to the new style Synch array.
   // dwMaxEpi may be reduced as deleated and empty episodes are stripped out. 
   // This conversion process also fills in file offset entries for each episode.

   // Allocate a temporary buffer and read the Synch array into it.
   // Old synch arrays are guaranteed to be less than 64k, so one read will do it.
   UINT uSize = (UINT)pFH->lSynchArraySize * 2;      // two short entries per episode
   CArrayPtr<short> pnOldSynch(uSize);
   if (pnOldSynch == NULL)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);
   
   // Seek to the start of the synch block.
//   VERIFY(pFI->Seek( LONGLONG(pFH->lSynchArrayPtr) * ABF_BLOCKSIZE, FILE_BEGIN));
   
   // Read the Synch array into the buffer
   if (!pFI->Read(pnOldSynch, uSize*sizeof(short)))
      ERRORRETURN(pnError, ABF_EREADSYNCH);

   // Convert old Synch array to new Synch array, checking for edited
   // (missing) episodes in older file versions.
   UINT  uMissing = 0;
   long  lStart = 0L;
   short *pn = pnOldSynch;
   for (long lSrc=0; lSrc < pFH->lSynchArraySize; lSrc++)
   {
      int nCount  = *pn++;
      int nLength = *pn++;

      long lFileOffset = pFH->lNumSamplesPerEpisode * sizeof(short) * lSrc;

      if (nLength < 0)
      {
         // Zero length acquisition found (i.e. an episode with invalid
         // data) increment Missing% count, but not
         // destination index to effectively edit it out of the file for
         // analysis.

         // Negative Synch entry means that an episode was deleated.
         // Destination index is not incremented, and Missing% count is
         // updated.
         uMissing++;
      }
      else
      {
         long lLength, lSkip;
         
         if (nCount == 0)
         {
            // Adjust the offset for incomplete episodes.
            lFileOffset += pFH->lNumSamplesPerEpisode * sizeof(short) - nLength;
            lSkip = 0;
            lLength = nLength / sizeof(short);   // convert bytes to samples
         }
         else
         {
            // If count is != 0 a full episode was acquired, with possibly missing data before it started.
            lLength = pFH->lNumSamplesPerEpisode;
            lSkip   = pFH->lNumSamplesPerEpisode * long(nCount-1) + long(nLength / sizeof(short));

            // Old fetchan source code disregards MissingSamples if they are less than zero.
            if (lSkip < 0)
               lSkip = 0;
         }

         lStart += lSkip;

         // Check that episode is within the physical file.
         if (lFileOffset+lLength*long(sizeof(short)) > lFileLength-1024)
            ERRORRETURN(pnError, ABF_EBADSYNCH);

         pFI->PutSynchEntry(lStart, lLength, lFileOffset);
         lStart += lLength;
      }
   }
   pFH->lSynchArraySize -= uMissing;
   *pdwMaxEpi = UINT(pFH->lSynchArraySize);
   pFI->SetSynchMode( CSynch::eREADMODE );
   return TRUE;
}

//===============================================================================================
// FUNCTION: GetSynchEntry
// PURPOSE:  Gets a synch entry describing the requested episode (if possible).
// RETURNS:  TRUE = OK, FALSE = Episode number out of range.
// NOTES:    Episode number is one-relative!
//
static BOOL GetSynchEntry( const ABFFileHeader *pFH, CFileDescriptor *pFI, UINT uEpisode, 
                           Synch *pSynchEntry )
{
   if (!pFI->CheckEpisodeNumber(uEpisode))
      return FALSE;
      
   // If a synch array is not present, create a synch entry for this chunk,
   // otherwise, read it from the synch array.
   if (pFI->GetSynchCount() == 0)
   {
      UINT uSampleSize = SampleSize(pFH);
      UINT uChunkSize  = UINT(pFH->lNumSamplesPerEpisode);    // Chunk size in samples
      
      // In continuous files, the last episode may be smaller than the episode size used
      // for the rest of the file. This is calculated in the ABF.Open routine.
      if ((pFH->nOperationMode == ABF_GAPFREEFILE) && (uEpisode == pFI->GetAcquiredEpisodes()))
         pSynchEntry->dwLength = pFI->GetLastEpiSize();
      else
         pSynchEntry->dwLength = uChunkSize;
         
      pSynchEntry->dwFileOffset = uChunkSize * uSampleSize * (uEpisode - 1);
      pSynchEntry->dwStart      = pSynchEntry->dwFileOffset / uSampleSize;
            
      return TRUE;
   }
   return pFI->GetSynchEntry( uEpisode, pSynchEntry );
}


//===============================================================================================
// FUNCTION: ABF_MultiplexRead
// PURPOSE:  This routine reads an episode of data from the data file previously opened.
//
// INPUT:
//   nFile          the file index into the g_FileData structure array
//   dwEpisode      the episode number to be read. Episodes start at 1
// 
// OUTPUT:
//   pvBuffer       the data buffer for the data
//   puSizeInSamples the number of valid points in the data buffer
// 
BOOL WINAPI ABF_MultiplexRead(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                              void *pvBuffer, UINT *puSizeInSamples, int *pnError)
{
//   ABFH_ASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // Set the sample size in the data.
   UINT uSampleSize = SampleSize(pFH);
   UINT uBytesPerEpisode = UINT(pFH->lNumSamplesPerEpisode) * uSampleSize;

   // If a synch array is not present, create a synch entry for this chunk,
   // otherwise, read it from the synch array.
   Synch SynchEntry;
   if (!GetSynchEntry( pFH, pFI, dwEpisode, &SynchEntry ))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);
      
   // return the size of the episode to be read.
   if (puSizeInSamples)
      *puSizeInSamples = UINT(SynchEntry.dwLength);

   // Add the distance to the start of the data to the data offset
   LONGLONG lFileOffset = LONGLONG(GetDataOffset(pFH)) + SynchEntry.dwFileOffset;

   // Seek to the calculated file position.
   VERIFY(pFI->Seek(lFileOffset, FILE_BEGIN));

   UINT uSizeInBytes = SynchEntry.dwLength * uSampleSize;
//   ARRAYASSERT((BYTE *)pvBuffer, uSizeInBytes);

   // Do the file read
   if (!pFI->Read(pvBuffer, uSizeInBytes))
      ERRORRETURN(pnError, ABF_EREADDATA);

   // If episode is not full, pad it out with 0's
   if (uSizeInBytes < uBytesPerEpisode)
      memset((char *)pvBuffer + uSizeInBytes, '\0', uBytesPerEpisode - uSizeInBytes);

   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: SynchCountToSamples
// PURPOSE:  Rounds a synch count to the nearest sample count.
//
inline UINT SynchCountToSamples(const ABFFileHeader *pFH, UINT uSynchStart)
{
   double dMS = 0.0;
   ABFH_SynchCountToMS(pFH, uSynchStart, &dMS);
   double dSampleInterval = ABFH_GetFirstSampleInterval(pFH);
   return UINT(dMS/dSampleInterval*1E3 + 0.5);
}

//===============================================================================================
// FUNCTION: ABF_MultiplexWrite
// PURPOSE:  This routine writes an episode of data from the end of the data file
//           previously opened with a ABF_WriteOpen call. Episodes may only be written
//           sequentially.
// INPUT:
//   nFile          the file index into the g_FileData structure array
//   uFlags         flags governing the write process
//   uSizeInSamples the number of valid points in the data buffer
//   dwEpiStart     the start time (in synch time units) of this episode
//   pvBuffer       the data buffer for the data
// 
BOOL WINAPI ABF_MultiplexWrite(int nFile, ABFFileHeader *pFH, UINT uFlags, const void *pvBuffer, 
                               DWORD dwEpiStart, UINT uSizeInSamples, int *pnError)
{
   ABFH_WASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // Return an error if writing is inappropriate.
   if (pFI->TestFlag(FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);

   // Check parameters in a debug build.
#ifdef _DEBUG
   if ((pFH->nOperationMode==ABF_WAVEFORMFILE) || 
       (pFH->nOperationMode==ABF_HIGHSPEEDOSC) || 
       (pFH->nOperationMode==ABF_FIXLENEVENTS))
   {
      ASSERT(pFH->lNumSamplesPerEpisode==long(uSizeInSamples));
      ASSERT((uFlags & ABF_APPEND) == 0);
   }
#endif

   // Set the sample size in the data.
   UINT uSampleSize  = SampleSize(pFH);
   UINT uSizeInBytes = uSizeInSamples * uSampleSize;
   ARRAYASSERT((short *)pvBuffer, uSizeInBytes/2);

   // Seek to the end of the file.
   VERIFY(pFI->Seek( 0L, FILE_END));

   if (!pFI->Write(pvBuffer, uSizeInBytes))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   UINT uAcquiredEpisodes = pFI->GetAcquiredEpisodes();
   UINT uAcquiredSamples  = pFI->GetAcquiredSamples();
   UINT uSynchCount       = pFI->GetSynchCount();

   // Clear the append flag if there is nothing to append to.
   if (uSynchCount == 0)
      uFlags &= ~ABF_APPEND;

   switch (pFH->nOperationMode)
   {
      case ABF_GAPFREEFILE:
      {
         UINT uEpiStartInSamples = SynchCountToSamples(pFH, dwEpiStart);

         // If there is a synch array already...
         if (uSynchCount != 0)
         {
            UINT uStartOfLast = SynchCountToSamples(pFH, pFI->EpisodeStart( uAcquiredEpisodes ));
            UINT uEndOfLast   = uStartOfLast + pFI->EpisodeLength( uAcquiredEpisodes );
            if (uEpiStartInSamples <= uEndOfLast)      // If we are just appending onto the previous event...
               uFlags |= ABF_APPEND;
            // FALL THROUGH TO DEFAULT CASE FOR SYNCH ARRAY PROCESSING!!!
         }
         else  // No synch array as yet - either append or add one.
         {
            // if the first block is being extended - no synch array required.
            if (uEpiStartInSamples <= uAcquiredSamples) 
            {
               pFI->SetAcquiredEpisodes( 1 );
               break;
            }

            // If some data has been acquired but no synch entries added...
            if (uAcquiredSamples > 0) 
            {
               // Add in the first synch entry for data already written
               pFI->PutSynchEntry(0, uAcquiredSamples, 0);
               pFI->SetAcquiredEpisodes( 1 );
            }
            // FALL THROUGH TO DEFAULT CASE FOR SYNCH ARRAY PROCESSING!!!
         }
      }
      
      default:
         if (uFlags & ABF_APPEND)
            pFI->IncreaseEventLength( uSizeInSamples );
         else
         {
            pFI->PutSynchEntry(dwEpiStart, uSizeInSamples, uAcquiredSamples * uSampleSize);
            pFI->SetAcquiredEpisodes(++uAcquiredEpisodes);
         }
         break;
   }

   uAcquiredSamples += uSizeInSamples;
   pFI->SetAcquiredSamples(uAcquiredSamples);
   pFH->lActualAcqLength = (long)uAcquiredSamples;
   pFH->lActualEpisodes = (long)pFI->GetAcquiredEpisodes();

   return TRUE;
}


//===============================================================================================
// FUNCTION: ABF_SetEpisodeStart
// PURPOSE:  Sets the start time of a given sweep in synch time units.
// INPUT:
//   nFile          the file index into the g_FileData structure array
//   uEpisode       the (one based) episode number.
//   uEpiStart      the start time (in synch time units) of this episode
// 
BOOL WINAPI ABF_SetEpisodeStart(int nFile, UINT uEpisode, UINT uEpiStart, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // Return an error if writing is inappropriate.
   if (pFI->TestFlag(FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);

   UINT uSynchCount = pFI->GetSynchCount();
   if (uEpisode > uSynchCount)
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   pFI->SetEpisodeStart(uEpisode, uEpiStart);
   return TRUE;
}


//===============================================================================================
// FUNCTION: ABF_WriteRawData
// PURPOSE:  This routine writes a raw buffer of binary data to the current position of an 
//           ABF file previously opened with a ABF_WriteOpen call. This routine is provided
//           for acquisition programs that buffer up episodic data and then write it out in 
//           large chunks. This provides an alternative to retrieving the low-level file handle 
//           and acting on it, as this can be non-portable, and assumptions would have to be 
//           made regarding the type of file handle returned (DOS or C runtime).
// INPUT:
//   nFile          the file index into the g_FileData structure array
//   pvBuffer       the data buffer for the data
//   dwSizeInBytes  the number of bytes of data to write
// 
BOOL WINAPI ABF_WriteRawData(int nFile, const void *pvBuffer, DWORD dwSizeInBytes, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
#ifdef _DEBUG
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   
#endif

   ARRAYASSERT((short *)pvBuffer, UINT(dwSizeInBytes/2));
   if (!pFI->Write(pvBuffer, dwSizeInBytes))
      ERRORRETURN(pnError, ABF_EDISKFULL);
   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: PackSamples
// PURPOSE:  Packs the samples from the source array into the destination array,
//           assuming the given skip factor
// INPUT:
//   pvSource        the pointer to the source of data.
//   pvDestination   the pointer to the destination of data.
//   uSourceLen      the length of the data to be packed
//   uFirstSample    the starting index of the first element
//   uSkip           the skip factor for the packing
//
static void PackSamples(void *pvSource, void *pvDestination, UINT uSourceLen, UINT uFirstSample,
                        UINT uSampleSize, UINT uSkip)
{
   ASSERT(uSkip > 0);
//   ARRAYASSERT((BYTE *)pvSource, uSourceLen * uSampleSize);
//   ARRAYASSERT((BYTE *)pvDestination, (uSourceLen / uSkip) * uSampleSize);

   if (uSampleSize == sizeof(short))
   {
      // adjust the starting offset
      short *piSource      = (short *)pvSource;
      short *piDestination = (short *)pvDestination;
      for (UINT i=uFirstSample; i<uSourceLen; i+=uSkip)
         *piDestination++ = piSource[i];
   }
   else 
   {
      // adjust the starting offset
      long *plSource      = (long *)pvSource;
      long *plDestination = (long *)pvDestination;
      for (UINT i=uFirstSample; i<uSourceLen; i+=uSkip)
         *plDestination++ = plSource[i];
   }
}

//===============================================================================================
// FUNCTION: ConvertADCToFloats
// PURPOSE:  Convert an array of ADC values to UserUnits.
//
static void ConvertADCToFloats( const ABFFileHeader *pFH, int nChannel, UINT uChannelOffset, 
                                float *pfDestination, short *pnSource )
{
//   ABFH_ASSERT(pFH);
//   ARRAYASSERT(pfDestination, (UINT)(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels));
//   ARRAYASSERT(pnSource, (UINT)(pFH->lNumSamplesPerEpisode));
   
   UINT uSkip      = (UINT)pFH->nADCNumChannels;
   UINT uSourceLen = (UINT)pFH->lNumSamplesPerEpisode;
   
   float fValToUUFactor, fValToUUShift;
   ABFH_GetADCtoUUFactors( pFH, nChannel, &fValToUUFactor, &fValToUUShift);

   for (UINT i=uChannelOffset; i<uSourceLen; i+=uSkip)
      *pfDestination++ = pnSource[i] * fValToUUFactor + fValToUUShift;
}

//===============================================================================================
// FUNCTION: ConvertInPlace
// PURPOSE:  Convert a single channel of two byte integers to floats, in-place.
//
static void ConvertInPlace(const ABFFileHeader *pFH, int nChannel, UINT uNumSamples, void *pvBuffer)
{
//   ABFH_ASSERT(pFH);
//   ARRAYASSERT((float *)pvBuffer, uNumSamples);
   
   ADC_VALUE *pnSource      = ((ADC_VALUE *)pvBuffer);
   float     *pfDestination = ((float *)pvBuffer);
   
   float fValToUUFactor, fValToUUShift;
   ABFH_GetADCtoUUFactors( pFH, nChannel, &fValToUUFactor, &fValToUUShift);

   for (int i=uNumSamples-1; i>=0; i--)
      pfDestination[i] = pnSource[i] * fValToUUFactor + fValToUUShift;
}

//===============================================================================================
// FUNCTION: ConvertADCToResults
// PURPOSE:  Get the results array for the math channel.
//
static BOOL ConvertADCToResults(const ABFFileHeader *pFH, float *pfDestination, short *pnSource)
{
//   ABFH_ASSERT(pFH);
//   ARRAYASSERT(pfDestination, (UINT)(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels));
//   ARRAYASSERT(pnSource, (UINT)(pFH->lNumSamplesPerEpisode));
   UINT uAOffset, uBOffset;
   short *pnSourceA, *pnSourceB;

   int nChannelA = pFH->nArithmeticADCNumA;
   int nChannelB = pFH->nArithmeticADCNumB;

   UINT i, uSkip = pFH->nADCNumChannels;
   UINT uSourceArrayLen = (UINT)pFH->lNumSamplesPerEpisode;

   float fValToUUFactorA, fValToUUShiftA;
   float fValToUUFactorB, fValToUUShiftB;
   float fUserUnitA, fUserUnitB;

   if (!ABFH_GetChannelOffset(pFH, nChannelA, &uAOffset))
      return FALSE;

   if (!ABFH_GetChannelOffset(pFH, nChannelB, &uBOffset))
      return FALSE;

   ABFH_GetADCtoUUFactors( pFH, nChannelA, &fValToUUFactorA, &fValToUUShiftA);
   ABFH_GetADCtoUUFactors( pFH, nChannelB, &fValToUUFactorB, &fValToUUShiftB);

   pnSourceA = pnSource + uAOffset;  // adjust the starting offset
   pnSourceB = pnSource + uBOffset;  // adjust the starting offset
   uSourceArrayLen -= max(uAOffset, uBOffset);
   for (i=0; i<uSourceArrayLen; i+=uSkip)
   {
      fUserUnitA = pnSourceA[i] * fValToUUFactorA + fValToUUShiftA;
      fUserUnitB = pnSourceB[i] * fValToUUFactorB + fValToUUShiftB;

      ABFH_GetMathValue(pFH, fUserUnitA, fUserUnitB, pfDestination++);
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: ConvertToResults
// PURPOSE:  Fills the math channel array from a multichannel buffer of float's.
//
static BOOL ConvertToResults(const ABFFileHeader *pFH, float *pfDestination, float *pfSource)
{
//   ARRAYASSERT(pfDestination, pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels);
//   ARRAYASSERT(pfSource, pFH->lNumSamplesPerEpisode);
   
   int nChannelA = pFH->nArithmeticADCNumA;
   int nChannelB = pFH->nArithmeticADCNumB;

   UINT uSkip           = pFH->nADCNumChannels;
   UINT uSourceArrayLen = (UINT)pFH->lNumSamplesPerEpisode;

   UINT uAOffset, uBOffset;
   if (!ABFH_GetChannelOffset(pFH, nChannelA, &uAOffset))
      return FALSE;

   if (!ABFH_GetChannelOffset(pFH, nChannelB, &uBOffset))
      return FALSE;

   float *pfSourceA = pfSource + uAOffset;  // adjust the starting offset
   float *pfSourceB = pfSource + uBOffset;  // adjust the starting offset
   uSourceArrayLen -= max(uAOffset, uBOffset);
   for (UINT i=0; i<uSourceArrayLen; i+=uSkip)
      ABFH_GetMathValue(pFH, pfSourceA[i], pfSourceB[i], pfDestination++);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_ReadChannel
// PURPOSE:  This function reads a complete multiplexed episode from the data file and
//           then converts a single de-multiplexed channel to "UserUnits" in pfBuffer.
//
// The required size of the passed buffer is:
// pfBuffer     -> pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels  (floats)
//
BOOL WINAPI ABF_ReadChannel(int nFile, const ABFFileHeader *pFH, int nChannel, DWORD dwEpisode, 
                            float *pfBuffer, UINT *puNumSamples, int *pnError)
{
//   ABFH_ASSERT(pFH);
//   ARRAYASSERT(pfBuffer, (UINT)(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels));
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // Get the offset into the multiplexed data array for the first point
   UINT uChannelOffset;
   if (!ABFH_GetChannelOffset(pFH, nChannel, &uChannelOffset))
      ERRORRETURN(pnError, ABF_EINVALIDCHANNEL);

   // If there is only one channel, read the data directly into the passed buffer,
   // converting it in-place if required.
   if ((pFH->nADCNumChannels == 1) && (nChannel >= 0))
   {
      if (!ABF_MultiplexRead(nFile, pFH, dwEpisode, pfBuffer, puNumSamples, pnError))
         return FALSE;

      if (pFH->nDataFormat == ABF_INTEGERDATA)      // if data is 2byte ints, convert to floats
         ConvertInPlace(pFH, nChannel, *puNumSamples, pfBuffer);
      return TRUE;
   }      

   // Set the sample size in the data.
   UINT uSampleSize = SampleSize(pFH);

   // Only create the read buffer on demand, it is freed when the file is closed.
   if (!pFI->GetReadBuffer())
   {      
      if (!pFI->AllocReadBuffer(pFH->lNumSamplesPerEpisode * uSampleSize))
         ERRORRETURN(pnError, ABF_OUTOFMEMORY);
   }

   // Read the whole episode from the ABF file only if it is not already cached.
   UINT uEpisodeSize = pFI->GetCachedEpisodeSize();
   if (dwEpisode != pFI->GetCachedEpisode())
   {         
      uEpisodeSize = (UINT)pFH->lNumSamplesPerEpisode;
      if (!ABF_MultiplexRead(nFile, pFH, dwEpisode, pFI->GetReadBuffer(), &uEpisodeSize, pnError))
      {
         pFI->SetCachedEpisode(UINT(-1), 0);
         return FALSE;
      }
      pFI->SetCachedEpisode(dwEpisode, uEpisodeSize);
   }
   
   // if data is 2byte ints, convert to floats
   if (pFH->nDataFormat == ABF_INTEGERDATA)
   {
      // Cast the read buffer to the appropriate format.
      ADC_VALUE *pnReadBuffer = (ADC_VALUE *)pFI->GetReadBuffer();

      // A channel number of -1 refers to the results channel
      if (nChannel >= 0)
         ConvertADCToFloats(pFH, nChannel, uChannelOffset, pfBuffer, pnReadBuffer);
      else if (!ConvertADCToResults(pFH, pfBuffer, pnReadBuffer))
         ERRORRETURN(pnError, ABF_BADMATHCHANNEL);
   }
   else     // Data is 4-byte floats.
   {
      // Cast the read buffer to the appropriate format.
      float *pfReadBuffer = (float *)pFI->GetReadBuffer();

      // A channel number of -1 refers to the results channel
      if (nChannel >= 0)
         PackSamples(pfReadBuffer, pfBuffer, uEpisodeSize, uChannelOffset,
                     uSampleSize, pFH->nADCNumChannels);
      else if (!ConvertToResults(pFH, pfBuffer, pfReadBuffer))
         ERRORRETURN(pnError, ABF_BADMATHCHANNEL);
   }
   
   // Return the length of the data block.
   if (puNumSamples)
      *puNumSamples = uEpisodeSize / pFH->nADCNumChannels;
   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: ABF_ReadRawChannel
// PURPOSE:  This function reads a complete multiplexed episode from the data file and
//           then decimates it, returning single de-multiplexed channel in the raw data format.
//
// The required size of the passed buffer is:
// pfBuffer     -> pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels  (floats)
//
BOOL WINAPI ABF_ReadRawChannel(int nFile, const ABFFileHeader *pFH, int nChannel, DWORD dwEpisode, 
                               void *pvBuffer, UINT *puNumSamples, int *pnError)
{
   ABFH_ASSERT(pFH);

   // Set the sample size in the data.
   UINT uSampleSize = SampleSize(pFH);
   ARRAYASSERT((short *)pvBuffer, pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels*uSampleSize/2);
   
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // Get the offset into the multiplexed data array for the first point
   UINT uChannelOffset;
   if (!ABFH_GetChannelOffset(pFH, nChannel, &uChannelOffset) || (nChannel < 0))
      ERRORRETURN(pnError, ABF_EINVALIDCHANNEL);

   // If there is only one channel, read the data directly into the passed buffer,
   if (pFH->nADCNumChannels == 1)
      return ABF_MultiplexRead(nFile, pFH, dwEpisode, pvBuffer, puNumSamples, pnError);

   // Only create the read buffer on demand, it is freed when the file is closed.
   if (!pFI->GetReadBuffer())
   {      
      if (!pFI->AllocReadBuffer(pFH->lNumSamplesPerEpisode * uSampleSize))
         ERRORRETURN(pnError, ABF_OUTOFMEMORY);
   }

   // Read the whole episode from the ABF file only if it is not already cached.
   UINT uEpisodeSize = pFI->GetCachedEpisodeSize();
   if (dwEpisode != pFI->GetCachedEpisode())
   {         
      uEpisodeSize = (UINT)pFH->lNumSamplesPerEpisode;
      if (!ABF_MultiplexRead(nFile, pFH, dwEpisode, pFI->GetReadBuffer(), &uEpisodeSize, pnError))
      {
         pFI->SetCachedEpisode(UINT(-1), 0);
         return FALSE;
      }
      pFI->SetCachedEpisode(dwEpisode, uEpisodeSize);
   }
   
   PackSamples(pFI->GetReadBuffer(), pvBuffer, uEpisodeSize, uChannelOffset,
               uSampleSize, pFH->nADCNumChannels);
   
   // Return the length of the data block.
   if (puNumSamples)
      *puNumSamples = uEpisodeSize / pFH->nADCNumChannels;
   return TRUE;
}

                                   
//===============================================================================================
// FUNCTION: ABF_ReadDACFileEpi
// PURPOSE:  This function reads an episode from the DACFile section. Users will normally
//           retrieve DAC file information transparently through the Get Waveform call.
//
BOOL WINAPI ABF_ReadDACFileEpi(int nFile, const ABFFileHeader *pFH, short *pnDACArray,
                               DWORD dwEpisode, int *pnError)
{
   return ABF_ReadDACFileEpiEx(nFile, pFH, pnDACArray, pFH->nActiveDACChannel, dwEpisode, pnError);
}

BOOL WINAPI ABF_ReadDACFileEpiEx(int nFile, const ABFFileHeader *pFH, short *pnDACArray,
                                 UINT nChannel, DWORD dwEpisode, int *pnError)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

#if USE_DACFILE_FIX
// PRC DEBUG
//   UINT uNumSamples = NewFH.lNumSamplesPerEpisode / NewFH.nADCNumChannels;
   UINT uNumSamples = NewFH.lNumSamplesPerEpisode;
#else
   UINT uNumSamples = NewFH.lNumSamplesPerEpisode / NewFH.nADCNumChannels;
#endif

   ARRAYASSERT( pnDACArray, uNumSamples );
   ASSERT( nChannel < ABF_WAVEFORMCOUNT );

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // If the requested episode is after the last one, then use the last one in the file.
   if( NewFH.lDACFileNumEpisodes[nChannel] < (long)dwEpisode )
      dwEpisode = (DWORD) NewFH.lDACFileNumEpisodes[nChannel];

   if (NewFH.lDACFilePtr[nChannel]==0)
   {
      if (!pFI->GetDACFileSweep(nChannel, dwEpisode-1, pnDACArray, uNumSamples))
         ERRORRETURN(pnError, ABF_EREADDACEPISODE);
   }
   else
   {
      UINT uOffset = NewFH.lDACFilePtr[nChannel] * ABF_BLOCKSIZE + 
                     (dwEpisode-1) * uNumSamples * sizeof(short);
      VERIFY(pFI->Seek( uOffset, FILE_BEGIN));

      // Read the DACFile episode into the passed buffer
      UINT uBytesToRead = uNumSamples * sizeof(short);
      if (!pFI->Read(pnDACArray, uBytesToRead))
         ERRORRETURN(pnError, ABF_EREADDACEPISODE);

#if USE_DACFILE_FIX
      // PRC DEBUG
      // Tempory hack to decimate by number of channels.
      int nNumChans = NewFH.nADCNumChannels;
      if( nNumChans > 1 )
      {
         for( UINT i=0; i<uNumSamples; i++ )
         {
            pnDACArray[i/nNumChans] = pnDACArray[i];
         }
      }
#endif
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_WriteDACFileEpi
// PURPOSE:  This function writes an episode to the DACFile section. Episodes must be
//           written sequentially to the DACFile section, after the data has been written
//           to the DATA section.
//
BOOL WINAPI ABF_WriteDACFileEpi(int nFile, ABFFileHeader *pFH, const short *pnDACArray, int *pnError)
{
   return ABF_WriteDACFileEpiEx(nFile, pFH, pFH->nActiveDACChannel, pnDACArray, pnError);
}


BOOL WINAPI ABF_WriteDACFileEpiEx(int nFile, ABFFileHeader *pFH, UINT uDACChannel, const short *pnDACArray, int *pnError)
{
   ABFH_WASSERT(pFH);
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Coerce to safe value.
   if( uDACChannel >= ABF_WAVEFORMCOUNT )
   {
      TRACE1( "ABF_WriteDACFileEpi: uDACChannel changed from %d to 0.\n", uDACChannel );
      uDACChannel = 0;
   }
   ARRAYASSERT(pnDACArray, (UINT)(pFH->lNumSamplesPerEpisode));

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   // Take a copy of the passed in header to ensure it is 5k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   if (!pFI->PutDACFileSweep( uDACChannel, NewFH.lDACFileNumEpisodes[uDACChannel], pnDACArray, NewFH.lNumSamplesPerEpisode ))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   NewFH.lDACFileNumEpisodes[uDACChannel]++;

   // Copy the original parameters back into the old header.
   ABFH_DemoteHeader( pFH, &NewFH );

   return TRUE;
}


#if USE_DACFILE_FIX
// PRC DEBUG
   static int s_nFudgeChannels = -1;
#endif


//===============================================================================================
// FUNCTION: ScaleDACBuffer
// PURPOSE:  Fill the float buffer with DAC UU values that correspond to a particular 
//           multiplex offset.
//
static void ScaleDACBuffer(const ABFFileHeader *pFH, UINT uDACChannel, UINT uADCChannelOffset, 
                           short *pnReadBuffer, float *pfBuffer)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Coerce to safe value.
   if( uDACChannel >= ABF_WAVEFORMCOUNT )
   {
      TRACE1( "ScaleDACBuffer: uDACChannel changed from %d to 0.\n", uDACChannel );
      uDACChannel = 0;
   }

   UINT uNumSamples = (UINT)pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels;

#if USE_DACFILE_FIX
// PRC DEBUG
   ARRAYASSERT(pnReadBuffer, pFH->lNumSamplesPerEpisode );
#else
   ARRAYASSERT(pnReadBuffer, uNumSamples);
#endif

   ARRAYASSERT(pfBuffer, uNumSamples);
   
   float fDACToUUFactor, fDACToUUShift;
   ABFH_GetDACtoUUFactors( pFH, uDACChannel, &fDACToUUFactor, &fDACToUUShift );

#if USE_DACFILE_FIX
   UINT uNumDACFileChannels = pFH->nADCNumChannels + s_nFudgeChannels;
   for (UINT i=uADCChannelOffset; i<uNumSamples; i+=uNumDACFileChannels)
#else
   for (UINT i=uADCChannelOffset; i<uNumSamples; i++)
#endif
      *pfBuffer++ = pnReadBuffer[i] * fDACToUUFactor + fDACToUUShift;
}



//===============================================================================================
// FUNCTION: ABF_GetWaveform
// PURPOSE:  This function forms the de-multiplexed DAC output waveform for the
//           particular channel in the pfBuffer, in DAC UserUnits.
//
BOOL WINAPI ABF_GetWaveform(int nFile, const ABFFileHeader *pFH, int nADCChannel, DWORD dwEpisode, 
                            float *pfBuffer, int *pnError)
{
   // Note: we now ignore the nADCChannel parameter.
   return ABF_GetWaveformEx(nFile, pFH, pFH->nActiveDACChannel, dwEpisode, 
                            pfBuffer, pnError);
}

BOOL WINAPI ABF_GetWaveformEx(int nFile, const ABFFileHeader *pFH, UINT uDACChannel, DWORD dwEpisode, 
                            float *pfBuffer, int *pnError)
{
  ABFH_ASSERT(pFH);
  ARRAYASSERT(pfBuffer, (UINT)(pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels));

   if( pFH->nOperationMode != ABF_WAVEFORMFILE )
      ERRORRETURN(pnError, ABF_ENOWAVEFORM);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );
   
   if( (NewFH.nWaveformEnable[uDACChannel] == FALSE) ||
	   (NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED))
      ERRORRETURN(pnError, ABF_ENOWAVEFORM);

   if (NewFH.nWaveformSource[uDACChannel] == ABF_EPOCHTABLEWAVEFORM)
   {
      if (!ABFH_GetWaveformEx( &NewFH, uDACChannel, dwEpisode, pfBuffer, NULL))
         ERRORRETURN(pnError, ABF_EBADWAVEFORM);
      return TRUE;
   }
   
   ASSERT(NewFH.nWaveformSource[uDACChannel] == ABF_DACFILEWAVEFORM);
   
#if USE_DACFILE_FIX
// PRC DEBUG
   CArrayPtr<DAC_VALUE> pnWorkBuffer(NewFH.lNumSamplesPerEpisode);
#else
   CArrayPtr<DAC_VALUE> pnWorkBuffer(NewFH.lNumSamplesPerEpisode / NewFH.nADCNumChannels);
#endif
   if (!pnWorkBuffer)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   if (!ABF_ReadDACFileEpiEx(nFile, &NewFH, pnWorkBuffer, uDACChannel, dwEpisode, pnError))
      return FALSE;

   ScaleDACBuffer(&NewFH, uDACChannel, 0, pnWorkBuffer, pfBuffer);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_WriteTag
// PURPOSE:  This function buffers tags to a temporary file through the CABFItem object in the
//           file descriptor.
//
BOOL WINAPI ABF_WriteTag(int nFile, ABFFileHeader *pFH, const ABFTag *pTag, int *pnError)
{
   ABFH_WASSERT(pFH);
   WPTRASSERT(pTag);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->PutTag(pTag))
      ERRORRETURN(pnError, pFI->GetLastError());
      
   pFH->lNumTagEntries = pFI->GetTagCount();
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_UpdateTag
// PURPOSE:  This function updates a tag entry in a writeable file.
//
BOOL WINAPI ABF_UpdateTag(int nFile, UINT uTag, const ABFTag *pTag, int *pnError)
{
   WPTRASSERT(pTag);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->UpdateTag(uTag, pTag))
      ERRORRETURN(pnError, pFI->GetLastError());
      
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_ReadTags
// PURPOSE:  This function reads a tag array from the TagArray section
//
BOOL WINAPI ABF_ReadTags(int nFile, const ABFFileHeader *pFH, DWORD dwFirstTag, 
                         ABFTag *pTagArray, UINT uNumTags, int *pnError)
{
   ABFH_ASSERT(pFH);
   ARRAYASSERT(pTagArray, uNumTags);
   UINT i;

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   // If this file is being written, the tags will be in the virtual tag buffer.
   if (pFI->GetTagCount() > 0)
   {
      if (!pFI->ReadTags(dwFirstTag, pTagArray, uNumTags))
         ERRORRETURN(pnError, ABF_EREADTAG);
      return TRUE;
   }

   // If there are no tags present, return an error.   
   if ((pFH->lTagSectionPtr==0) || (pFH->lNumTagEntries==0))
      ERRORRETURN(pnError, ABF_ENOTAGS);
      
   if (dwFirstTag+uNumTags > UINT(pFH->lNumTagEntries))
      ERRORRETURN(pnError, ABF_EREADTAG);

   // Read and convert old FETCHEX tags.
   if (pFH->nFileType != ABF_ABFFILE)
   {
      // Seek to the start of the requested segment (first entry is the count of tags, this is
      // placed in pFH->lNumTagEntries when the header is read).
      UINT uSeekPos = UINT(pFH->lTagSectionPtr) * ABF_BLOCKSIZE + dwFirstTag * sizeof(long) + sizeof(long);
      VERIFY(pFI->Seek(uSeekPos, FILE_BEGIN));

      // Allocate a temporary buffer to read the old tags into.      
      CArrayPtr<long> plTags(uNumTags);
      if (!plTags)
         ERRORRETURN(pnError, ABF_OUTOFMEMORY);
      
      // Do the read.   
      if (!pFI->Read(plTags, uNumTags * sizeof(long)))
      {
         TRACE( "Tags could not be read from the file.\n" );
         // Do not flag the error - this allows the tags to be quietly ignored.
         // ERRORRETURN(pnError, ABF_EREADTAG);
      }
      
      // Convert the tags to ABFtags.
      for (i=0; i<uNumTags; i++)
      {
         // Only add tags with a positive time.
         if( plTags[i] >= 0 )
         {
            pTagArray[i].lTagTime = plTags[i];
            memset(pTagArray[i].sComment, ' ', ABF_TAGCOMMENTLEN);
            pTagArray[i].nTagType = ABF_TIMETAG;
            pTagArray[i].nVoiceTagNumber = 0;
         }
      }
   }
   else
   {
      // Seek to the start of the requested segment.
      LONGLONG llSeekPos = LONGLONG(pFH->lTagSectionPtr) * ABF_BLOCKSIZE + dwFirstTag * sizeof(ABFTag);
      VERIFY(pFI->Seek(llSeekPos, FILE_BEGIN));
   
      // Read the Tag Array directly into the passed buffer
      UINT uBytesToRead = uNumTags * sizeof(ABFTag);
      if (!pFI->Read(pTagArray, uBytesToRead))
         ERRORRETURN(pnError, ABF_EREADTAG);
   }   
   
   // AxoTape V2.0 filled the comment field with '\0's - convert to spaces.
   if (pFH->fFileVersionNumber < 1.3F)
   {
      // Set the comment string to all spaces.
      for (i=0; i<uNumTags; i++)
      {
         memset(pTagArray[i].sComment, ' ', ABF_TAGCOMMENTLEN);
         pTagArray[i].nTagType = ABF_TIMETAG;
         pTagArray[i].nVoiceTagNumber = 0;
      }
   }
   
   return TRUE;
}

static char *GetTagComment(ABFTag *pTag)
{
   static char szRval[ABF_TAGCOMMENTLEN+1];
   char *ps = pTag->sComment;
   UINT i=0;
   for (i=0; i<ABF_TAGCOMMENTLEN; i++)
      if (*ps++!=' ')
         break;
   if (i<ABF_TAGCOMMENTLEN)
   {
      strncpy(szRval, pTag->sComment, ABF_TAGCOMMENTLEN-i);
      szRval[ABF_TAGCOMMENTLEN-i] = '\0';
   }
   else
      LoadString(g_hInstance, IDS_NONE, szRval, sizeof(szRval));
   return szRval;
}


//===============================================================================================
// FUNCTION: ABF_FormatTag
// PURPOSE:  This function reads a tag TagArray section and formats it as ASCII text.
// NOTE:     If tag number -1 is requested, the ASCII text returns column headings.
//
BOOL WINAPI ABF_FormatTag(int nFile, const ABFFileHeader *pFH, long lTagNumber, 
                          char *pszBuffer, UINT uSize, int *pnError)
{
   ABFH_ASSERT(pFH);
   ARRAYASSERT(pszBuffer, uSize);
   
   BOOL bEpisodic = ((pFH->nOperationMode==ABF_WAVEFORMFILE) || (pFH->nOperationMode==ABF_HIGHSPEEDOSC));
   if (lTagNumber < 0)
   {
      int nStringID = bEpisodic ? IDS_EPITAGHEADINGS : IDS_CONTTAGHEADINGS;
      return (BOOL)LoadString(g_hInstance, nStringID, pszBuffer, uSize);
   }

   ABFTag Tag;
   char szBuf[ABF_MAXTAGFORMATLEN+4];
   if (!ABF_ReadTags(nFile, pFH, UINT(lTagNumber), &Tag, 1, pnError))
      return FALSE;

   double dTimeInMS = 0.0;
   ABFH_SynchCountToMS(pFH, Tag.lTagTime, &dTimeInMS);

   char szTagTime[32];
   ABFU_FormatDouble(dTimeInMS/1E3, 10, szTagTime, sizeof(szTagTime));
   
   char *ps = GetTagComment(&Tag);

   if (bEpisodic)
   {
      DWORD dwEpisode = 1;
      DWORD dwSynchCount = Tag.lTagTime;
      ABF_EpisodeFromSynchCount(nFile, pFH, &dwSynchCount, &dwEpisode, NULL);
      // "Tag #   Time (s)  Episode  Comment"
      sprintf(szBuf, "%4ld %11.11s    %4ld    %-56.56s", lTagNumber+1, szTagTime, dwEpisode, ps);
      // NOTE: the above must NOT expand out to more than ABF_MAXTAGFORMATLEN
   }
   else
      // "Tag #   Time (s)   Comment"
      sprintf(szBuf, "%4ld %11.11s     %-56.56s", lTagNumber+1, szTagTime, ps);

   strncpy(pszBuffer, szBuf, uSize-1);
   pszBuffer[uSize-1] = '\0';   
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_WriteDelta
// PURPOSE:  This function buffers tags to a temporary file through the CABFItem object in the
//           file descriptor.
//
BOOL WINAPI ABF_WriteDelta(int nFile, ABFFileHeader *pFH, const ABFDelta *pDelta, int *pnError)
{
   ABFH_WASSERT(pFH);
   WPTRASSERT(pDelta);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->PutDelta(pDelta))
      ERRORRETURN(pnError, pFI->GetLastError());
      
   pFH->lNumDeltas = pFI->GetDeltaCount();
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_ReadDeltas
// PURPOSE:  This function reads a Delta array from the DeltaArray section
//
BOOL WINAPI ABF_ReadDeltas(int nFile, const ABFFileHeader *pFH, DWORD dwFirstDelta, 
                           ABFDelta *pDeltaArray, UINT uNumDeltas, int *pnError)
{
   ABFH_ASSERT(pFH);
   ARRAYASSERT(pDeltaArray, uNumDeltas);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   // If this file is being written, the Deltas will be in the virtual Delta buffer.
   if (pFI->GetDeltaCount() > 0)
   {
      if (!pFI->ReadDeltas(dwFirstDelta, pDeltaArray, uNumDeltas))
         ERRORRETURN(pnError, ABF_EREADDELTA);
      return TRUE;
   }

   // If there are no Deltas present, return an error.   
   if ((pFH->lDeltaArrayPtr==0) || (pFH->lNumDeltas==0))
      ERRORRETURN(pnError, ABF_ENODELTAS);
      
   if (dwFirstDelta+uNumDeltas > UINT(pFH->lNumDeltas))
      ERRORRETURN(pnError, ABF_EREADDELTA);

   // Seek to the start of the requested segment.
   LONGLONG llSeekPos = LONGLONG(pFH->lDeltaArrayPtr) * ABF_BLOCKSIZE + dwFirstDelta * sizeof(ABFDelta);
   VERIFY(pFI->Seek(llSeekPos, FILE_BEGIN));

   // Read the Delta Array directly into the passed buffer
   UINT uBytesToRead = uNumDeltas * sizeof(ABFDelta);
   if (!pFI->Read(pDeltaArray, uBytesToRead))
      ERRORRETURN(pnError, ABF_EREADTAG);
   
   return TRUE;
}

//===============================================================================================
// FUNCTION: FormatAsBinary
// PURPOSE:  Formats a .
//
static int FormatAsBinary(UINT uValue, LPSTR pszBuffer, UINT uBufferLength)
{
   UINT uNumBits = 8;
   if (uNumBits >= uBufferLength)
      uNumBits = uBufferLength - 1;
   for (UINT i=0; i<uNumBits; i++)
      pszBuffer[i] = (uValue & (1<<i)) ? '1' : '0';
   pszBuffer[uNumBits] = '\0';
   strrev(pszBuffer);
   return uNumBits;
}

//===============================================================================================
// FUNCTION: ABF_FormatDelta
// PURPOSE:  This function builds an ASCII string to describe a delta.
//
BOOL WINAPI ABF_FormatDelta(const ABFFileHeader *pFH, const ABFDelta *pDelta, char *pszText, UINT uTextLen, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pDelta);
   ARRAYASSERT(pszText, uTextLen);

   // Format a comment string to describe the delta.
   char szText[128];
   switch (pDelta->lParameterID)
   {
      case ABF_DELTA_HOLDING0:
      case ABF_DELTA_HOLDING1:
      case ABF_DELTA_HOLDING2:
      case ABF_DELTA_HOLDING3:
      {
         UINT uDAC = pDelta->lParameterID - ABF_DELTA_HOLDING0;
         char szSignal[ABF_DACNAMELEN+1] = { '#', char(uDAC+'0'), '\0' };
         char szUnits[ABF_DACUNITLEN+2] = { ' ', '\0' };

         ABF_GET_STRING(szSignal, pFH->sDACChannelName[uDAC], sizeof(szSignal));
         ABF_GET_STRING(szUnits+1, pFH->sDACChannelUnits[uDAC], sizeof(szUnits)-1);

         _snprintf(szText, sizeof(szText), 
                   "Holding on '%s' => %g", szSignal, pDelta->fNewParamValue);
         if (szUnits[1] != '\0')
            strcat(szText, szUnits);
         break;
      }
      case ABF_DELTA_DIGITALOUTS:
      {
         char szBuffer[9];
         FormatAsBinary(UINT(pDelta->lNewParamValue), szBuffer, sizeof(szBuffer));
         _snprintf(szText, sizeof(szText), 
                   "Digital Outputs => %s", szBuffer);
         break;
      }
      case ABF_DELTA_THRESHOLD:
         _snprintf(szText, sizeof(szText), 
                   "Threshold => %g", pDelta->fNewParamValue);
         break;
      case ABF_DELTA_PRETRIGGER:
         _snprintf(szText, sizeof(szText), 
                   "Pre-trigger => %d", (int)( pDelta->lNewParamValue / pFH->nADCNumChannels ) );
         break;
      default:
         if ((pDelta->lParameterID >= ABF_DELTA_AUTOSAMPLE_GAIN) && 
             (pDelta->lParameterID < ABF_DELTA_AUTOSAMPLE_GAIN+ABF_ADCCOUNT))
         {
            _snprintf(szText, sizeof(szText), 
                      "Autosample gain => %g", pDelta->fNewParamValue);
            break;
         }
         ERRORMSG1("ABFDelta: Unexpected parameter ID '%d'.", pDelta->lParameterID);
         ERRORRETURN(pnError, ABF_EBADDELTAID);
   }
   strncpy(pszText, szText, uTextLen-1);
   pszText[uTextLen-1] = '\0';
   return TRUE;
}


//===============================================================================================
// FUNCTION: ABF_EpisodeFromSynchCount
// PURPOSE:  This routine returns the episode number for the synch count that is
//           passed as an argument.
// INPUT:
//   nFile           the file index into the g_FileData structure array
//   pdwSynchCount   the synch count to search for.
// 
// OUTPUT:
//   pdwEpisode      the episode number which contains the requested sample
//   pdwSynchCount   the synch count of the start of the episode
// 
BOOL WINAPI ABF_EpisodeFromSynchCount(int nFile, const ABFFileHeader *pFH, DWORD *pdwSynchCount, 
                                      DWORD *pdwEpisode, int *pnError)
{
   ABFH_ASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   UINT uAcquiredEpisodes = pFI->GetAcquiredEpisodes();

   // For data that is continuous in time or for a Waveform data file, just
   // calculate the episode number by dividing the synch count by the episode
   // size in samples.

   if (pFI->GetSynchCount() == 0)    // (old ABF_WAVEFORMFILE or ABF_GAPFREEFILE)
   {
      UINT uEpiSize = UINT(pFH->lNumSamplesPerEpisode);
      UINT uEpisode = *pdwSynchCount / uEpiSize + 1;
      if (uEpisode > uAcquiredEpisodes)
         uEpisode = uAcquiredEpisodes;
      *pdwSynchCount = uEpiSize * (uEpisode - 1);
      *pdwEpisode    = uEpisode;
      return TRUE;
   }

   // Search the data file for the target sample number, taking into account
   // the missing samples between episodes.

   UINT uEpiStart = pFI->EpisodeStart(1);
   if (uEpiStart > *pdwSynchCount)
   {
      *pdwEpisode    = 1;
      *pdwSynchCount = uEpiStart;
      return TRUE;
   }

   // Do a linear search on the synch array to find the episode that corresponds
   // to this sample number. This may be changed to a binary search in the future if
   // it seems to be too slow on really big data files.
   UINT uCounter = uEpiStart;
   for (UINT i=2; i <= uAcquiredEpisodes; i++)
   {
      uEpiStart = pFI->EpisodeStart(i);
      if (uEpiStart > *pdwSynchCount)
      {
         *pdwEpisode    = i - 1;
         *pdwSynchCount = uCounter;
         return TRUE;
      }
      uCounter = uEpiStart;
   }

   // Return the results.
   *pdwEpisode    = uAcquiredEpisodes;
   *pdwSynchCount = uCounter;
   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: ABF_SynchCountFromEpisode
// PURPOSE:  This routine returns the synch count for the start of the given
//           episode number that is passed as an argument.
// INPUT:
//   nFile           the file index into the g_FileData structure array
//   pdwEpisode      the episode number which is being searched for
// 
// OUTPUT:
//   pdwSynchCount   the synch count of the start of the episode
// 
BOOL WINAPI ABF_SynchCountFromEpisode(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                                      DWORD *pdwSynchCount, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdwSynchCount);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // For data that is continuous in time or is a Waveform data file then just
   // calculate the synch count by multiplying the episode number by the
   // episode size in samples.

   if (pFI->GetSynchCount() != 0)
      *pdwSynchCount = pFI->EpisodeStart(dwEpisode);
   else if (pFH->nOperationMode != ABF_WAVEFORMFILE)
      *pdwSynchCount = UINT(pFH->lNumSamplesPerEpisode) * (dwEpisode - 1);
   else
   {
      // (old ABF_WAVEFORMFILE)
      double dStartToStartUS = 0.0;
      ABFH_GetEpisodeStartToStart(pFH, &dStartToStartUS);

      *pdwSynchCount= ABFH_MSToSynchCount(pFH, dStartToStartUS/1E3 * (dwEpisode-1));
   }
   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: ABF_GetEpisodeFileOffset
// PURPOSE:  This routine returns the sample point offset in the ABF file for the start of the given
//           episode number that is passed as an argument.
// INPUT:
//   nFile           the file index into the g_FileData structure array
//   pdwEpisode      the episode number which is being searched for
// 
// OUTPUT:
//   plFileOffset the Sample point number of the first point in the episode (per channel).
// 
BOOL WINAPI ABF_GetEpisodeFileOffset(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                                     DWORD *pdwFileOffset, int *pnError)
{
   ABFH_ASSERT(pFH);
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // For data that is continuous in time or is a Waveform data file then just
   // calculate the sample number by multiplying the episode number by the
   // episode size in samples.

   if (pFI->GetSynchCount() == 0)          // (ABF_WAVEFORMFILE or ABF_GAPFREEFILE)
   {
      UINT uEpiSize = (UINT)(pFH->lNumSamplesPerEpisode / pFH->nADCNumChannels);
      *pdwFileOffset = uEpiSize * (dwEpisode - 1);
   }
   else
      *pdwFileOffset = pFI->FileOffset(dwEpisode) / pFH->nADCNumChannels / SampleSize(pFH);
   return TRUE;
}


//===============================================================================================
// FUNCTION: ABF_GetMissingSynchCount
// PURPOSE:  This routine returns the number of samples missing for event detected data for
//           the episode number passed as an argument.
// INPUT:
//   nFile                 the file index into the g_FileData structure array
//   dwEpisode             the episode number of interest
// 
// OUTPUT:
//   pdwMissingSynchCount  the number of synch counts absent prior to this episode
//
BOOL WINAPI ABF_GetMissingSynchCount(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                                     DWORD *pdwMissingSynchCount, int *pnError)
{
   ABFH_ASSERT(pFH);
   DWORD dwSynchCount = 0;
   if (!ABF_SynchCountFromEpisode(nFile, pFH, dwEpisode, &dwSynchCount, pnError))
      return FALSE;

   UINT uMissing = 0;
   if (dwEpisode == 1)
      uMissing = dwSynchCount;
   else
   {
      ASSERT(dwEpisode > 1);

      DWORD dwLastSynchCount = 0;
      if (!ABF_SynchCountFromEpisode(nFile, pFH, dwEpisode-1, &dwLastSynchCount, pnError))
         return FALSE;

      // Get the duration in ms.
      double dDurationMS = 0.0;
      if (!ABF_GetEpisodeDuration(nFile, pFH, dwEpisode-1, &dDurationMS, pnError))
         return FALSE;

      // Convert the duration to synch count.
      dwLastSynchCount += ABFH_MSToSynchCount(pFH, dDurationMS);

      // Calculate the number of missing synch counts.
      if (dwLastSynchCount > dwSynchCount)
         uMissing = 0;
      else
         uMissing = dwSynchCount - dwLastSynchCount;
   }
   *pdwMissingSynchCount = uMissing;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_HasOverlappedData
// PURPOSE:  Returns true if the file contains overlapped data.
//
BOOL WINAPI ABF_HasOverlappedData(int nFile, BOOL *pbHasOverlapped, int *pnError)
{
   WPTRASSERT(pbHasOverlapped);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   // Return an error if writing is inappropriate.
   if (!pFI->TestFlag(FI_READONLY))
      ERRORRETURN(pnError, ABF_EWRITEONLYFILE);   

   *pbHasOverlapped = pFI->GetOverlappedFlag();
   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: ABF_GetNumSamples
// PURPOSE:  This routine returns the number of samples per channel in a given episode.
// INPUT:
//   nFile          the file index into the g_FileData structure array
//   dwEpisode      the episode number of interest
// 
// OUTPUT:
//   NumSamples%    the number of data points in this episode
// 
BOOL WINAPI ABF_GetNumSamples(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                              UINT *puNumSamples, int *pnError)
{
//   ABFH_ASSERT(pFH);
   UINT uRealSize;
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->CheckEpisodeNumber(dwEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   if (pFI->GetSynchCount() == 0) /// (ABF_WAVEFORMFILE or ABF_GAPFREEFILE)
   {
      if ((pFH->nOperationMode == ABF_GAPFREEFILE) && (dwEpisode == pFI->GetAcquiredEpisodes()))
         uRealSize = pFI->GetLastEpiSize();
      else
         uRealSize = UINT(pFH->lNumSamplesPerEpisode);
   }
   else
      uRealSize = (UINT)pFI->EpisodeLength(dwEpisode);
   *puNumSamples = uRealSize / pFH->nADCNumChannels;
   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: ABF_GetEpisodeDuration
// PURPOSE:  Get the duration of a given episode in ms.
//
BOOL WINAPI ABF_GetEpisodeDuration(int nFile, const ABFFileHeader *pFH, DWORD dwEpisode, 
                                   double *pdDuration, int *pnError)
{
   ABFH_ASSERT(pFH);
   ASSERT(dwEpisode >0);
   WPTRASSERT(pdDuration);

   *pdDuration = 0.0;
   double dDurationUS = 0.0;

   if (pFH->nOperationMode == ABF_WAVEFORMFILE)
      ABFH_GetEpisodeDuration(pFH, &dDurationUS);
   else
   {
      UINT uNumSamples;
      if (!ABF_GetNumSamples(nFile, pFH, dwEpisode, &uNumSamples, pnError))
         return FALSE;
      
      // Calculate the duration in us.
      dDurationUS = ABFH_GetFirstSampleInterval(pFH) * uNumSamples * pFH->nADCNumChannels;
   }
   *pdDuration = dDurationUS / 1E3;  // Convert from us to ms.
   ASSERT(*pdDuration != 0.0);
   return TRUE;   
}                                          

//===============================================================================================
// FUNCTION: ABF_GetTrialDuration
// PURPOSE:  Calculate the trial duration in ms.
//           This is the duration between the start of the file and the last sample in the file.
//
BOOL WINAPI ABF_GetTrialDuration(int nFile, const ABFFileHeader *pFH, double *pdDuration, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdDuration);

   *pdDuration = 0.0;

   // Get the start time of the last sweep.
   double dLastSweepStart = 0;
   if( !ABF_GetStartTime( nFile, pFH, pFH->nADCSamplingSeq[0], pFH->lActualEpisodes, &dLastSweepStart, pnError ) )
      return FALSE;

   // Now the duration of the last sweep.
   double dLastSweepDuration = 0;
   if( !ABF_GetEpisodeDuration( nFile, pFH, pFH->lActualEpisodes, &dLastSweepDuration, pnError ) )
      return FALSE;

   double dTotalDuration = dLastSweepStart + dLastSweepDuration;
   ASSERT( dTotalDuration > 0 );
   *pdDuration = dTotalDuration;
   
   return TRUE;   
}
*/
//===============================================================================================
// FUNCTION: ABF_GetStartTime
// PURPOSE:  Get the start time for the first sample of the given episode in ms.
//
BOOL WINAPI ABF_GetStartTime(int nFile, const ABFFileHeader *pFH, int nChannel, DWORD dwEpisode, 
                             double *pdStartTime, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdStartTime);
   ASSERT(dwEpisode > 0);
   
   DWORD dwSynchCount = 0;
   if (!ABF_SynchCountFromEpisode(nFile, pFH, dwEpisode, &dwSynchCount, pnError))
      return FALSE;
   
   // test for the average sweep
   if ( dwSynchCount == ABF_AVERAGESWEEPSTART )
   {
      *pdStartTime = 0.0;
      return TRUE;
   }

   ABFH_SynchCountToMS(pFH, dwSynchCount, pdStartTime);

   // Get the offset into the multiplexed data array for the first point
   UINT uChannelOffset;
   if (!ABFH_GetChannelOffset(pFH, nChannel, &uChannelOffset))
      ERRORRETURN(pnError, ABF_EINVALIDCHANNEL);

   *pdStartTime += uChannelOffset * ABFH_GetFirstSampleInterval(pFH) / 1E3;
   return TRUE;
}
/*
//###############################################################################################
//###
//###   Functions to read and write scope configuration data.
//###
//###############################################################################################

//===============================================================================================
// FUNCTION: _UpdateOldDisplayEntries
// PURPOSE:  Updates the old display entries in the ABF header for backward compatability.
//
static void _UpdateOldDisplayEntries(ABFFileHeader *pFH, const ABFScopeConfig *pCfg)
{
   if ((pFH->nOperationMode == ABF_WAVEFORMFILE) || (pFH->nOperationMode == ABF_HIGHSPEEDOSC))
   {
      pFH->lStartDisplayNum = long(pCfg->fDisplayStart);
      pFH->lFinishDisplayNum= long(pCfg->fDisplayEnd);
   }
   else
      pFH->lSamplesPerTrace = long(pCfg->fDisplayEnd);
   
   for (int i=0; i<int(pFH->nADCNumChannels); i++)
   {
      float fGain = 1.0F;
      float fOffset = 0.0F;
      int nChannel = pFH->nADCSamplingSeq[i];
      
      const ABFSignal *pT = pCfg->TraceList;
      for (int j=0; j<pCfg->nTraceCount; j++, pT++)
         if ((pT->nMxOffset==i) && !pT->bFloatData)
         {
            fGain   = pT->fDisplayGain;
            fOffset = pT->fDisplayOffset;
            break;
         }
         
      pFH->fADCDisplayAmplification[nChannel] = fGain;
      pFH->fADCDisplayOffset[nChannel]        = fOffset;
   }
}

#include <stddef.h>                                        
//===============================================================================================
// FUNCTION: ABF_WriteScopeConfig
// PURPOSE:  Saves the current scope configuration info to the data file.
//
BOOL WINAPI ABF_WriteScopeConfig(int nFile, ABFFileHeader *pFH, int nScopes, 
                                 const ABFScopeConfig *pCfg, int *pnError)
{
   ABFH_WASSERT(pFH);
   if (nScopes == 0)
   {
      pFH->lNumScopes = 0;
      pFH->lScopeConfigPtr = 0;
      return TRUE;
   }
   
   BOOL bHasData = ABF_HasData(nFile, pFH);
   
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag(FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->FillToNextBlock(&pFH->lScopeConfigPtr))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   // The ABFScopeConfig has been extended for ABF file v1.68.
   // The original scope configurations defined as 'Section1' are written out to file first, 
   // to keep backwards comapatability.
   // The new configuration data, known as 'Section2' will be appended to the scope configuration data.

   UINT uSizeofVersion1  = offsetof(ABFScopeConfig,nSizeofOldStructure);
   UINT uSizeofVersion2  = sizeof(ABFScopeConfig) - uSizeofVersion1;
   UINT uSizeofWhole     = uSizeofVersion2 + uSizeofVersion1;     
   ASSERT( uSizeofWhole == sizeof(ABFScopeConfig) );

   // Prevent compiler warnings.
   uSizeofWhole = uSizeofWhole;

   // Write out section1 ABF scope configuration for backwards compatability.
   for( int i = 0; i < nScopes; i ++ )
   {
      if (!pFI->Write( &pCfg[i], uSizeofVersion1 ))
      {
         pFH->lScopeConfigPtr = 0;
         ERRORRETURN(pnError, ABF_EDISKFULL);
      }
   }

   // Write the new section2 ABFScopeConfig data.
   for(int i = 0; i < nScopes; i ++ )
   {
      if (!pFI->Write( (char*)&pCfg[i] + uSizeofVersion1, uSizeofVersion2 ))
      {
         pFH->lScopeConfigPtr = 0;
         ERRORRETURN(pnError, ABF_EDISKFULL);
      }
   }

   // Update the number of scopes in the header.
   pFH->lNumScopes = nScopes;

   if (!bHasData && !pFI->FillToNextBlock(&pFH->lDataSectionPtr))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   _UpdateOldDisplayEntries(pFH, pCfg);
      
   LONGLONG llHere = 0;
   VERIFY(pFI->Seek(0, FILE_CURRENT, &llHere));
   
   // Update the header on disk.
   VERIFY(pFI->Seek( 0, FILE_BEGIN));
   UINT uBytesToWrite = sizeof(ABFFileHeader);   
   if (!pFI->Write( pFH, uBytesToWrite ))
   {
      pFH->lScopeConfigPtr = 0;
      ERRORRETURN(pnError, ABF_EDISKFULL);
   }

   VERIFY(pFI->Seek(llHere, FILE_BEGIN));
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_ReadScopeConfig
// PURPOSE:  Retrieves the scope configuration info from the data file.
//
BOOL WINAPI ABF_ReadScopeConfig(int nFile, ABFFileHeader *pFH, ABFScopeConfig *pCfg, 
                                UINT uMaxScopes, int *pnError)
{
   ABFH_WASSERT(pFH);
   ARRAYASSERT(pCfg, (UINT)(pFH->lNumScopes));
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   if ((pFH->lNumScopes < 1) || (pFH->lScopeConfigPtr == 0))
   {
      pFH->lNumScopes = 1;
      ABFH_InitializeScopeConfig(pFH, pCfg);
      return TRUE;
   }

   UINT uOffset = pFH->lScopeConfigPtr * ABF_BLOCKSIZE;
   VERIFY(pFI->Seek( uOffset, FILE_BEGIN));
   
   UINT uScopes = (uMaxScopes < UINT(pFH->lNumScopes)) ? uMaxScopes : UINT(pFH->lNumScopes);

   // The ABFScopeConfig has been extended for ABF file v1.68.
   // The original scope configurations defined as 'Section1' are read in first, 
   // to keep backwards compatability.
   // The new configuration data, known as 'Section2' is appended after the scope configuration data
   // and is read last, and only for the new files that support it.

   UINT uSizeofSection1  = offsetof(ABFScopeConfig,nSizeofOldStructure);
   UINT uSizeofSection2  = sizeof(ABFScopeConfig) - uSizeofSection1;
   UINT uSizeofWhole     = uSizeofSection2 + uSizeofSection1;
   ASSERT( uSizeofWhole == sizeof(ABFScopeConfig) );

   // Prevent compiler warnings.
   uSizeofWhole = uSizeofWhole;

   // Read old section of the scope config structure
   for( int i = 0; i < pFH->lNumScopes; i ++ )
   {
      if (!pFI->Read( &pCfg[i], uSizeofSection1))
         ERRORRETURN(pnError, ABF_EREADSCOPECONFIG);
   }

   // Read the new section ABFScopeConfig structures into the buffer
   if( pFH->fHeaderVersionNumber >= 1.68F )
   {
      for(int i = 0; i < pFH->lNumScopes; i++ )
      {
         if (!pFI->Read( (char*)&pCfg[i] + uSizeofSection1, uSizeofSection2))
            ERRORRETURN(pnError, ABF_EREADSCOPECONFIG);        
      }
   }
      
   pFH->lNumScopes = uScopes;

   if (pFH->fFileVersionNumber < 1.5)
   {
      for (UINT i=0; i<uScopes; i++)
         OLDH_CorrectScopeConfig(pFH, pCfg+i);

      // Changes for V1.5:
      // Change ABFSignal parameters from UUTop & UUBottom to
      // fDisplayGain & fDisplayOffset.
   }
   return TRUE;
}
                                        

//===============================================================================================
// FUNCTION: ABF_WriteStatisticsConfig
// PURPOSE:  Write the scope config structure for the statistics window out to the ABF file.
//
BOOL WINAPI ABF_WriteStatisticsConfig( int nFile, ABFFileHeader *pFH, 
                                       const ABFScopeConfig *pCfg, int *pnError)
{
   ABFH_WASSERT(pFH);
   RPTRASSERT(pCfg);
   
   BOOL bHasData = ABF_HasData(nFile, pFH);
   
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag(FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->FillToNextBlock(&pFH->lStatisticsConfigPtr))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   if (!pFI->Write( pCfg, sizeof(ABFScopeConfig)))
   {
      pFH->lStatisticsConfigPtr = 0;
      ERRORRETURN(pnError, ABF_EDISKFULL);
   }

   if (!bHasData && !pFI->FillToNextBlock(&pFH->lDataSectionPtr))
      ERRORRETURN(pnError, ABF_EDISKFULL);

   LONGLONG llHere = 0;
   VERIFY(pFI->Seek(0, FILE_CURRENT, &llHere));
   
   // Update the header on disk.
   VERIFY(pFI->Seek( 0, FILE_BEGIN));
   if (!pFI->Write( pFH, sizeof(ABFFileHeader) ))
   {
      pFH->lStatisticsConfigPtr = 0;
      ERRORRETURN(pnError, ABF_EDISKFULL);
   }

   VERIFY(pFI->Seek(llHere, FILE_BEGIN));
   return TRUE;
}
                                        
//===============================================================================================
// FUNCTION: ABF_ReadStatisticsConfig
// PURPOSE:  Read the scope config structure for the statistics window in form the ABF file.
//
BOOL WINAPI ABF_ReadStatisticsConfig( int nFile, const ABFFileHeader *pFH, ABFScopeConfig *pCfg, 
                                      int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pCfg);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   if (pFH->lStatisticsConfigPtr == 0)
      ERRORRETURN(pnError, ABF_ENOSTATISTICSCONFIG);

   UINT uOffset = pFH->lStatisticsConfigPtr * ABF_BLOCKSIZE;
   VERIFY(pFI->Seek( uOffset, FILE_BEGIN));
   
   // The ABFScopeConfig structure has been extended for ABF file version 1.68.
   // If the file is a new file, reading is unaffected as only one structure is saved for the statistics config.
   // If the file is an older file, only the size of section1 of the scope config is read to avoid reading junk data.
   if( pFH->fFileVersionNumber >= 1.68F )
   {
      if (!pFI->Read(pCfg, sizeof(ABFScopeConfig)))
         ERRORRETURN(pnError, ABF_EREADSTATISTICSCONFIG);
   }  
   else
   {
      UINT uSizeofSection1  = offsetof(ABFScopeConfig,nSizeofOldStructure);
      // Read only size of version 1.
      if ( !pFI->Read(pCfg, uSizeofSection1 ))
         ERRORRETURN(pnError, ABF_EREADSTATISTICSCONFIG);
   }
   return TRUE;
}

//###############################################################################################
//###
//###   Functions to read and write voice tags.
//###
//###############################################################################################

//===============================================================================================
// FUNCTION: ABF_SaveVoiceTag
// PURPOSE:  Saves a reference to a temporary file containing a voice tag.
//
BOOL WINAPI ABF_SaveVoiceTag( int nFile, LPCSTR pszFileName, long lDataOffset,
                              ABFVoiceTagInfo *pVTI, int *pnError)
{
   LPSZASSERT(pszFileName);
   WPTRASSERT(pVTI);
   
   // Get the file descriptor for this ABF file.
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   if (!pFI->SaveVoiceTag( pszFileName, lDataOffset, pVTI ))
      ERRORRETURN( pnError, pFI->GetLastError() );
      
   return TRUE;
}
                              
                              
//===============================================================================================
// FUNCTION: ABF_GetVoiceTag
// PURPOSE:  Retrieves a voice tag into a new file, leaving space for a header.
//
BOOL WINAPI ABF_GetVoiceTag( int nFile, const ABFFileHeader *pFH, UINT uTag, LPCSTR pszFileName, 
                             long lDataOffset, ABFVoiceTagInfo *pVTI, int *pnError)
{
   LPSZASSERT(pszFileName);
   WPTRASSERT(pVTI);

   // Get the file descriptor for this ABF file.
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   if (long(uTag) >= pFH->lVoiceTagEntries)
      ERRORRETURN( pnError, ABF_EREADTAG );

   if (!pFI->GetVoiceTag( uTag, pszFileName, lDataOffset, pVTI, pFH->lVoiceTagPtr ))
      ERRORRETURN( pnError, pFI->GetLastError() );
      
   return TRUE;
}                                    
*/                              
//===============================================================================================
// FUNCTION: ABF_BuildErrorText
// PURPOSE:  This routine returns the last error as a text string.
//
BOOL WINAPI ABF_BuildErrorText(int nErrorNum, const char *szFileName, char *sTxtBuf, UINT uMaxLen)
{
//   LPSZASSERT(szFileName);
//   ARRAYASSERT(sTxtBuf, uMaxLen);
   
   if (uMaxLen < 2)
   {
//      ERRORMSG("String too short!");
      return FALSE;
   }

   if (nErrorNum > ABFH_FIRSTERRORNUMBER)
      return ABFH_GetErrorText( nErrorNum, sTxtBuf, uMaxLen);

   BOOL rval = TRUE;        // OK return value
   char szTemplate[128];
   if (!c_LoadString(g_hInstance, nErrorNum, szTemplate, sizeof(szTemplate)))
   {
      char szErrorMsg[128];
	  c_LoadString(g_hInstance, IDS_ENOMESSAGESTR, szTemplate, sizeof(szTemplate));
	  sprintf(szErrorMsg, szTemplate, nErrorNum);
 //     ERRORMSG(szErrorMsg);

      strncpy(sTxtBuf, szErrorMsg, uMaxLen-1);
      sTxtBuf[uMaxLen-1] = '\0';
      rval = FALSE;
   }
   else
#ifdef _WINDOWS
      _snprintf(sTxtBuf, uMaxLen, szTemplate, szFileName);
#else
      snprintf(sTxtBuf, uMaxLen, szTemplate, szFileName);
#endif	
   return rval;
}
/*
//===============================================================================================
// FUNCTION: ABF_SetErrorCallback
// PURPOSE:  This routine sets a callback function to be called in the event of an error occuring.
//
BOOL WINAPI ABF_SetErrorCallback(int nFile, ABFCallback fnCallback, void *pvThisPointer, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   return pFI->SetErrorCallback(fnCallback, pvThisPointer);
}
                                 
// ***********************************************************************************************
// ***********************************************************************************************
// ***
// ***    ABF_GetSynchArray exposes an internal synch object to enable direct access to the
// ***    synch array by real-time data acquisition components.
// ***
// ***********************************************************************************************
// ***********************************************************************************************

//===============================================================================================
// FUNCTION: ABF_GetFileHandle
// PURPOSE:  Returns the DOS file handle for the ABF file.
//
BOOL WINAPI ABF_GetFileHandle(int nFile, HANDLE *phHandle, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   *phHandle = pFI->GetFileHandle();
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_GetSynchArray
// PURPOSE:  Returns a pointer to the CSynch object used to buffer the Synch array to disk.
//           Use with care!!
//
void *WINAPI ABF_GetSynchArray(int nFile, int *pnError)
{
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return NULL;
   
   return pFI->GetSynchObject();
}


// ***********************************************************************************************
// ***********************************************************************************************
// ***
// ***    Functions used to implement "modifiable ABF".
// ***
// ***********************************************************************************************
// ***********************************************************************************************

//===============================================================================================
// FUNCTION: ABF_UpdateEpisodeSamples
// PURPOSE:  This function updates a selection of samples in a particular episode.
// NOTES:    Only floating point data may be modified with this function -- integer data is sacrosanct.
//           Math channels may not be written to.
//           uStartSample is zero-based
//           uNumSamples os on-based.
//
BOOL WINAPI ABF_UpdateEpisodeSamples(int nFile, const ABFFileHeader *pFH, int nChannel, UINT uEpisode, 
                                     UINT uStartSample, UINT uNumSamples, float *pfBuffer, int *pnError)
{
   ABFH_ASSERT(pFH);
   ASSERT( uNumSamples > 0 );

   ARRAYASSERT(pfBuffer, uNumSamples);

   UINT uPerChannel = (UINT)(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels);
   ASSERT(uStartSample+uNumSamples <= uPerChannel);
   
   // Prevent compiler warnings.
   uPerChannel = uPerChannel;

   // Writing is not allowed for two-byte integer files.
   ASSERT(pFH->nDataFormat != ABF_INTEGERDATA);
   if (pFH->nDataFormat == ABF_INTEGERDATA)
      ERRORRETURN(pnError, ABF_EWRITERAWDATAFILE);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // Check that the episode number is in range.
   if (!pFI->CheckEpisodeNumber(uEpisode))
      ERRORRETURN(pnError, ABF_EEPISODERANGE);

   // Cannot write to a math channel.
   if (nChannel < 0)
      ERRORRETURN(pnError, ABF_EWRITEMATHCHANNEL);

   // Get the offset into the multiplexed data array for the first point
   UINT uChannelOffset;
   if (!ABFH_GetChannelOffset(pFH, nChannel, &uChannelOffset))
      ERRORRETURN(pnError, ABF_EINVALIDCHANNEL);

   // Set the sample size in the data.
   UINT uSampleSize = SampleSize(pFH);

   // Only create the read buffer on demand, it is freed when the file is closed.
   if (!pFI->GetReadBuffer())
   {      
      if (!pFI->AllocReadBuffer(pFH->lNumSamplesPerEpisode * uSampleSize))
         ERRORRETURN(pnError, ABF_OUTOFMEMORY);
   }

   // Read the whole episode from the ABF file only if it is not already cached.
   if (uEpisode != pFI->GetCachedEpisode())
   {         
      UINT uEpisodeSize = (UINT)pFH->lNumSamplesPerEpisode;
      if (!ABF_MultiplexRead(nFile, pFH, uEpisode, pFI->GetReadBuffer(), &uEpisodeSize, pnError))
      {
         pFI->SetCachedEpisode(UINT(-1), 0);
         return FALSE;
      }
      pFI->SetCachedEpisode(uEpisode, uEpisodeSize);
   }

   // Update the samples in the episode cache.
   UINT   uEpisodeOffset = uStartSample * pFH->nADCNumChannels;
   float *pfEpisodeBuffer = (float *)pFI->GetReadBuffer() + uEpisodeOffset;
   float *pfData = pfEpisodeBuffer + uChannelOffset;
   for (UINT i=0; i<uNumSamples; i++)
   {
      *pfData = pfBuffer[i];
      pfData += pFH->nADCNumChannels;
   }

   // Commit the change to file.
   BOOL bReadOnly = pFI->TestFlag(FI_READONLY);
   if (bReadOnly)
      VERIFY(pFI->Reopen(FALSE));

   Synch SynchEntry = { 0 };
   VERIFY(GetSynchEntry( pFH, pFI, uEpisode, &SynchEntry ));
   UINT uOffset = GetDataOffset(pFH) + SynchEntry.dwFileOffset + uEpisodeOffset * sizeof(float);
   pFI->Seek(uOffset, FILE_BEGIN);
   pFI->Write(pfEpisodeBuffer, uNumSamples*pFH->nADCNumChannels*sizeof(float));
   
   if (bReadOnly)
      VERIFY(pFI->Reopen(TRUE));

   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_SetChunkSize
// PURPOSE:  This routine can be called on files of type ABF_GAPFREEFILE or ABF_VARLENEVENTS to change
//           the size of the data chunks returned by the read routines.
// INPUT:
//   hFile          ABF file number of this file (NOT the DOS handle)
//   pFH            the current acquisition parameters for the data file
//   puMaxSamples   points to the requested size of data blocks to be returned.
//                  This is only used in the case of GAPFREE and EVENT-DETECTED-
//                  VARIABLE-LENGTH acquisitions. Otherwise the size of the
//                  Episode is used. 80x86 limitations require this to be 
//                  less than or equal to 64k.
//   pdwMaxEpi      The maximum number of episodes to be read.
// OUTPUT:
//   pFH            the acquisition parameters that were read from the data file
//   puMaxSamples   the maximum number of samples that can be read contiguously
//                  from the data file.
//   pdwMaxEpi      the number of episodes of puMaxSamples points that exist
//                  in the data file.
// 
BOOL WINAPI ABF_SetChunkSize( int nFile, ABFFileHeader *pFH, UINT *puMaxSamples, 
                              DWORD *pdwMaxEpi, int *pnError )
{
   ASSERT(nFile != ABF_INVALID_HANDLE);

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   return _SetChunkSize( pFI, pFH, puMaxSamples, pdwMaxEpi, pnError );
}

//===============================================================================================
// FUNCTION: ABF_SetOverlap
// PURPOSE:  Changes the overlap flag and processes the synch array to edit redundant data out if no overlap.
//
BOOL WINAPI ABF_SetOverlap(int nFile, const ABFFileHeader *pFH, BOOL bAllowOverlap, int *pnError)
{
   ASSERT(nFile != ABF_INVALID_HANDLE);
   ABFH_ASSERT(pFH);

   // Get the file descriptor.
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   return _SetOverlap(pFI, pFH, bAllowOverlap, pnError);
}

// ***********************************************************************************************
// ***********************************************************************************************
// ***
// ***    Functions used to read and write annotations.
// ***
// ***********************************************************************************************
// ***********************************************************************************************

//===============================================================================================
// FUNCTION: ABF_WriteAnnotation
// PURPOSE:  Write an annotation to the Annotations Section of the ABF file.
//
BOOL WINAPI ABF_WriteAnnotation( int nFile, ABFFileHeader *pFH, LPCSTR pszText, int *pnError )
{
   ASSERT(nFile != ABF_INVALID_HANDLE);
   ABFH_ASSERT(pFH);
   
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
      
   // Return an error if writing is inappropriate.
   if (pFI->TestFlag( FI_PARAMFILE | FI_READONLY))
      ERRORRETURN(pnError, ABF_EREADONLYFILE);   

   if (!pFI->PutAnnotation( pszText))
      ERRORRETURN(pnError, pFI->GetLastError());
      
   NewFH.lNumAnnotations = pFI->GetAnnotationCount();

   ABFH_DemoteHeader( pFH, &NewFH );

   return TRUE;
}

BOOL WINAPI ABF_WriteStringAnnotation( int nFile, ABFFileHeader *pFH, LPCSTR pszName, LPCSTR pszData, int *pnError )
{
   LPSZASSERT(pszName);
   LPSZASSERT(pszData);
   const char c_pszTag[] = "<s,1>";
   CArrayPtrEx<char> Ann(strlen(pszName)+strlen(c_pszTag)+strlen(pszData)+1);
   if (!Ann)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   AXU_strncpyz(Ann, pszName, Ann.GetCount());
   AXU_strncatz(Ann, c_pszTag, Ann.GetCount());
   AXU_strncatz(Ann, pszData, Ann.GetCount());
   return ABF_WriteAnnotation( nFile, pFH, Ann, pnError );
}

BOOL WINAPI ABF_WriteIntegerAnnotation( int nFile, ABFFileHeader *pFH, LPCSTR pszName, int nData, int *pnError )
{
   LPSZASSERT(pszName);
   const char c_pszTag[] = "<i,1>";
   CArrayPtrEx<char> Ann(strlen(pszName)+strlen(c_pszTag)+32+1);
   if (!Ann)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   AXU_strncpyz(Ann, pszName, Ann.GetCount());
   AXU_strncatz(Ann, c_pszTag, Ann.GetCount());
   itoa(nData, Ann+strlen(Ann), 10);
   return ABF_WriteAnnotation( nFile, pFH, Ann, pnError );
}

//===============================================================================================
// FUNCTION: ABF_ReadAnnotation
// PURPOSE:  Read an annotation to the Annotations Section of the ABF file..
//
BOOL WINAPI ABF_ReadAnnotation( int nFile, const ABFFileHeader *pFH, DWORD dwIndex, 
                                LPSTR pszText, DWORD dwBufSize, int *pnError )
{
   ASSERT( nFile != ABF_INVALID_HANDLE );
   ABFH_ASSERT( pFH );   
   ARRAYASSERT( pszText, dwBufSize);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   // If there are no annotations present, return an error.   
   if( NewFH.lNumAnnotations==0 )
      ERRORRETURN(pnError, ABF_ENOANNOTATIONS);
      
   // If there are annotations in the file, but not in the virtual buffer, read them now.
   if( (NewFH.lAnnotationSectionPtr>0) && (pFI->GetAnnotationCount() == 0) )
   {
      if( !pFI->ReadAllAnnotations( NewFH.lAnnotationSectionPtr ) )
         ERRORRETURN(pnError, ABF_EREADANNOTATION);
   }

   if( !pFI->ReadAnnotation( dwIndex, pszText, dwBufSize ) )
      ERRORRETURN(pnError, ABF_EREADANNOTATION);

   return TRUE;
}

//===============================================================================================
// FUNCTION: ABF_ParseStringAnnotation
// PURPOSE:  This function parses a String annotation.
//           e.g. Name<s,1>Value
//
BOOL WINAPI ABF_ParseStringAnnotation( LPCSTR pszAnn, LPSTR pszName, UINT uSizeName, 
                                       LPSTR pszValue, UINT uSizeValue, int *pnError)
{
   LPCSTR pszStart = pszAnn;
   while (*pszStart==' ')
      ++pszStart;

   LPCSTR pszEnd = strchr(pszStart, '<');
   if (pszEnd)
   {
      AXU_strncpyz(pszName, pszStart, min(uSizeName, UINT(pszEnd-pszStart+1)));
      pszEnd = strchr(pszEnd, '>');
      if (!pszEnd)
         ERRORRETURN(pnError, ABF_EREADANNOTATION);
      pszStart = pszEnd+1;
   }
   AXU_strncpyz(pszValue, pszStart, uSizeValue);
   return true;
}

//===============================================================================================
// FUNCTION: ABF_ReadStringAnnotation
// PURPOSE:  This function reads and parses a String annotation.
//           e.g. Name<s,1>Value
//
BOOL WINAPI ABF_ReadStringAnnotation( int nFile, const ABFFileHeader *pFH, DWORD dwIndex, 
                                     LPSTR pszName, UINT uSizeName, LPSTR pszValue, UINT uSizeValue, 
                                     int *pnError )
{
   ARRAYASSERT( pszName, uSizeName);
   ARRAYASSERT( pszValue, uSizeValue);

   UINT uLen = ABF_GetMaxAnnotationSize( nFile, pFH );
   if (!uLen)
      ERRORRETURN(pnError, ABF_EREADANNOTATION);

   CArrayPtrEx<char> Ann(uLen);
   if (!Ann)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   if (!ABF_ReadAnnotation( nFile, pFH, dwIndex, Ann, Ann.GetCount(), pnError ))
      return FALSE;

   return ABF_ParseStringAnnotation( Ann, pszName, uSizeName, pszValue, uSizeValue, pnError);
}

//===============================================================================================
// FUNCTION: ABF_ReadIntegerAnnotation
// PURPOSE:  This function reads and parses an integer annotation.
//           e.g. Name<i,1>Value
//           and parses the name and value
//
BOOL WINAPI ABF_ReadIntegerAnnotation( int nFile, const ABFFileHeader *pFH, DWORD dwIndex, 
                                       LPSTR pszName, UINT uSizeName, int *pnValue, int *pnError )
{
   ARRAYASSERT( pszName, uSizeName);

   UINT uLen = ABF_GetMaxAnnotationSize( nFile, pFH );
   if (!uLen)
      ERRORRETURN(pnError, ABF_EREADANNOTATION);

   CArrayPtrEx<char> Ann(uLen);
   if (!Ann)
      ERRORRETURN(pnError, ABF_OUTOFMEMORY);

   if (!ABF_ReadAnnotation( nFile, pFH, dwIndex, Ann, Ann.GetCount(), pnError ))
      return FALSE;

   LPCSTR pszStart = AXU_StripWhiteSpace(Ann);
   LPCSTR pszEnd = strchr(pszStart, '<');
   if (pszEnd)
   {
      AXU_strncpyz(pszName, pszStart, min(uSizeName, UINT(pszEnd-pszStart+1)));
      pszEnd = strchr(pszEnd, '>');
      if (!pszEnd)
         ERRORRETURN(pnError, ABF_EREADANNOTATION);
      pszStart = pszEnd+1;
      if (pszStart[0]!='i' || pszStart[1] != ',')
         ERRORRETURN(pnError, ABF_EREADANNOTATION);
   }
   if (pnValue)
      *pnValue = atoi(pszStart);

   return true;
}

//===============================================================================================
// FUNCTION: ABF_GetMaxAnnotationSize
// PURPOSE:  Return the size in bytes of the largest annotation in the file.
//
DWORD WINAPI ABF_GetMaxAnnotationSize( int nFile, const ABFFileHeader *pFH )
{
   ASSERT(nFile != ABF_INVALID_HANDLE);
   ABFH_ASSERT( pFH );
   
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   CFileDescriptor *pFI = NULL;
   int nError = 0;
   if( !GetFileDescriptor( &pFI, nFile, &nError ) )
      return 0;
   
   // If there are annotations in the file, but not in the virtual buffer, read them now.
   if( (NewFH.lAnnotationSectionPtr>0) && (pFI->GetAnnotationCount() == 0) )
   {
      if( !pFI->ReadAllAnnotations( pFH->lAnnotationSectionPtr ) )
         ERRORRETURN( &nError, ABF_EREADANNOTATION);
   }

   // If this file is being written, the annotations will be in the virtual buffer.
   if (pFI->GetAnnotationCount() > 0)
      return pFI->GetMaxAnnotationSize();

   return 0;
}

//===============================================================================================
// FUNCTION: ABF_GetFileName
// PURPOSE:  Return the filename from a currently open file.
//
BOOL WINAPI ABF_GetFileName( int nFile, LPSTR pszFilename, UINT uTextLen, int *pnError )
{
   WARRAYASSERT( pszFilename, uTextLen );
   
   // Get the File Descriptor.   
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;
   
   AXU_strncpyz( pszFilename, pFI->GetFileName(), uTextLen );
   return TRUE;
}

//==============================================================================================
// FUNCTION: ABF_ValidateFileCRC
// PURPOSE:  Function to validate the file using the CRC embedded in the ABF header.
//           The CRC is generated at the time of writing the file and can be used to 
//           check if the file has been modified outside the application.
// RETURNS:  TRUE if CRC validation is OK.
//           FALSE if validation failed.
//
BOOL WINAPI ABF_ValidateFileCRC( int nFile, int *pnError )
{
   int nError = 0;
   ABFFileHeader NewFH;
   CFileDescriptor *pFI = NULL;

   if (!GetFileDescriptor(&pFI, nFile, pnError))
      return FALSE;

   // Read the data file parameters.
   if (!ABFH_ParamReader(pFI->GetFileHandle(), &NewFH, &nError))
   {
      nError = (nError == ABFH_EUNKNOWNFILETYPE) ? ABF_EUNKNOWNFILETYPE : ABF_EBADPARAMETERS;
      ERRORRETURN( pnError, nError );
   }
   
   // Validate checksum.
   if( !ValidateFileCRC( pFI, &NewFH, sizeof( ABFFileHeader ) ) )
   {
      nError = ABF_ECRCVALIDATIONFAILED;
      ERRORRETURN(pnError, nError);
   }

   return TRUE;
}

  
// ***********************************************************************************************
// ***********************************************************************************************
// ***
// ***    Superceded functions.
// ***
// ***********************************************************************************************
// ***********************************************************************************************

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

BOOL WINAPI ABF_UpdateAfterAcquisition(int nFile, ABFFileHeader *pFH, DWORD dwAcquiredEpisodes,
                                       DWORD dwAcquiredSamples, int *pnError);

#ifdef __cplusplus
}
#endif

//===============================================================================================
// FUNCTION: ABF_UpdateAfterAcquisition
// PURPOSE:  Update ABF internal housekeeping of acquired data, this must be called
//           before ABF_UpdateHeader if the file has been written to through the handle
//           retrieved by ABF_GetFileHandle.
//
BOOL WINAPI ABF_UpdateAfterAcquisition(int nFile, ABFFileHeader *pFH, DWORD dwAcquiredEpisodes, 
                                       DWORD dwAcquiredSamples, int *pnError)
{
   ABFH_ASSERT(pFH);
   ERRORMSG("ABF_UpdateAfterAcquisition has been retired.\n");
   return FALSE;
}


#if 0

//===============================================================================================
// FUNCTION: ABF_AppendOpen
// PURPOSE:  This routine opens an existing data file for appending.
// INPUT:
//   szFileName     the name of the data file that will be opened
//   puMaxSamples   points to the requested size of data blocks to be returned.
//                  This is only used in the case of GAPFREE and EVENT-DETECTED-
//                  VARIABLE-LENGTH acquisitions. Otherwise the size of the
//                  Episode is used. 80x86 limitations require this to be 
//                  less than or equal to 64k.
//   pdwMaxEpi      The maximum number of episodes to be read.
// OUTPUT:
//   pFH            the acquisition parameters that were read from the data file
//   phFile         pointer to the ABF file number of this file (NOT the DOS handle)
//   puMaxSamples   the maximum number of samples that can be read contiguously
//                  from the data file.
//   pdwMaxEpi      the number of episodes of puMaxSamples points that exist
//                  in the data file.
// 
BOOL WINAPI ABF_AppendOpen(LPCSTR szFileName, int *phFile, ABFFileHeader *pFH, 
                           UINT *puMaxSamples, DWORD *pdwMaxEpi, int *pnError)
{
   LPSZASSERT(szFileName);
   WPTRASSERT(phFile);
   ABFH_WASSERT(pFH);
   WPTRASSERT(puMaxSamples);
   WPTRASSERT(pdwMaxEpi);
   
   // Open the file for reading.
   int hFile = ABF_INVALID_HANDLE;
   if (!ABF_ReadOpen(szFileName, phFile, ABF_DATAFILE, pFH, puMaxSamples, pdwMaxEpi, pnError))
      return FALSE;

   // Get the File Descriptor.   
   CFileDescriptor *pFI = NULL;
   if (!GetFileDescriptor(&pFI, *phFile, pnError))
   {
      ABF_Close(*phFile, NULL);
      return FALSE;
   }
   
   // Fix up file descriptor etc...
//   if (!pFI->ChangeStatus())
//   {
//      ABF_Close(hFile, NULL);
//      ERRORRETURN( pnError, ABF_EOPENFILE );
//   }
      
   // Read the tags into a temporary file.
   if (pFH->lNumTagEntries > 0)
   {
      ERRORMSG("Transfer tags to temp file!!!");
      ABF_Close(hFile, NULL);
      return FALSE;
   }
   
   // Seek to the end of the Data section.
   UINT uOffset = GetDataOffset(pFH) + pFH->lActualAcqLength * SampleSize(pFH);
   pFI->Seek( uOffset, FILE_BEGIN);
   return TRUE;
}

#endif
*/

