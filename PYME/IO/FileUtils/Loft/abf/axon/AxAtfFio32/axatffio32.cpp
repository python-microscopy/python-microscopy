//***********************************************************************************************
//                                                                            
//   Written 1990 - 1997 by AXON Instruments Inc.                             
//                                                                            
//   This file is not protected by copyright. You are free to use, modify     
//   and copy the code in this file.                                          
//                                                                            
//***********************************************************************************************
//
// MODULE:  AXATFFIO32.CPP
// PURPOSE: Contains routines for reading and writing text files in the ATF format.
// 
// An ANSI C compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL AXATFFIO32.CPP)
//
// NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE
// For consistency and code reuse, the DLL version of this module uses
// AxonValidation.lib to perform float/double formatting.
// NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE=NOTE

#include "axatffio32.h"
#include "atfutil.h"
#include "atfintl.h"

#include "../Common/axodebug.h"

#define ATF_FILE_ID           "ATF"
#define PAF_FILE_ID           "PAF"
#define FILE_ID_LEN           3

#define PREVIOUS_PAF_VERSION  5.0F

#define ATF_AVCOLUMNSIZE      50                      // average number of chars per column
#define ATF_MINBUFSIZE        1024                    // minimum size for the pszIOBuffer

#define ATF_DBL_SIG_DIGITS    12
#define ATF_FLT_SIG_DIGITS    6

// #define ATF_DBL_STR_LEN    ATF_DBL_SIG_DIGITS + 7     // 1 for sign, 1 for decimal, 
// #define ATF_FLT_STR_LEN    ATF_FLT_SIG_DIGITS + 7     // 2 for '+E' of exponential notation, 3 for exponent
#define ATF_DBL_STR_LEN       VAL_EXTERNBUFSIZE          // for use with axovdate
#define ATF_FLT_STR_LEN       VAL_EXTERNBUFSIZE

//
// Set the maximum number of files that can be open simultaneously.
// This can be overridden from the compiler command line.
//
#define ATF_MAXFILES 64
static PATF_FILEINFO g_FileDescriptor[ATF_MAXFILES];

//
// If the file is opened in BINARY mode, CR/LF pairs are NOT translated to LFs
// on reading, and LFs are NOT translated to CR/LF on writing.
//
static const char s_szEndOfLine[]    = "\r\n";
static const char s_szWhitespace[]   = "\t\r\n ,";
static const char s_szWhiteNoSpace[] = "\t\r\n,";
static const char s_szDelimiter[]    = "\t,";
static const char s_szLineTerm[]     = "\r\n";

// HINSTANCE g_hInstance = NULL;

#define ENDOFFILE 0x1A        // End-of-file character


static BOOL ReadDataRecord(ATF_FILEINFO *pATF, int *pnError);

//===============================================================================================
// FUNCTION:   ATF_Initialize()
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
BOOL WINAPI ATF_Initialize(HINSTANCE hDLL)
{
   // Protect against multiple calls.
   if (g_hInstance != NULL)
      return TRUE;

   // Save the DLL instance handle.
   g_hInstance = hDLL;
   for (int i=0; i<ATF_MAXFILES; i++)
      g_FileDescriptor[i] = NULL;

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_Cleanup
// PURPOSE:  Cleanup function, only applicable to DOS & Windows programs, not DLLs.
//
void WINAPI ATF_Cleanup(void)
{
   // Close any files that are still open
   for (int i=0; i<ATF_MAXFILES; i++)
   {
      if (g_FileDescriptor[i] != NULL)
      {
         DLLTRACE("An ATF file was not closed!\n");
         ATF_CloseFile(i);
      }
   }
}

//===============================================================================================
// FUNCTION: GetNewFileDescriptor
// PURPOSE:  Allocate a new file descriptor and return it.
//
static BOOL GetNewFileDescriptor(ATF_FILEINFO **ppATF, int *pnFile, int *pnError)
{
   WPTRASSERT(ppATF);
   WPTRASSERT(pnFile);
   *pnFile = ATF_INVALID_HANDLE;
   
   // Find an empty slot.
   int nFile = 0;
   for (nFile=0; nFile < ATF_MAXFILES; nFile++)
      if (g_FileDescriptor[nFile] == NULL)
         break;
   
   // Return an error if no space left.   
   if (nFile == ATF_MAXFILES)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYFILES);

   // Allocate a new descriptor.
   ATF_FILEINFO *pATF = (ATF_FILEINFO *)calloc(1, sizeof(ATF_FILEINFO));
   if (pATF == NULL)
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);
   
   // Initialize any members if required.   
   pATF->szSeparator[0] = s_szDelimiter[0];
   *ppATF  = g_FileDescriptor[nFile] = pATF;
   *pnFile = nFile;
   return TRUE;
}

//-----------------------------------------------------------------------------------------------
// FUNCTION: GetFileDescriptor
// PURPOSE:  Retreive an existing file descriptor.
//
static BOOL GetFileDescriptor(ATF_FILEINFO **ppATF, int nFile, int *pnError)
{
   WPTRASSERT(ppATF);

   // Check that index is within range.
   if ((nFile < 0) || (nFile >= ATF_MAXFILES))
      ERRORRETURN(pnError, ATF_ERROR_BADFILENUM);

   // Get a pointer to the descriptor.
   ATF_FILEINFO *pATF = g_FileDescriptor[nFile];
   if (pATF == NULL)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   // Return the descriptor.
   *ppATF = pATF;
   return TRUE;
}

//===============================================================================================
// FUNCTION: AllocIOBuffer
// PURPOSE:  Allocate an IOBuffer for this file.
//
static BOOL AllocIOBuffer(ATF_FILEINFO *pATF)
{
   WPTRASSERT(pATF);
   pATF->nIOBufferSize = pATF->nColumns * ATF_AVCOLUMNSIZE;
   if (pATF->nIOBufferSize < ATF_MINBUFSIZE)
      pATF->nIOBufferSize = ATF_MINBUFSIZE;

   pATF->pszIOBuffer = (char *)calloc(pATF->nIOBufferSize, sizeof(char));
   if (pATF->pszIOBuffer == NULL)
   {
      pATF->nIOBufferSize = 0;
      return FALSE;
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: FreeIOBuffer
// PURPOSE:  Free the IO buffer used by this file.
//
static void FreeIOBuffer(ATF_FILEINFO *pATF)
{
   WPTRASSERT(pATF);
   if (pATF->pszIOBuffer != NULL)
      free(pATF->pszIOBuffer);
   pATF->pszIOBuffer = NULL;
   pATF->nIOBufferSize = 0;
}

//==============================================================================================
// FUNCTION: strncpyz
// PURPOSE:  Does a strncpy but guarantees that a terminating zero is placed on the 
//           destination string.
// RETURNS:  The destination buffer.
//
static LPSTR strncpyz(LPSTR pszDest, LPCSTR pszSrce, UINT uBufSize)
{
   ARRAYASSERT(pszDest, uBufSize);
#if 0
   LPSZASSERT(pszSrce);
#endif
   strncpy(pszDest, pszSrce, uBufSize-1);
   pszDest[uBufSize-1] = '\0';
   return pszDest;
}

//===============================================================================================
// FUNCTION: GetNumber
// PURPOSE:  Parse the next double out of the buffer, skipping delimeters etc.
//
static char *GetNumber(char *psBuf, double *pdNum)
{
#if 0
   LPSZASSERT(psBuf);
#endif   
   // Skip space characters to get the start of the number
   char *ps = psBuf;
   while (*ps==' ')
      ++ps;

   // Save a pointer to the start of the number.
   char *psStart = ps;

   // search for the end of this token
   while (*ps && !strchr(s_szWhitespace, *ps))
      ++ps;

   // skip trailing spaces.
   while (*ps==' ')
      ++ps;

   // Null terminate the string (knocking out the next delimiter) and step past the null if not end of line.
   if (*ps && !strchr(s_szLineTerm, *ps))
      *ps++ = '\0';
   else 
      *ps = '\0';

   if (pdNum)
   {
      WPTRASSERT(pdNum);
      *pdNum = atof(psStart);
   }
   return ps;
}

//===============================================================================================
// FUNCTION: GetVersion
// PURPOSE:  Parse the version number out of the data stream and check it against acceptable
//           values.
//
static BOOL GetVersion(char *psBuf, double *pdATFVersion, int *pnError)
{
#if 0
   LPSZASSERT(psBuf);
#endif
   WPTRASSERT(pdATFVersion);

   double dNum = 0;

   if (strlen(psBuf) < 5)
      ERRORRETURN(pnError, ATF_ERROR_INVALIDFILE);

   // Skip leading whitespace.
   char *psz = psBuf+FILE_ID_LEN;
   while (*psz && strchr(s_szWhitespace, *psz))
      psz++;

   GetNumber(psz, &dNum);

   if (strncmp(psBuf, ATF_FILE_ID, FILE_ID_LEN) == 0)
   {
      if ((dNum > ATF_CURRENTVERSION) || (dNum==0.0))
         ERRORRETURN(pnError, ATF_ERROR_BADVERSION);

      // File is an ATF file
      *pdATFVersion = dNum;
   }
   else if (strncmp(psBuf, PAF_FILE_ID, FILE_ID_LEN) == 0)
   {
      if (dNum != PREVIOUS_PAF_VERSION)
         ERRORRETURN(pnError, ATF_ERROR_BADVERSION);

      // Previous PAF V5.0 files are supported as ATF V0.0
      *pdATFVersion = 0.0;
   }
   else
      ERRORRETURN(pnError, ATF_ERROR_INVALIDFILE);

   return TRUE;
}

//===============================================================================================
// FUNCTION: CleanupMem
// PURPOSE:  Frees strings that have been strdup'ed into the array.
//
static void CleanupMem(char **ppsz, int nItems)
{
   if (!ppsz)
      return;
      
   while (nItems--)
   {
      if (*ppsz)
         free(*ppsz);
      ppsz++;
   }
}

//==============================================================================================
// FUNCTION: StripSpaces
// PURPOSE:  Strips space characters out of the start and end of the passed string.
// NOTES:    This routine is DESTRUCTIVE in that it replaces any trailing spaces with '\0's.
//           It contains its own static definition of whitespace, not that it does not 
//           include ','.
// RETURNS:  Pointer to the first non-white-space character.
//
static LPSTR StripSpaces(LPSTR pszSource)
{
   // Garbage in == garbage out.
   if (!pszSource)
      return pszSource;

   // Characters that are regarded as white space.
   static const char szWhiteSpace[] = " \t\r\n";
#if 0
   LPSZASSERT(pszSource);
#endif
   // Strip leading white space.
   char *pszFirstChar = pszSource + strspn(pszSource, szWhiteSpace);

   // If nothing left, return.
   if (*pszFirstChar=='\0')
   {
      pszSource[0] = '\0';
      return pszSource;
   }

   // Strip trailing white space.
   char *pszLastChar = pszFirstChar + strlen(pszFirstChar) - 1;
   while (pszLastChar > pszFirstChar)
   {
      if (strchr(szWhiteSpace, *pszLastChar) == NULL)
         break;
      *pszLastChar-- = '\0';
   }
   
   // Move the sub-string to the start of the source string.
   if (pszFirstChar > pszSource)
      memmove(pszSource, pszFirstChar, strlen(pszFirstChar)+1);

   return pszSource;
}

//===============================================================================================
// FUNCTION: FixColumnTitles
// PURPOSE:  This function reads PAF V5.0 (ATF 0.0) headings.
//
static BOOL FixColumnTitles(int nColumns, ATF_FILEINFO *pATF)
{
   WPTRASSERT(pATF);
   char *ps = pATF->pszIOBuffer;
   char *psEnd = pATF->pszIOBuffer + pATF->nIOBufferSize;

   for (int i=0; i<nColumns; i++)
   {
      // Search for the start of the title.
      while ((*ps=='"') || (*ps=='\t'))
      {
         ++ps;
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;
      }

      // Save the start of the title.
      char *psStart = ps;

      // Search for the end of the title.
      while ((*ps!='"') && (*ps!='\t'))
      {
         ++ps;
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;
      }
      *ps++ = '\0';
#ifndef _WINDOWS
      pATF->apszFileColTitles[i] = strdup(StripSpaces(psStart));
#else
      pATF->apszFileColTitles[i] = _strdup(StripSpaces(psStart));
#endif
      if (pATF->apszFileColTitles[i] == NULL)
         return FALSE;
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: FixColumnUnits
// PURPOSE:  This function reads PAF V5.0 (ATF 0.0) headings.
//
static BOOL FixColumnUnits(int nColumns, ATF_FILEINFO *pATF)
{
   WPTRASSERT(pATF);
   char *ps = pATF->pszIOBuffer;
   char *psEnd = pATF->pszIOBuffer + pATF->nIOBufferSize;

   for (int i=0; i<nColumns; i++)
   {
      // Search for the start of the units.
      while ((*ps=='"') || (*ps=='\t'))
      {
         ++ps;
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;
      }

      // Save the start of the units
      char *psStart = ps;

      // Search for the end of the units.
      while ((*ps!='"') && (*ps!='\t'))
      {
         ++ps;
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;
      }
      *ps++ = '\0';

#ifndef _WINDOWS
      pATF->apszFileColUnits[i] = strdup(StripSpaces(psStart));
#else
	  pATF->apszFileColUnits[i] = _strdup(StripSpaces(psStart));
#endif
	  if (pATF->apszFileColUnits[i] == NULL)
         return FALSE;
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: FixColumnHeadings
// PURPOSE:  Store away the column headings and the column units strings for ATF files.
//
static BOOL FixColumnHeadings(int nColumns, ATF_FILEINFO *pATF)
{
   WPTRASSERT(pATF);
   char *ps = pATF->pszIOBuffer;
   char *psEnd = pATF->pszIOBuffer + pATF->nIOBufferSize;

   for (int i=0; i<nColumns; i++)
   {
      BOOL bDoubleQuote = FALSE;
      BOOL bUnits = FALSE;

      // Search for the start of the title.
      while (strchr(s_szDelimiter, *ps))
      {
         ++ps;
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;
      }
      if (*ps=='"')
      {
         bDoubleQuote = TRUE;
         ++ps;
      }

      // Save the start of the title
      char *psStart = ps;

      for (; *ps && !strchr(s_szWhiteNoSpace, *ps); ++ps)
      {
         if (ps >= psEnd)    // return an error if past end of buffer.
            return FALSE;

         if (bDoubleQuote && (*ps=='"'))
            break;

         if (*ps == '(')
         {
            if (*(ps-1) == ' ')
               *(ps-1) = '\0';
            bUnits = TRUE;
            break;
         }
      }
      *ps++ = '\0';
#ifndef _WINDOWS
      pATF->apszFileColTitles[i] = strdup(StripSpaces(psStart));
#else
	  pATF->apszFileColTitles[i] = _strdup(StripSpaces(psStart));
#endif
	  if (pATF->apszFileColTitles[i] == NULL)
         return FALSE;

      if (bUnits)
      {
         psStart = ps;

         while (!strchr(s_szDelimiter, *ps) && (*ps!=')'))
         {
            if (bDoubleQuote && (*ps=='"'))
            {
               bDoubleQuote = FALSE;
               break;
            }
            ++ps;
            if (ps >= psEnd)    // return an error if past end of buffer.
               return FALSE;
         }
         *ps++ = '\0';
         
#ifndef _WINDOWS
		 pATF->apszFileColUnits[i] = strdup(StripSpaces(psStart));
#else
		 pATF->apszFileColUnits[i] = _strdup(StripSpaces(psStart));
#endif
		 if (pATF->apszFileColUnits[i] == NULL)
            return FALSE;

         if (bDoubleQuote)
         {
            while (*ps && (*ps!='"'))
               ps++;
            *ps++ = '\0';
         }
      }
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadLine
// PURPOSE:  This function masks gets and processes error conditions
//
static BOOL ReadLine(ATF_FILEINFO *pATF, int nEOFError, int *pnError)
{
   WPTRASSERT(pATF);
   char *pszIOBuffer = pATF->pszIOBuffer;

   int  nReturn = getsBuf(pATF, pszIOBuffer, pATF->nIOBufferSize);
   if (nReturn == GETS_EOF)
      ERRORRETURN(pnError, nEOFError);
   if (nReturn == GETS_ERROR)
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);
   if (nReturn == GETS_NOEOL)
      ERRORRETURN(pnError, ATF_ERROR_LINETOOLONG);

   if (pszIOBuffer[0] == ENDOFFILE)
      pszIOBuffer[0] = '\0';

   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadHeaderInfo
// PURPOSE:  Reads the header information out of the file and stores it away in the file
//           descriptor.
//
static BOOL ReadHeaderInfo(ATF_FILEINFO *pATF, int *pnColumns, int *pnError)
{
   WPTRASSERT(pnColumns);
   
   char szReadBuf[128];
   double dATFVersion, dNum;
   int nHeaders, nColumns, i;
   long lCurrentPos;

   // Check if the version number is suported

   if (getsBuf(pATF, szReadBuf, sizeof(szReadBuf)))
      ERRORRETURN(pnError, ATF_ERROR_INVALIDFILE);

   if (!GetVersion(szReadBuf, &dATFVersion, pnError))
      return FALSE;

   // Read in the number of header records and the number of columns of data
   // in the file

   if (getsBuf(pATF, szReadBuf, sizeof(szReadBuf)))
      ERRORRETURN(pnError, ATF_ERROR_BADHEADER);

   if (dATFVersion == 0.0)
   {
      // Read number of headers and columns as in previous (PAF 5.0) version
      GetNumber(szReadBuf, &dNum);
      nHeaders = (int)dNum;

      if (getsBuf(pATF, szReadBuf, sizeof(szReadBuf)))
         ERRORRETURN(pnError, ATF_ERROR_BADHEADER);

      GetNumber(szReadBuf, &dNum);
      nColumns = (int)dNum;
   }
   else
   {
      // Parse out first number (Headers)
      char *ps = GetNumber(szReadBuf, &dNum);
      nHeaders = (int)dNum;

      // Parse out second number (Columns)
      GetNumber(ps, &dNum);
      nColumns = (int)dNum;
   }

   if (nHeaders < 0)
      nHeaders = 0;
   if (nColumns < 0)
      nColumns = 0;

   // Return the number of columns to the user if a pointer was passed in
   if (pnColumns)
      *pnColumns = nColumns;
   if (nColumns > ATF_MAXCOLUMNS)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYCOLS);

   // Save the various pieces of information about this file

   // pATF->hFile          = hFile;  // now set in _CreateFileBuf
   pATF->nHeaders       = nHeaders;
   pATF->nColumns       = nColumns;
   pATF->eState         = eOPENED;
   pATF->bWriting       = FALSE;
   pATF->dFileVersion   = dATFVersion;

   if (!AllocIOBuffer(pATF))
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);

   // Now save the current position and skip the header records and get the
   // column title and unit strings
   lCurrentPos = SetFilePointerBuf(pATF, 0, NULL, FILE_CURRENT);

   for (i=0; i<nHeaders; i++)
      if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
         return FALSE;

   pATF->apszFileColTitles = (char **)calloc(nColumns, sizeof(char *));
   if (pATF->apszFileColTitles == NULL)
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);

   pATF->apszFileColUnits  = (char **)calloc(nColumns, sizeof(char *));
   if (pATF->apszFileColUnits == NULL)
   {
      free(pATF->apszFileColTitles);
      pATF->apszFileColTitles = NULL;
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);
   }

   if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
      return FALSE;

   if (dATFVersion == 0.0)
   {
      if (!FixColumnTitles(nColumns, pATF))
      {
         CleanupMem(pATF->apszFileColTitles, nColumns);
         ERRORRETURN(pnError, ATF_ERROR_BADHEADER);
      }

      if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
      {
         CleanupMem(pATF->apszFileColTitles, nColumns);
         return FALSE;
      }
      if (!FixColumnUnits(nColumns, pATF))
      {
         CleanupMem(pATF->apszFileColTitles, nColumns);
         CleanupMem(pATF->apszFileColUnits, nColumns);
         ERRORRETURN(pnError, ATF_ERROR_BADHEADER);
      }
   }
   else if (!FixColumnHeadings(nColumns, pATF))
   {
      CleanupMem(pATF->apszFileColTitles, nColumns);
      CleanupMem(pATF->apszFileColUnits, nColumns);
      ERRORRETURN(pnError, ATF_ERROR_BADHEADER);
   }

   // Return to the saved position
   SetFilePointerBuf(pATF, lCurrentPos, NULL, FILE_BEGIN);
   return TRUE;
}

//===============================================================================================
// FUNCTION: WriteHeaderInfo
// PURPOSE:  Write the first compulsory header lines out to the file.
//
static BOOL WriteHeaderInfo(ATF_FILEINFO *pATF, int nColumns, int *pnError)
{
   // Writing a new file; Setup any necessary info
   pATF->dFileVersion = ATF_CURRENTVERSION;
   pATF->eState       = eOPENED;
   pATF->bWriting     = TRUE;
   pATF->bDataOnLine  = FALSE;
   pATF->nHeaders     = 0;
   pATF->nColumns     = nColumns;

   // Allocate / re-allocate the IOBuffer.
   if (!AllocIOBuffer(pATF))
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);

   // Allocate space to hold column titles and units.
   pATF->apszFileColTitles = (char **)calloc(nColumns, sizeof(char *));
   if (pATF->apszFileColTitles == NULL)
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);

   pATF->apszFileColUnits  = (char **)calloc(nColumns, sizeof(char *));
   if (pATF->apszFileColUnits == NULL)
   {
      free(pATF->apszFileColTitles);
      pATF->apszFileColTitles = NULL;
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);
   }

   if (pATF->uFlags & ATF_DONTWRITEHEADER)
   {
      pATF->lFilePos = 0;
      return TRUE;
   }

   // Write the special marker string and the file version number
   char *pszIOBuffer = pATF->pszIOBuffer;
   sprintf(pszIOBuffer, "%s%s%.1f%s", ATF_FILE_ID, pATF->szSeparator, ATF_CURRENTVERSION, s_szEndOfLine);
   if (!putsBuf(pATF, pszIOBuffer))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Save the current position so that we may update the number of header records
   pATF->lFilePos = SetFilePointerBuf(pATF, 0, NULL, FILE_CURRENT);

   // Write the number of headers and columns that will be written, leaving
   // enough trailing spaces for 4 digits of headers (more than enough)
   sprintf(pszIOBuffer, "0%s%d     %s", pATF->szSeparator, nColumns, s_szEndOfLine);
   if (!putsBuf(pATF, pszIOBuffer))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   return TRUE;
}

//===============================================================================================
// FUNCTION: _FormatNumber
// PURPOSE:  Formats the digits of dNum into pszString, suppressing trailing zeros.
//           In the event that "0." is produced as output from gcvt, truncate to "0"
//
// RETURNS:  TRUE if success; FALSE if failure
//

#ifdef _WINDOWS
// #include "..\AxonValidation\AxonValidation.h"     // for VAL_* functions (standard Axon float/double->string formatting)
#endif

static BOOL _FormatNumber(double dNum, int nDigits, char *pszString, UINT uSize)
{
   ARRAYASSERT(pszString, uSize);

#ifndef _WINDOWS
   gcvt(dNum, nDigits, pszString);
#else
   _gcvt(dNum, nDigits, pszString);
#endif
   int l = (int)strlen(pszString);
   if ((l > 0) && (pszString[l-1]=='.'))
      pszString[l-1] = '\0';
   return TRUE;
/*
#else

   return VAL_ConvertDblToStr(dNum, pszString, uSize, VAL_FMT_SIG_DIGITS, 0, nDigits, NULL);

#endif
*/
}

//===============================================================================================
// FUNCTION: ATF_OpenFile
// PURPOSE:  Opens an ATF file for either reading or writing.
//
BOOL WINAPI ATF_OpenFile(LPCSTR szFileName, UINT uFlags, int *pnColumns, int *pnFile, int *pnError)
{
#if 0
   LPSZASSERT(szFileName);
#endif
   WPTRASSERT(pnColumns);
   WPTRASSERT(pnFile);
   
   // keep state of DONTWRITEHEADER flag
   BOOL bDontWriteHeader = uFlags & ATF_DONTWRITEHEADER;

   int nColumns, nFile;
   HANDLE hFile = INVALID_HANDLE_VALUE;
   ATF_FILEINFO *pATF = NULL;
   if (!GetNewFileDescriptor(&pATF, &nFile, pnError))
      return FALSE;
      
   // copy name:
#ifndef _WINDOWS
   pATF->pszFileName = strdup(szFileName);
#else
   pATF->pszFileName = _strdup(szFileName);
#endif
   if (pATF->pszFileName == NULL)
      goto OpenError;

   if (uFlags & ATF_READONLY)
   {
      hFile = CreateFileBuf(pATF, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (hFile == INVALID_HANDLE_VALUE)
      {
         if (pnError)
            *pnError = ATF_ERROR_NOFILE;
         goto OpenError;
      }

      pATF->uFlags = uFlags;

      // Read the header information into the global variables
      if (!ReadHeaderInfo(pATF, pnColumns, pnError))
         goto OpenError;
   }
   else
   {
      nColumns = *pnColumns;
      if (nColumns > ATF_MAXCOLUMNS)
      {
         if (pnError)
            *pnError = ATF_ERROR_TOOMANYCOLS;
         goto OpenError;
      }

      // Try to open any existing file with same name

#ifdef _WINDOWS
      hFile = CreateFileBuf(pATF, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (hFile != INVALID_HANDLE_VALUE)
      {
         if ((uFlags & ATF_OVERWRTIFEXIST)==0)
         {
            if (pnError)
               *pnError = ATF_ERROR_FILEEXISTS;
            goto OpenError;
         }
         if ((uFlags & ATF_APPENDIFEXIST)!=0)
         {
            // If we are appending to the file, read the header
            // to confirm that it is a valid ATF file that we are
            // appending to.

            if (!ReadHeaderInfo(pATF, NULL, pnError))
               goto OpenError;
         }
         CloseHandleBuf(pATF);
      }
      else
#endif
         uFlags = ATF_WRITEONLY;

      pATF->uFlags = uFlags;

      // Now, open the file for output
      hFile = CreateFileBuf(pATF, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL,
                            uFlags & ATF_APPENDIFEXIST ? OPEN_EXISTING : CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, NULL);
      if (hFile == INVALID_HANDLE_VALUE)
      {
         if (pnError)
            *pnError = ATF_ERROR_NOFILE;
         goto OpenError;
      }

      if ((uFlags & ATF_APPENDIFEXIST)!=0)
      {
         // pATF->hFile       = hFile;
         pATF->bDataOnLine = FALSE;
         pATF->eState      = eDATAAPPENDED;
         pATF->bWriting    = TRUE;
      }
      else
      {
         // Set the DONTWRITEHEADER flag if needed
         pATF->uFlags |= bDontWriteHeader;

         // Write the header information and set up global variables
         if (!WriteHeaderInfo(pATF, nColumns, pnError))
         {
            CloseHandleBuf(pATF);
            goto OpenError;
         }
      }
   }
   *pnFile = nFile;
   return TRUE;

OpenError:
   if (hFile != INVALID_HANDLE_VALUE)
      CloseHandleBuf(pATF);
   if (pATF->pszFileName != NULL)
      free(pATF->pszFileName);
   free(pATF);
   g_FileDescriptor[nFile] = NULL;
   return FALSE;
}

//===============================================================================================
// FUNCTION: HasUnits
// PURPOSE:  Checks for the presence of a units string.
//
static BOOL HasUnits(LPCSTR pszUnits)
{
   return (pszUnits && (pszUnits[0]!=0));
}

//===============================================================================================
// FUNCTION: UpdateHeaders
// PURPOSE:  Update the headers in the ATF file.
//
static BOOL UpdateHeaders(ATF_FILEINFO *pATF, int *pnError)
{
   WPTRASSERT(pATF);

   if (pATF->uFlags & ATF_DONTWRITEHEADER)
      return TRUE;

   char *pszIOBuffer = pATF->pszIOBuffer;

   // if there are unterminated header lines then terminate the last one
   if (pATF->bDataOnLine)
   {
      if (!putsBuf(pATF, s_szEndOfLine))
         ERRORRETURN(pnError, ATF_ERROR_IOERROR);
      pATF->nHeaders++;
      pATF->bDataOnLine = FALSE;
   }

   // Save the current position of the file
   long lCurrentPos = SetFilePointerBuf(pATF, 0, NULL, FILE_CURRENT);

   // Move to the location of the record containing the number of header records
   SetFilePointerBuf(pATF, pATF->lFilePos, NULL, FILE_BEGIN);

   // Create the string for the number of header records
   {
      sprintf(pszIOBuffer, "%d%s%d", pATF->nHeaders, pATF->szSeparator, pATF->nColumns);
      if (!putsBuf(pATF, pszIOBuffer))
         ERRORRETURN(pnError, ATF_ERROR_IOERROR);
   }

   // Restore the position of the file
   SetFilePointerBuf(pATF, lCurrentPos, NULL, FILE_BEGIN);

   // Output the column titles and units record
   for (int i=0; i<pATF->nColumns; i++)
   {
      // Start with separator if not the first column
      if (i > 0)
         strcpy(pszIOBuffer, pATF->szSeparator);
      else 
         pszIOBuffer[0] = '\0';

      // Add next title string
      strcat(pszIOBuffer, "\"");
      if (pATF->apszFileColTitles[i] != NULL)
      {
         strcat(pszIOBuffer, pATF->apszFileColTitles[i]);
         if (HasUnits(pATF->apszFileColUnits[i]))
            strcat(pszIOBuffer, " ");
      }

      if (HasUnits(pATF->apszFileColUnits[i]))
      {
         strcat(pszIOBuffer, "(");
         strcat(pszIOBuffer, pATF->apszFileColUnits[i]);
         strcat(pszIOBuffer, ")");
      }
      strcat(pszIOBuffer, "\"");
      if (!putsBuf(pATF, pszIOBuffer))
         ERRORRETURN(pnError, ATF_ERROR_IOERROR);
   }

   if (!putsBuf(pATF, s_szEndOfLine))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   pATF->bDataOnLine = FALSE;
   pszIOBuffer[0] = '\0';
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_CloseFile
// PURPOSE:  Close an ATF file that was opened with ATF_OpenFile.
//
BOOL WINAPI ATF_CloseFile(int nFile)
{
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, NULL))
      return FALSE;

   // Check that we have written out the headers at least.
   if ((pATF->eState < eDATAWRITTEN) && pATF->bWriting)
      UpdateHeaders(pATF, NULL);

   // Close the file and update the state variable
   CloseHandleBuf(pATF);
   CleanupMem(pATF->apszFileColTitles, pATF->nColumns);
   free(pATF->apszFileColTitles);
   CleanupMem(pATF->apszFileColUnits, pATF->nColumns);
   free(pATF->apszFileColUnits);

   FreeIOBuffer(pATF);

   if (pATF->pszFileName != NULL)
      free(pATF->pszFileName);

   free(pATF);
   g_FileDescriptor[nFile] = NULL;

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_SetSeperator
// PURPOSE:  Sets the data item separator to be used when writing out an ATF file.
//
BOOL WINAPI ATF_SetSeperator(int nFile, BOOL bUseCommas, int *pnError)
{
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   pATF->szSeparator[0] = s_szDelimiter[bUseCommas ? 1 : 0];
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_IsAppending
// PURPOSE:  Returns TRUE if the file has been opened for appending.
//
BOOL WINAPI ATF_IsAppending(int nFile)
{
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, NULL))
      return FALSE;

   return (pATF->eState == eDATAAPPENDED);
}

//===============================================================================================
// FUNCTION: ATF_RewindFile
// PURPOSE:  Rewind the data section of the ATF file back to the beginning.
//
BOOL WINAPI ATF_RewindFile(int nFile, int *pnError)
{
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   if (pATF->bWriting)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   if (pATF->eState != eDATAREAD)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   SetFilePointerBuf(pATF, pATF->lDataPtr, NULL, FILE_BEGIN);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_CountDataLines
// PURPOSE:  Counts lines of data in an ATF file, stopping at the first empty line, or EOF.
//
BOOL WINAPI ATF_CountDataLines(int nFile, long *plNumLines, int *pnError)
{
   WPTRASSERT(plNumLines);
   long lDataLines = 0L;
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   while (ReadDataRecord(pATF, pnError))
   {
      if (strchr(s_szLineTerm, pATF->pszIOBuffer[0]))
         break;
      ++lDataLines;
   }

   ATF_RewindFile(nFile, NULL);
  
   *plNumLines = lDataLines;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_GetNumHeaders
// PURPOSE:  Gets the number of optional data records in the ATF file.
//
BOOL WINAPI ATF_GetNumHeaders(int nFile, int *pnHeaders, int *pnError)
{
   WPTRASSERT(pnHeaders);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;
   *pnHeaders = pATF->nHeaders;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_WriteHeaderRecord
// PURPOSE:  Writes an optional header r3cord to the ATF file.
//
BOOL WINAPI ATF_WriteHeaderRecord(int nFile, LPCSTR pszText, int *pnError)
{
#if 0
   LPSZASSERT(pszText);
#endif
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   if (pATF->uFlags & ATF_DONTWRITEHEADER)
      return TRUE;

   char *pszIOBuffer = pATF->pszIOBuffer;

   // Check that we are in the correct state for this file
   if (pATF->eState > eHEADERED)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   // Update the state if necessary
   pATF->eState = eHEADERED;

   // Write out the header record
   if (pATF->bDataOnLine)
      strcpy(pszIOBuffer, pATF->szSeparator);
   else
      pszIOBuffer[0] = '\0';

   strcat(pszIOBuffer, "\"");
   strcat(pszIOBuffer, pszText);
   strcat(pszIOBuffer, "\"");
   if (!putsBuf(pATF, pszIOBuffer))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   pATF->bDataOnLine = TRUE;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_SetColumnTitle
// PURPOSE:  Sets the next column title. Columns are labeled sequentially.
//
BOOL WINAPI ATF_SetColumnTitle(int nFile, LPCSTR pszText, int *pnError)
{
#if 0
   LPSZASSERT(pszText);
#endif
   int i;
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Check that we are in the correct state for this file
   if (pATF->eState > eHEADERED)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   for (i=0; i<pATF->nColumns; i++)
      if (pATF->apszFileColTitles[i] == NULL)
         break;

   if (i==pATF->nColumns)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYCOLS);

   // Copy the new column title into the string buffer and save a pointer to it.
#ifndef _WINDOWS
   LPSTR psz = strdup(pszText);
#else
   LPSTR psz = _strdup(pszText);
#endif
   if (psz == NULL)
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);

   pATF->apszFileColTitles[i] = psz;

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_SetColumnUnits
// PURPOSE:  Sets the next column units label. Columns are labeled sequentially.
//
BOOL WINAPI ATF_SetColumnUnits(int nFile, LPCSTR pszText, int *pnError)
{
#if 0
   LPSZASSERT(pszText);
#endif
   int i;
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Check that we are in the correct state for this file
   if (pATF->eState > eHEADERED)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   for (i=0; i<pATF->nColumns; i++)
      if (pATF->apszFileColUnits[i] == NULL)
         break;

   if (i==pATF->nColumns)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYCOLS);

   // Copy the new column Unit into the string buffer and save a pointer to it.
#ifndef _WINDOWS
   LPSTR psz = strdup(pszText);
#else
   LPSTR psz = _strdup(pszText);
#endif
   if (psz == NULL)
      ERRORRETURN(pnError, ATF_ERROR_NOMEMORY);
   pATF->apszFileColUnits[i] = psz;

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_WriteEndOfLine
// PURPOSE:  Terminates the current data line when writing to an ATF file.
//
BOOL WINAPI ATF_WriteEndOfLine(int nFile, int *pnError)
{
   ATF_FILEINFO * pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Write out the EndOfLine character to the file
   if (!putsBuf(pATF, s_szEndOfLine))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Reset data-on-line flag
   pATF->bDataOnLine = FALSE;

   // If a header is being written, increment the count of header records
   if (pATF->eState == eHEADERED)
      pATF->nHeaders++;

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_WriteDataRecord
// PURPOSE:  Writes a data record out to the ATF file.
//
BOOL WINAPI ATF_WriteDataRecord(int nFile, LPCSTR pszText, int *pnError)
{
#if 0
   LPSZASSERT(pszText);
#endif
   ATF_FILEINFO * pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Check that we are in the correct state for this file
   if (pATF->eState < eDATAWRITTEN)
   {
      if (!UpdateHeaders(pATF, pnError))
         return FALSE;

      pATF->eState = eDATAWRITTEN;
   }
   else if (pATF->bDataOnLine)   
      if (!putsBuf(pATF, pATF->szSeparator))
         ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Write out the text string to the file
   if (!putsBuf(pATF, pszText))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Set data-on-line flag
   pATF->bDataOnLine = TRUE;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_WriteDataComment
// PURPOSE:  Writes a comment record out to the ATF file.
//
BOOL WINAPI ATF_WriteDataComment(int nFile, LPCSTR pszText, int *pnError)
{
#if 0
   LPSZASSERT(pszText);
#endif
   char buf[128];
#ifdef _WINDOWS
   _snprintf(buf, sizeof(buf), "\"%s\"", pszText);
#else
   snprintf(buf, sizeof(buf), "\"%s\"", pszText);
#endif
   return ATF_WriteDataRecord(nFile, buf, pnError);
}

//===============================================================================================
// FUNCTION: ATF_WriteDataRecordArray
// PURPOSE:  Writes an array of data values out to a line of the ATF file.
//
BOOL WINAPI ATF_WriteDataRecordArray(int nFile, int nCount, double *pdVals, int *pnError)
{
   ARRAYASSERT(pdVals, nCount);
   char  psTemp[ATF_DBL_STR_LEN];
   
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   if (nCount > pATF->nColumns)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYCOLS);

   char *pszIOBuffer = pATF->pszIOBuffer;

   // Check that we are in the correct state for this file
   if (pATF->eState < eDATAWRITTEN)
   {
      if (!UpdateHeaders(pATF, pnError))
         return FALSE;
      pATF->eState = eDATAWRITTEN;
   }

   // Create the output string for this record
   char *ps = pszIOBuffer;
   *ps = '\0';

   if (nCount > 0)
   {
      if (pATF->bDataOnLine)
         strcpy(ps++, pATF->szSeparator);

      if (!_FormatNumber(*pdVals++, ATF_DBL_SIG_DIGITS, psTemp, ATF_DBL_STR_LEN))
         ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
      strcpy(ps, psTemp);
      ps += strlen(psTemp);
   }

   for (int i=1; i<nCount; i++)
   {
      strcpy(ps, pATF->szSeparator);
      ps += strlen(pATF->szSeparator);
      if (!_FormatNumber(*pdVals++, ATF_DBL_SIG_DIGITS, psTemp, ATF_DBL_STR_LEN))
         ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
      strcpy(ps, psTemp);
      ps += strlen(psTemp);
   }

   // Write out the text string to the file
   if (!putsBuf(pATF, pszIOBuffer))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Set data-on-line flag
   pATF->bDataOnLine = TRUE;
   return TRUE;
}


//===============================================================================================
// FUNCTION: ATF_WriteDataRecordArrayFloat
// PURPOSE:  Writes an array of float data values out to a line of the ATF file.
//
BOOL WINAPI ATF_WriteDataRecordArrayFloat(int nFile, int nCount, float *pfVals, int *pnError)
{
   ARRAYASSERT(pfVals, nCount);
   char     psTemp[ATF_FLT_STR_LEN];
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   if (nCount > pATF->nColumns)
      ERRORRETURN(pnError, ATF_ERROR_TOOMANYCOLS);

   char *pszIOBuffer = pATF->pszIOBuffer;

   // Check that we are in the correct state for this file
   if (pATF->eState < eDATAWRITTEN)
   {
      if (!UpdateHeaders(pATF, pnError))
         return FALSE;
      pATF->eState = eDATAWRITTEN;
   }

   // Create the output string for this record
   char *ps = pszIOBuffer;
   *ps = '\0';

   if (nCount > 0)
   {
      if (pATF->bDataOnLine)
         strcpy(ps++, pATF->szSeparator);

      if (!_FormatNumber((double)*pfVals++, ATF_FLT_SIG_DIGITS, psTemp, ATF_FLT_STR_LEN))
         ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
      strcpy(ps, psTemp);
      ps += strlen(psTemp);
   }

   for (int i=1; i<nCount; i++)
   {
      strcpy(ps, pATF->szSeparator);
      ps += strlen(pATF->szSeparator);
      if (!_FormatNumber((double)*pfVals++, ATF_FLT_SIG_DIGITS, psTemp, ATF_FLT_STR_LEN))
         ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
      strcpy(ps, psTemp);
      ps += strlen(psTemp);
   }

   // Write out the text string to the file

   if (!putsBuf(pATF, pszIOBuffer))
      ERRORRETURN(pnError, ATF_ERROR_IOERROR);

   // Set data-on-line flag

   pATF->bDataOnLine = TRUE;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_WriteDataRecord1
// PURPOSE:  Writes a single double data item out to the ATF file.
//
BOOL WINAPI ATF_WriteDataRecord1(int nFile, double dNum1, int *pnError)
{
   char  psTemp[ATF_DBL_STR_LEN];

   if (!_FormatNumber(dNum1, ATF_DBL_SIG_DIGITS, psTemp, ATF_DBL_STR_LEN))
      ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
   return ATF_WriteDataRecord(nFile, psTemp, pnError);
}

//===============================================================================================
// FUNCTION: ATF_WriteDataRecord1Float
// PURPOSE:  Writes a single float data item out to the ATF file.
//
BOOL WINAPI ATF_WriteDataRecord1Float(int nFile, float fNum1, int *pnError)
{
   char psTemp[ATF_FLT_STR_LEN];

   if (!_FormatNumber((double)fNum1, ATF_FLT_SIG_DIGITS, psTemp, ATF_FLT_STR_LEN))
      ERRORRETURN(pnError, ATF_ERROR_BADFLTCNV);
   return ATF_WriteDataRecord(nFile, psTemp, pnError);
}

//===============================================================================================
// FUNCTION: GetComment
// PURPOSE:  Strips a comment string out of the data stream, removing separators and 
//           double quotes.
//
static char *GetComment(char *pszStr)
{
   // Strip off any leading or trailing space characters
   pszStr = StripSpaces(pszStr);

   // If nothing left, return.
   if (*pszStr=='\0')
      return pszStr;

   // Strip leading and trailing quotes from the string
   if (*pszStr == '"')
   {
      pszStr++;

      char *ps = pszStr;
      while (*ps && (*ps!='"'))
         ps++;

      if (*ps != '\0')
         *ps = '\0';
   }
   return pszStr;
}

//===============================================================================================
// FUNCTION: ReadHeaderLine
// PURPOSE:  Reads an optional header line from the file.
//
static BOOL ReadHeaderLine(ATF_FILEINFO *pATF, int *pnError)
{
   WPTRASSERT(pATF);
   // Check that we are in the correct state for this file
   if (pATF->eState > eHEADERED)
      ERRORRETURN(pnError, ATF_ERROR_BADSTATE);

   pATF->eState = eHEADERED;

   // Check if there are any more header records left to be read
   if (pATF->nHeaders < 1)
      ERRORRETURN(pnError, ATF_ERROR_NOMORE);

   // Read a header record and copy it into the return buffer
   if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
      return FALSE;

   // remove any trailing white space at the end of the line.
   StripSpaces(pATF->pszIOBuffer);
   pATF->nHeaders--;
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_ReadHeaderLine
// PURPOSE:  Exported version of ReadHeaderLine.
//
BOOL WINAPI ATF_ReadHeaderLine(int nFile, char *psBuf, int nMaxLen, int *pnError)
{
   ARRAYASSERT(psBuf, nMaxLen);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Load the next line, checking state etc
   if (!ReadHeaderLine(pATF, pnError))
      return FALSE;

   strncpyz(psBuf, pATF->pszIOBuffer, nMaxLen);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_ReadHeaderNoQuotes
// PURPOSE:  Reads a header record, stripping of any double quotes if present.
//
BOOL WINAPI ATF_ReadHeaderNoQuotes(int nFile, char *psBuf, int nMaxLen, int *pnError)
{
   ARRAYASSERT(psBuf, nMaxLen);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Load the next line, checking state etc
   if (!ReadHeaderLine(pATF, pnError))
      return FALSE;

   // Strip off any leading and trailing quotes from the string
   char *psComment = GetComment(pATF->pszIOBuffer);
   strncpyz(psBuf, psComment, nMaxLen);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_GetColumnTitle
// PURPOSE:  Returns the title of a particular column in the ATF file.
//
BOOL WINAPI ATF_GetColumnTitle(int nFile, int nColumn, char *pszText, int nMaxTxt, 
                                      int *pnError)
{
   ARRAYASSERT(pszText, nMaxTxt);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Check that the column number is within range.
   if ((nColumn < 0) || (nColumn >= pATF->nColumns))
      ERRORRETURN(pnError, ATF_ERROR_BADCOLNUM);

   if (pATF->apszFileColTitles[nColumn] != NULL)
      strncpyz(pszText, pATF->apszFileColTitles[nColumn], nMaxTxt);
   else 
      pszText[0] = '\0';
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_GetColumnUnits
// PURPOSE:  Returns the units of a particular column in the ATF file.
//
BOOL WINAPI ATF_GetColumnUnits(int nFile, int nColumn, char *pszText, int nMaxTxt, 
                                      int *pnError)
{
   ARRAYASSERT(pszText, nMaxTxt);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Check that the column number is within range.
   if ((nColumn < 0) || (nColumn >= pATF->nColumns))
      ERRORRETURN(pnError, ATF_ERROR_BADCOLNUM);

   if (pATF->apszFileColUnits[nColumn] != NULL)
      strncpyz(pszText, pATF->apszFileColUnits[nColumn], nMaxTxt);
   else 
      pszText[0] = '\0';
   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadDataRecord
// PURPOSE:  Internal workhorse function for parsing lines out of the data stream.
//
static BOOL ReadDataRecord(ATF_FILEINFO *pATF, int *pnError)
{
   WPTRASSERT(pATF);

   // Check that we are in the correct state for this file
   if (pATF->eState < eDATAREAD)
   {
      // Read any header records that were not processed
      while (pATF->nHeaders-- > 0)
         if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
            return FALSE;

      // Skip the column titles and units
      if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
         return FALSE;
      if (pATF->dFileVersion == 0.0)
         if (!ReadLine(pATF, ATF_ERROR_BADHEADER, pnError))
            return FALSE;

      pATF->lDataPtr = SetFilePointerBuf(pATF, 0, NULL, FILE_CURRENT);
      pATF->eState = eDATAREAD;
   }

   // Update the file state and check for end-of-file

   // JT - we may need this:
   // if (feof(pATF->hFile))
   //    ERRORRETURN(pnError, ATF_ERROR_NOMORE);

   // Read the next record from the data file and return it

   // JT - we may need to reconsider what error we use here:
   // return ReadLine(pATF, ATF_ERROR_IOERROR, pnError);
   return ReadLine(pATF, ATF_ERROR_NOMORE, pnError);
}

//===============================================================================================
// FUNCTION: ATF_ReadDataRecord
// PURPOSE:  Returns the next complete line from the ATF file.
//
BOOL WINAPI ATF_ReadDataRecord(int nFile, char *pszText, int nMaxLen, int *pnError)
{
   ARRAYASSERT(pszText, nMaxLen);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   if (!ReadDataRecord(pATF, pnError))
      return FALSE;

   strncpyz(pszText, pATF->pszIOBuffer, nMaxLen);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_ReadDataRecordArray
// PURPOSE:  Reads an array of data values from the ATF file.
//
BOOL WINAPI ATF_ReadDataRecordArray(int nFile, int nCount, double *pdVals,
                                           char *pszComment, int nMaxLen, int *pnError)
{
   ARRAYASSERT(pdVals, nCount);
   ARRAYASSERT(pszComment, nMaxLen);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Perform the necessary checks and read the data record
   if (!ReadDataRecord(pATF, pnError))
      return FALSE;

   // Read out the requested numbers
   char *ps = pATF->pszIOBuffer;
   for (int i=0; i<nCount; i++)
      ps = GetNumber(ps, pdVals+i);

   if (pszComment != NULL)
   {
      ps = GetComment(ps);
      strncpyz(pszComment, ps, nMaxLen);
   }

   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_ReadDataColumn
// PURPOSE:  Reads the next line from the ATF file but only returns the data value for a
//           particular column.
//
BOOL WINAPI ATF_ReadDataColumn(int nFile, int nColumn, double *pdVal, int *pnError)
{
   WPTRASSERT(pdVal);
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   // Perform the necessary checks and read the data record
   if (!ReadDataRecord(pATF, pnError))
      return FALSE;

   // Skip over the prior columns
   char *ps = pATF->pszIOBuffer;
   for (int i=0; i<nColumn; i++)
      ps = GetNumber(ps, NULL);

   // Read out the requested number
   ps = GetNumber(ps, pdVal);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ATF_BuildErrorText
// PURPOSE:  Builds an error string from an Error ID and a filename.
//
BOOL WINAPI ATF_BuildErrorText(int nErrorNum, LPCSTR szFileName, char *sTxtBuf, int nMaxLen)
{
#if 0
   LPSZASSERT(szFileName);
#endif
   ARRAYASSERT(sTxtBuf, nMaxLen);
   
   if (nMaxLen < 2)
   {
      TRACE("Error string destination buffer too short!\n");
      ASSERT(0);
      return FALSE;
   }

   char szTemplate[276];
#ifdef _WINDOWS
   if (!LoadStringA(g_hInstance, nErrorNum, szTemplate, sizeof(szTemplate)))
   {
      LoadStringA(g_hInstance, ATF_ERROR_NOMESSAGESTR, szTemplate, sizeof(szTemplate));
      _snprintf(sTxtBuf, nMaxLen, szTemplate, nErrorNum);
#else
   if (!c_LoadString(g_hInstance, nErrorNum, szTemplate, sizeof(szTemplate)))
   {
      c_LoadString(g_hInstance, ATF_ERROR_NOMESSAGESTR, szTemplate, sizeof(szTemplate));
      snprintf(sTxtBuf, nMaxLen, szTemplate, nErrorNum);
#endif
      //      ERRORMSG(sTxtBuf);
      return FALSE;
   }
#ifdef _WINDOWS
   _snprintf(sTxtBuf, nMaxLen, szTemplate, szFileName);
#else
   snprintf(sTxtBuf, nMaxLen, szTemplate, szFileName);
#endif
   return TRUE;
}   

//===============================================================================================
// FUNCTION: ATF_GetFileDateTime
// PURPOSE:  Gets the date and time at which the data file was created.
//
BOOL WINAPI ATF_GetFileDateTime(int nFile, long *plDate, long *plTime, int *pnError)
{
#ifdef _WINDOWS
   ATF_FILEINFO *pATF = NULL;
   if (!GetFileDescriptor(&pATF, nFile, pnError))
      return FALSE;

   SYSTEMTIME  Time = { 0 };
   FILETIME    CreationTime = { 0 };
   if (GetFileTime(pATF->hFile, &CreationTime,	NULL, NULL))
   {
      FILETIME LocalTime = { 0 };
      VERIFY(FileTimeToLocalFileTime(&CreationTime, &LocalTime));
      VERIFY(FileTimeToSystemTime(&LocalTime, &Time));
   }
   int nYear  = int(Time.wYear);

   if (plDate)
      *plDate = long(nYear*10000 + Time.wMonth*100 + Time.wDay);
   if (plTime)
      *plTime = long(((Time.wHour*60) + Time.wMinute)*60 + Time.wSecond);
#endif
   return TRUE;
}

