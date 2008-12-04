//***********************************************************************************************
//
//    Copyright (c) 1993-1999 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// FILE: ABFUTIL.CPP   A utilities module for the ABF file routines.
//
#include "../Common/wincpp.hpp"
#include "abfutil.h"
#include "abffiles.h"
#include "abfheadr.h"

#ifdef _WINDOWS
#define USE_AXOVDATE
#endif

#define  ABFU_VALID_SIG_CHARS     " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_#"

#if defined(__UNIX__) || defined(__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif
/*
//==============================================================================================
// FUNCTION: ABFU_GetTempFileName
// PURPOSE:  Gets a temporary file name in the directory pointed to by the %TEMP% environment
//           variable.
//
UINT WINAPI ABFU_GetTempFileName(LPCSTR szPrefix, UINT uUnique, LPSTR lpTempName)
{
#ifdef _WINDOWS
   ARRAYASSERT(lpTempName, _MAX_PATH);
   char szTempPath[_MAX_PATH];
   if (!GetTempPathA(_MAX_PATH, szTempPath))
   strcpy(szTempPath, "C:\\");
   return GetTempFileNameA(szTempPath, szPrefix, uUnique, lpTempName);
#else   
   strcpy(lpTempName,"ABFTMPXXXXXX");
   mkstemp(lpTempName);
   return 1;
#endif
}

//   tmpnam (lpTempName);
//	return 1;
//}*/

//===============================================================================================
// FUNCTION: ABFU_ReadFile
// PURPOSE:  Wrapper on the ReadFile function that returns FALSE at EOF.
//   
BOOL WINAPI ABFU_ReadFile(FILEHANDLE hFile, LPVOID lpBuf, DWORD dwBytesToRead)
{
   DWORD dwBytesRead;
   BOOL bOK = c_ReadFile(hFile, lpBuf, dwBytesToRead, &dwBytesRead, NULL);
   return (bOK && (dwBytesRead==dwBytesToRead));
}
/*
//===============================================================================================
// FUNCTION: ABFU_FormatDouble
// PURPOSE:  Formats the digits of dNum into pszString, suppressing trailing zeros.
//           In the event that "0." is produced as output from gcvt, truncate to "0"
//
// RETURNS:  TRUE if success; FALSE if failure
//
#ifdef USE_AXOVDATE
#include "..\AxonValidation\AxonValidation.h"     // for VAL_* functions (standard Axon float/double->string formatting)
#endif

BOOL WINAPI ABFU_FormatDouble(double dNum, int nDigits, char *pszString, UINT uSize)
{
#ifndef USE_AXOVDATE
   gcvt(dNum, nDigits, pszString);
   int l = strlen(pszString);
   if ((l > 0) && (pszString[l-1]=='.'))
      pszString[l-1] = '\0';
   return TRUE;
#else
   return VAL_ConvertDblToStr(dNum, pszString, uSize, VAL_FMT_SIG_DIGITS, 0, nDigits, NULL);
#endif
}

//===============================================================================================
// FUNCTION: ABFU_FormatHMS
// PURPOSE:  Formats the time in HH:MM:SS.
// RETURNS:  The number of characters written to the buffer.
// NOTE:     The buffer passed in should be at least AXU_MAXHMSLENGTH characters long.
//
int WINAPI ABFU_FormatHMS( UINT uSeconds, char *pszBuffer, UINT uMaxLen )
{
   UINT uHours   = (uSeconds / 3600U) % 100U;
   UINT uMinutes = (uSeconds % 3600U) / 60U;
   uSeconds %= 60U;
   return _snprintf(pszBuffer, uMaxLen, "%02u:%02u:%02u", uHours, uMinutes, uSeconds);
}
*/
//===============================================================================================
// FUNCTION: ABFU_SetABFString
// PURPOSE:  Fill a non zero-terminated string, padding with spaces if necessary.
//   
void WINAPI ABFU_SetABFString(LPSTR psDest, LPCSTR psSrce, int nMaxLength)
{
//   ARRAYASSERT(psDest, nMaxLength);
#if 0
   LPSZASSERT(psSrce);
#endif
   strncpy(psDest, psSrce, nMaxLength);

   int l = (int)strlen(psSrce);
   while (l < nMaxLength)
      psDest[l++] = ' ';
}

//===============================================================================================
// FUNCTION: ABFU_GetABFString
// PURPOSE:  Fill a zero-terminated string, from an ABF string, stripping spaces if necessary.
//   
void WINAPI ABFU_GetABFString(LPSTR psDest, int nMaxDest, LPCSTR psSrce, int nMaxSrce)
{
//   ARRAYASSERT(psDest, nMaxDest);
//   ARRAYASSERT(psSrce, nMaxSrce);

   // Skip any leading blank spaces.
   while (nMaxSrce > 0)
   {
      if (*psSrce!=' ')
         break;
      psSrce++;
      nMaxSrce--;
   }

   // copy to the limit of the destination or the source, whichever comes first.
   int l = min(nMaxDest-1, nMaxSrce);
   strncpy(psDest, psSrce, l);
   psDest[l] = '\0';

   // Zero out any trailing spaces.
   while (l > 0)
   {
      l--;
      if (psDest[l]!=' ')
         return;
      psDest[l] = '\0';
   }
}


//===============================================================================================
// FUNCTION: ABFU_FixFileStartDate
// PURPOSE:  Checks the lFileStartDate parameter contains a 4 digit year and fixes it if needed.
//
long WINAPI ABFU_FixFileStartDate( long lDate )
{

   long lStartDay   = lDate % 100L;
   long lStartMonth = (lDate % 10000L) / 100L;
   long lStartYear  = lDate / 10000L;
   if ( lStartYear < 1000)
   {
      if (lStartYear < 80L)
         lStartYear += 2000L;
      else
         lStartYear += 1900L;
   }

   return long(lStartYear*10000 + lStartMonth*100 + lStartDay);

}   
/*
//===============================================================================================
// FUNCTION: ABFU_IsValidSignalName
// PURPOSE:  Checks if the signal name is valid.
// RETURNS:  TRUE if name is valid.
//
BOOL WINAPI ABFU_IsValidSignalName( LPCSTR pszSignalName )
{
   LPSZASSERT( pszSignalName );

   // Signals cannot have the same name as the math channel.
   char szMathChannel[ ABF_ADCNAMELEN + 1 ];
   ABFH_GetMathChannelName( szMathChannel, sizeof( szMathChannel ) );
   if ( stricmp( pszSignalName, szMathChannel ) == 0 )
      return FALSE;

   if ( *pszSignalName == ' ' )
      return FALSE;

   if ( *pszSignalName == '\0' )
      return FALSE;

   // Check for invalid characters.
   int len = strlen( pszSignalName );
   int pos = strspn( pszSignalName, ABFU_GetValidSignalNameChars() );
   
   // If pos < len, then an invalid char was found.
   return (pos == len);
}

//===============================================================================================
// FUNCTION: ABFU_GetValidSignalNameChars
// PURPOSE:  Returns the set of valid characters for signal names.
//
LPCSTR WINAPI ABFU_GetValidSignalNameChars()
{
   
   return ABFU_VALID_SIG_CHARS;
}

//===============================================================================================
// FUNCTION: ABFU_FixSignalName
// PURPOSE:  Replaces invalid characters in a signal name with valid ones.
// RETURNS:  TRUE if name is valid.
//
void WINAPI ABFU_FixSignalName( LPSTR pszSignalName )
{
   LPSZASSERT( pszSignalName );

   // Check for invalid characters.
   int len = strlen( pszSignalName );
   for( int i=0; i<len; i++ )
   {
      char *pch = pszSignalName + i;
      int pos = strspn( pszSignalName + i, ABFU_GetValidSignalNameChars() );
      if( pos == 0 )
      {
         TRACE1( "%c found - replace !\n", pch);
         if( strncmp( pch, "-", 1 ) == 0 )
            strncpy( pch, "_", 1);
         else
            strncpy( pch, "#", 1);
      }
   }
   
}
*/
