//***********************************************************************************************
//
//    Copyright (c) 1993-1999 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// HEADER:  ABFUTIL.H    Prototypes for functions in ABFUTIL.CPP
// AUTHOR:  BHI  Feb 1995

#ifndef INC_ABFUTIL_H
#define INC_ABFUTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "AxAbffio32.h"
#include "abfheadr.h"

#ifdef _WINDOWS
UINT WINAPI ABFU_GetTempFileName(LPCSTR szPrefix, UINT uUnique, LPSTR lpTempName);
#endif

BOOL WINAPI ABFU_ReadFile(FILEHANDLE hFile, LPVOID lpBuf, DWORD dwBytesToRead);
/*
BOOL WINAPI ABFU_FormatDouble(double dNum, int nDigits, char *pszString, UINT uSize);

int  WINAPI ABFU_FormatHMS( UINT uSeconds, char *pszBuffer, UINT uMaxLen );
*/
void WINAPI ABFU_SetABFString(LPSTR psDest, LPCSTR psSrce, int nMaxLength);

void WINAPI ABFU_GetABFString(LPSTR psDest, int nMaxDest, LPCSTR psSrce, int nMaxSrce);
/*
BOOL WINAPI ABFU_IsValidSignalName(LPCSTR pszName);

LPCSTR WINAPI ABFU_GetValidSignalNameChars();

void WINAPI ABFU_FixSignalName( LPSTR pszSignalName );

*/
// Checks the date is Y2K compliant and fixes it if needed.
long WINAPI ABFU_FixFileStartDate( long lDate );


#define ABF_BLANK_FILL(d)       (memset((void *)(d), ' ', sizeof(d)))
#define ABF_SET_STRING(d, s)    (ABFU_SetABFString(d, s, sizeof(d)))
#define ABF_GET_STRING(d, s, n) (ABFU_GetABFString(d, n, s, sizeof(s)))
/*
// Assert that an ABF header is writeable - accounting for the header size.
inline void ABFH_WASSERT( ABFFileHeader * pFH )
{
#ifdef _DEBUG
   UINT uHeaderSize = ABFH_IsNewHeader(pFH) ? ABF_HEADERSIZE : ABF_OLDHEADERSIZE;
   ASSERT(pFH != NULL && !IsBadWritePtr( pFH, uHeaderSize ));
   //TRACE1( "ABF Header is writeable (%d bytes).\n", uHeaderSize );
#endif
}
*/
// Assert that an ABF header is readable - accounting for the header size.
inline void ABFH_ASSERT( const ABFFileHeader * pFH )
{
#ifdef _DEBUG
   UINT uHeaderSize = ABFH_IsNewHeader(pFH) ? ABF_HEADERSIZE : ABF_OLDHEADERSIZE;
   ASSERT(pFH != NULL && !IsBadReadPtr( pFH, uHeaderSize ));
   //TRACE1( "ABF Header is readable (%d bytes).\n", uHeaderSize );
#endif
}


#ifdef __cplusplus
}
#endif

#endif   // INC_ABFUTIL_H
