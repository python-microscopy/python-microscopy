/****************************************************************************\
*                                                                            *
*   Written 1990 - 1996 by AXON Instruments Inc.                             *
*                                                                            *
*   This file is not protected by copyright. You are free to use, modify     *
*   and copy the code in this file.                                          *
*                                                                            *
\****************************************************************************/

#ifndef INC_AXATFFIO32_H
#define INC_AXATFFIO32_H

#include "../Common/wincpp.hpp"

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

#define     VAL_EXTERNBUFSIZE  31                  /* anybody calling methods using conversion */

// This is AXATFFIO32.H; a companion file to AXATFFIO32.CPP
#define ATF_CURRENTVERSION  1.0

// Length required for I/O buffers
#define ATF_MAXCOLUMNS    8000

// maximum size of read & write buffers (each can be this size)
#define ATF_MAX_BUFFER_SIZE   65536                    

// Flags that may be combined and passed in the wFlags to ATF_OpenFile call
#define ATF_WRITEONLY        0
#define ATF_READONLY         1
#define ATF_OVERWRTIFEXIST   2
#define ATF_APPENDIFEXIST    4
#define ATF_DONTWRITEHEADER  8

// Value returned as the file handle if the file could not be opened.
#define ATF_INVALID_HANDLE -1

// Definitions for error results returned by AXATFFIO32 module
#define ATF_SUCCESS              0
#define ATF_ERROR_NOFILE         1001
#define ATF_ERROR_TOOMANYFILES   1002
#define ATF_ERROR_FILEEXISTS     1003
#define ATF_ERROR_BADVERSION     1004
#define ATF_ERROR_BADFILENUM     1005
#define ATF_ERROR_BADSTATE       1006
#define ATF_ERROR_IOERROR        1007
#define ATF_ERROR_NOMORE         1008
#define ATF_ERROR_BADHEADER      1009
#define ATF_ERROR_NOMEMORY       1012
#define ATF_ERROR_TOOMANYCOLS    1013
#define ATF_ERROR_INVALIDFILE    1014
#define ATF_ERROR_BADCOLNUM      1015
#define ATF_ERROR_LINETOOLONG    1016
#define ATF_ERROR_BADFLTCNV      1017
#define ATF_ERROR_NOMESSAGESTR   2000

// These functions are not exported from the DLL version as they are called implicitly on load/unload.
BOOL WINAPI ATF_Initialize(HINSTANCE hDLL);
void WINAPI ATF_Cleanup(void);

//---------------------- Exported Function Definitions -------------------------

BOOL WINAPI ATF_OpenFile(LPCSTR szFileName, UINT uFlags, int *pnColumns, int *pnFile, int *pnError);

BOOL WINAPI ATF_CloseFile(int nFile);

BOOL WINAPI ATF_SetSeperator(int nFile, BOOL bUseCommas, int *pnError);

BOOL WINAPI ATF_IsAppending(int nFile);

BOOL WINAPI ATF_RewindFile(int nFile, int *pnError);

BOOL WINAPI ATF_CountDataLines(int nFile, long *plNumLines, int *pnError);

BOOL WINAPI ATF_GetNumHeaders(int nFile, int *pnHeaders, int *pnError);

BOOL WINAPI ATF_WriteHeaderRecord(int nFile, LPCSTR pszText, int *pnError);

BOOL WINAPI ATF_SetColumnTitle(int nFile, LPCSTR pszText, int *pnError);

BOOL WINAPI ATF_SetColumnUnits(int nFile, LPCSTR pszText, int *pnError);

BOOL WINAPI ATF_WriteEndOfLine(int nFile, int *pnError);

BOOL WINAPI ATF_WriteDataRecord(int nFile, LPCSTR pszText, int *pnError);

BOOL WINAPI ATF_WriteDataComment(int nFile, LPCSTR pszText, int *pnError);

BOOL WINAPI ATF_WriteDataRecordArray(int nFile, int nCount, double *pdVals, int *pnError);

BOOL WINAPI ATF_WriteDataRecordArrayFloat(int nFile, int nCount, float *pfVals, int *pnError);

BOOL WINAPI ATF_WriteDataRecord1(int nFile, double dNum1, int *pnError);

BOOL WINAPI ATF_WriteDataRecord1Float(int nFile, float fNum1, int *pnError);

BOOL WINAPI ATF_ReadHeaderLine(int nFile, char *psBuf, int nMaxLen, int *pnError);

BOOL WINAPI ATF_ReadHeaderNoQuotes(int nFile, char *psBuf, int nMaxLen, int *pnError);

BOOL WINAPI ATF_GetColumnTitle(int nFile, int nColumn, char *pszText, int nMaxTxt, int *pnError);

BOOL WINAPI ATF_GetColumnUnits(int nFile, int nColumn, char *pszText, int nMaxTxt, int *pnError);

BOOL WINAPI ATF_ReadDataRecord(int nFile, char *pszText, int nMaxLen, int *pnError);

BOOL WINAPI ATF_ReadDataRecordArray(int nFile, int nCount, double *pdVals,
                                    char *pszComment, int nMaxLen, int *pnError);

BOOL WINAPI ATF_ReadDataColumn(int nFile, int nColumn, double *pdVal, int *pnError);

int WINAPI ATF_BuildErrorText(int nErrorNum, LPCSTR szFileName, char *sTxtBuf, int nMaxLen);

BOOL WINAPI ATF_GetFileDateTime(int nFile, long *plDate, long *plTime, int *pnError);

#ifdef __cplusplus
}
#endif

#endif   /* INC_AXATFFIO32_H */
