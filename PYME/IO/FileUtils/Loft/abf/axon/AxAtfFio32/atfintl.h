//***********************************************************************************************
//                                                                            
//   Written 1990 - 1996 by AXON Instruments Inc.                             
//                                                                            
//   This file is not protected by copyright. You are free to use, modify     
//   and copy the code in this file.                                          
//                                                                            
//***********************************************************************************************
//
// HEADER:  ATFINTL.H
// PURPOSE: Internal header file for the ATF file I/O routines.
// 

enum eFILE_STATE
{
   eCLOSED,
   eOPENED,
   eHEADERED,
   eDATAREAD,
   eDATAWRITTEN,
   eDATAAPPENDED,
};

struct ATF_FILEINFO
{
   HANDLE      hFile;
   eFILE_STATE eState;
   BOOL        bWriting;
   UINT        uFlags;
   int         nHeaders;
   long        lFilePos;
   int         nColumns;
   double      dFileVersion;
   long        lTitlesPtr;
   long        lDataPtr;
   BOOL        bDataOnLine;
   char        szSeparator[2];
   char **     apszFileColTitles;
   char **     apszFileColUnits;
   char *      pszIOBuffer;
   char *      pszFileName;
   int         nIOBufferSize;

   // buffering:
   long        lBufSize;
   char *      pszBuf;        // = lBufSize means no info is buffered
   long        lPos;          // current position in buffer - 0 means no info is buffered
   BOOL        bRead;         // TRUE = reading; FALSE = writing
   long        lBufReadLimit; // actual amount read in when a full buffer is not available
   char        cLineTerm;     // The character that will terminate each line.
};

typedef ATF_FILEINFO *PATF_FILEINFO;


//-----------------------------------------------------------------------------------------------
// Macros and functions to deal with returning error return codes through a pointer if given.

#define ERRORRETURN(p, e)  return ErrorReturn(p, e);
inline BOOL ErrorReturn(int *pnError, int nErrorNum)
{
   if (pnError)
      *pnError = nErrorNum;
   return FALSE;
}

//-----------------------------------------------------------------------------------------------

#define MAX_READ_SIZE   512
#define GETS_OK         0     // Success!
#define GETS_EOF        1     // End of file reached.
#define GETS_ERROR      2     // I/O error
#define GETS_NOEOL      3     // No end of line found.

//-----------------------------------------------------------------------------------------------
// declaration of low-level file I/O functions.

HANDLE CreateFileBuf(ATF_FILEINFO *pATF, DWORD dwDesiredAccess, DWORD dwShareMode,
                     LPSECURITY_ATTRIBUTES lpSecurityAttributes, 
                     DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, 
                     HANDLE hTemplateFile );
BOOL CloseHandleBuf(ATF_FILEINFO *pATF);
BOOL WriteFileBuf(ATF_FILEINFO *pATF, LPCVOID pvBuffer, DWORD dwBytes, DWORD *pdwWritten, LPOVERLAPPED lpOverlapped);
BOOL ReadFileBuf(ATF_FILEINFO *pATF, LPVOID pvBuffer, DWORD dwBytes, DWORD *pdwRead, LPOVERLAPPED lpOverlapped);
DWORD SetFilePointerBuf(ATF_FILEINFO *pATF, long lToMove, PLONG plDistHigh, DWORD dwMoveMethod);
int getsBuf(ATF_FILEINFO *pATF, LPSTR pszString, DWORD dwToRead);
int putsBuf(ATF_FILEINFO *pATF, LPCSTR pszString);
