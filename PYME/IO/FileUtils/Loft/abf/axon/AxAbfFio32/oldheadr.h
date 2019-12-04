//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// HEADER:  OLDHEADR.H.
// PURPOSE: Provides prototypes for functions implemented in OLDHEADR.CPP for 
//          reading old ABF file header.

#ifndef __OLDHEADR_H__
#define __OLDHEADR_H__

BOOL OLDH_GetFileVersion( FILEHANDLE hFile, UINT *puFileType, float *pfFileVersion,
                          BOOL *pbMSBinFormat);

void OLDH_ABFtoCurrentVersion(ABFFileHeader *pFH);
/*
void OLDH_CorrectScopeConfig(ABFFileHeader *pFH, ABFScopeConfig *pCfg);
*/
BOOL OLDH_ReadOldHeader( FILEHANDLE hFile, UINT uFileType, int bMSBinFormat,
                         ABFFileHeader *pFH, long lFileLength, int *pnError);

#endif   /* __OLDHEADR_H__ */
