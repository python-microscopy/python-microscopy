/****************************************************************************\
*                                                                            *
*   Written 1990 - 1995 by AXON Instruments Inc.                             *
*                                                                            *
*   This file is not protected by copyright. You are free to use, modify     *
*   and copy the code in this file.                                          *
*                                                                            *
\****************************************************************************/
//
// HEADER:  ATFUTIL.H    Prototypes for functions in ATFUTIL.CPP
// AUTHOR:  BHI  Feb 1995

#ifndef __ATFUTIL_H__
#define __ATFUTIL_H__

#ifndef _WINDOWS
int LoadString( HINSTANCE hInstance, int nErrorNum, char *sTxtBuf, UINT uMaxLen);
#endif

extern HINSTANCE g_hInstance;

#ifdef _WINDOWS
#include "./../Common/resource.h"
#endif

#endif
