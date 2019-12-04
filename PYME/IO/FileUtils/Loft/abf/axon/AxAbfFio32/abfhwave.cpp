//***********************************************************************************************
//
//    Copyright (c) 1993-2000 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// This is ABFHWAVE.CPP; a set of routines for returning the waveform from
//                       an ABF header definition.
// 
// An ANSI C compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL ABFFILES.C)

#include "../Common/wincpp.hpp"
#include "abfheadr.h"
#include <math.h>


#include "abfutil.h"
/*#include "./../Common/ArrayPtr.hpp"
#include "UserList.hpp"
#include "PopulateEpoch.hpp"
 
#define ERRORRETURN(p, e)  return ErrorReturn(p, e);
static BOOL ErrorReturn(int *pnError, int nErrorNum)
{
   if (pnError)
      *pnError = nErrorNum;
   return FALSE;
}
*/
//===============================================================================================
// FUNCTION: ABFH_GetChannelOffset
// PURPOSE:  Get the offset in the sampling sequence for the given physical channel.
//
BOOL WINAPI ABFH_GetChannelOffset(const ABFFileHeader *pFH, int nChannel, UINT *puChannelOffset)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(puChannelOffset);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   int nOffset;

   // check the ADC channel number, -1 refers to the math channel

   if (nChannel < 0)
   {
      if (!FH.nArithmeticEnable)
      {
         if (puChannelOffset)
            *puChannelOffset = 0;   // return the offset to this channel
         return FALSE;              // channel not found in sampling sequence
      }
      nChannel = FH.nArithmeticADCNumA;
   }

   for (nOffset = 0; nOffset < FH.nADCNumChannels; nOffset++)
   {
      if (FH.nADCSamplingSeq[nOffset] == nChannel)
      {
         if (puChannelOffset)
            *puChannelOffset = UINT(nOffset);  // return the offset to this channel
         return TRUE;
      }
   }

   if (puChannelOffset)
      *puChannelOffset = 0;  // return the offset to this channel
   return FALSE;
}
/*
//===============================================================================================
// FUNCTION: GetListEntry
// PURPOSE:  Gets the entry in the list that corresponds to the given episode number.
//
static LPSTR GetListEntry( const ABFFileHeader *pFH, UINT uListNum, UINT uEpisode, 
                           LPSTR szList, UINT uListSize )
{
   ABFH_ASSERT(pFH);
   ASSERT(uEpisode > 0);
   ASSERT(uListSize <= ABF_USERLISTLEN+1);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   // Turn the list field into an ASCIIZ string.

   strncpy(szList, FH.sULParamValueList[uListNum], uListSize-1);
   szList[uListSize-1] = '\0';

   char *psItem = strtok(szList, ",");
   if (!psItem)
   {
      szList[0] = '\0';
      psItem = szList;
   }
   else
   {
      for (UINT i=0; i<uEpisode-1; i++)
      {
         LPSTR psLast = psItem;
         psItem = strtok(NULL, ",");
         if (!psItem)
         {
            psItem = psLast;
            break;
         }
      }
   }
   return psItem;
}



//===============================================================================================
// FUNCTION: GetFloatEntry
// PURPOSE:  Gets the float entry in the list that corresponds to the given episode number.
//
static float GetFloatEntry(const ABFFileHeader *pFH, UINT uListNum, UINT uEpisode)
{
   char szList[ABF_USERLISTLEN+1];
   LPSTR pszItem = GetListEntry(pFH, uListNum, uEpisode, szList, sizeof(szList));
   return float(atof(pszItem));
}

//===============================================================================================
// FUNCTION: GetIntegerEntry
// PURPOSE:  Gets the integer entry in the list that corresponds to the given episode number.
//
static int GetIntegerEntry(const ABFFileHeader *pFH, UINT uListNum, UINT uEpisode)
{
   char szList[ABF_USERLISTLEN+1];
   LPSTR pszItem = GetListEntry(pFH, uListNum, uEpisode, szList, sizeof(szList));
   return atoi(pszItem);
}

//===============================================================================================
// FUNCTION: GetBinaryEntry
// PURPOSE:  Gets the binary entry in the list that corresponds to the given episode number.
//
static int GetBinaryEntry(const ABFFileHeader *pFH, UINT uListNum, UINT uEpisode)
{
   char szList[ABF_USERLISTLEN+1];
   LPSTR pszItem = GetListEntry(pFH, uListNum, uEpisode, szList, sizeof(szList));
   return strtoul(pszItem, NULL, 2);
}

//===============================================================================================
// FUNCTION: ABFH_GetEpochDuration
// PURPOSE:  Get the duration of an Epoch that corresponds to the given episode number.
//
int WINAPI ABFH_GetEpochDuration(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDACChannel] &&
      (FH.nActiveDACChannel == (int)uDACChannel) &&       // User list must be active channel at present.
      (FH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITDURATION) &&
      (nEpoch == FH.nULParamToVary[uDACChannel] - ABF_EPOCHINITDURATION))
      return GetIntegerEntry(pFH, uDACChannel, uEpisode);

   return FH.lEpochInitDuration[uDACChannel][nEpoch] + 
          (int)(uEpisode-1) * FH.lEpochDurationInc[uDACChannel][nEpoch];
}

//===============================================================================================
// FUNCTION: ABFH_GetEpochLevel
// PURPOSE:  Get the level of an Epoch that corresponds to the given episode number.
//
float WINAPI ABFH_GetEpochLevel(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDACChannel] &&
      (FH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITLEVEL) &&
      (FH.nULParamToVary[uDACChannel] < ABF_EPOCHINITDURATION) &&
      (nEpoch == FH.nULParamToVary[uDACChannel] - ABF_EPOCHINITLEVEL))
      return GetFloatEntry(pFH, uDACChannel, uEpisode);
      
   return FH.fEpochInitLevel[uDACChannel][nEpoch] + 
         (uEpisode-1) * FH.fEpochLevelInc[uDACChannel][nEpoch];
}

BOOL WINAPI ABFH_GetEpochLevelRange(const ABFFileHeader *pFH, UINT uDACChannel, int nEpoch, float *pfMin, float *pfMax)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nEpochType[uDACChannel][nEpoch] == ABF_EPOCHDISABLED )
   {
      return FALSE;
   }

   if (FH.nULEnable[uDACChannel] &&
      (FH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITLEVEL) &&
      (FH.nULParamToVary[uDACChannel] < ABF_EPOCHINITDURATION) &&
      (nEpoch == FH.nULParamToVary[uDACChannel] - ABF_EPOCHINITLEVEL))
   {
      *pfMin = 1e9;
      *pfMax = -1e9;
      // Iterate over all possible levels.
      // If lEpisodesPerRun less than number of user list values then only
      // check up to lEpisodesPerRun user list values.
      CUserList UserList;
      UserList.Initialize( &FH, uDACChannel );
      UINT uListEntries = UserList.GetNumEntries();
      int nCount = min( FH.lEpisodesPerRun, (int)uListEntries );
      for ( int n = 0; n<nCount; n++ )
      {
         float fLevel = GetFloatEntry( pFH, uDACChannel, n+1 );
         *pfMin = min( *pfMin, fLevel );
         *pfMax = max( *pfMax, fLevel );
      }
      return TRUE;
   }
   else
   {
      float fEpochFinalLevel = FH.fEpochInitLevel[uDACChannel][nEpoch] + 
         max(0,FH.lEpisodesPerRun-1) * FH.fEpochLevelInc[uDACChannel][nEpoch];
      
      *pfMin = min( FH.fEpochInitLevel[uDACChannel][nEpoch], fEpochFinalLevel );
      *pfMax = max( FH.fEpochInitLevel[uDACChannel][nEpoch], fEpochFinalLevel );
      return TRUE;
   }
}

UINT WINAPI ABFH_GetMaxPNSubsweeps(const ABFFileHeader *pFH, UINT uDACChannel)
{
   ABFH_ASSERT(pFH);

   int nMaxPNSubSweeps = 0;

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if ( FH.nPNEnable )
   {
      if (FH.nULEnable[uDACChannel] &&
         (FH.nULParamToVary[uDACChannel] == ABF_PNNUMPULSES) )
      {
         // Iterate over all possible P/N subsweep counts.
         // If lEpisodesPerRun less than number of user list values then only
         // check up to lEpisodesPerRun user list values.
         CUserList UserList;
         UserList.Initialize( &FH, uDACChannel );
         UINT uListEntries = UserList.GetNumEntries();
         int nCount = min( FH.lEpisodesPerRun, (int)uListEntries );
         for ( int n = 0; n<nCount; n++ )
         {
            nMaxPNSubSweeps = max( nMaxPNSubSweeps, GetIntegerEntry( pFH, uDACChannel, n+1 ) );
         }
      }
      else
      {
         nMaxPNSubSweeps = FH.nPNNumPulses;
      }
   }

   return nMaxPNSubSweeps;
}

//===============================================================================================
// FUNCTION: GetDigitalEpochLevel
// PURPOSE:  Get the level of a digital Epoch that corresponds to the given episode number.
//
static DWORD GetDigitalEpochLevel(const ABFFileHeader *pFH, UINT uEpisode, int nEpoch, UINT uDigitalChannel)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   UINT uListNum = FH.nActiveDACChannel;
   if (FH.nULEnable[uListNum] &&
      (FH.nULParamToVary[uListNum] >= ABF_PARALLELVALUE) &&
      (FH.nULParamToVary[uListNum] < ABF_EPOCHINITLEVEL) &&
      (nEpoch == FH.nULParamToVary[uListNum] - ABF_PARALLELVALUE))
      {
         CUserList UL;
         UL.Initialize( &FH, uListNum );

         UserListItem ULitem = UL.GetItem( uEpisode, NULL );

         return ULitem.n;
      }

   if( uDigitalChannel == (UINT)FH.nActiveDACChannel )
      return FH.nDigitalValue[nEpoch];
   else
      return FH.nAlternateDigitalValue[nEpoch];
}

//===============================================================================================
// FUNCTION: GetDigitalTrainEpochLevel
// PURPOSE:  Get the level of a digital Epoch that corresponds to the given episode number.
//
static DWORD GetDigitalTrainEpochLevel(const ABFFileHeader *pFH, UINT uEpisode, int nEpoch, UINT uDigitalChannel)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   UINT uListNum = FH.nActiveDACChannel;

   if (FH.nULEnable[uListNum] &&
      (FH.nULParamToVary[uListNum] >= ABF_PARALLELVALUE) &&
      (FH.nULParamToVary[uListNum] < ABF_EPOCHINITLEVEL) &&
      (nEpoch == FH.nULParamToVary[uListNum] - ABF_PARALLELVALUE))
      {
         CUserList UL;
         UL.Initialize( &FH, uListNum );

         UserListItem ULitem = UL.GetDigitalTrainItem( uEpisode, NULL );

         return ULitem.n;
      }

   if( uDigitalChannel == (UINT)FH.nActiveDACChannel )
      return FH.nDigitalTrainValue[nEpoch];
   else
      return FH.nAlternateDigitalTrainValue[nEpoch];
}

//===============================================================================================
// FUNCTION: GetPostTrainPeriod
// PURPOSE:  Get the post train period.
//
static float GetPostTrainPeriod(const ABFFileHeader *pFH, UINT uDAC, UINT uEpisode)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDAC < ABF_WAVEFORMCOUNT );

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDAC] &&
      (FH.nULParamToVary[uDAC] == ABF_CONDITPOSTTRAINDURATION))
      return GetFloatEntry(pFH, uDAC, uEpisode);
   
   return FH.fPostTrainPeriod[uDAC];
}

//===============================================================================================
// FUNCTION: GetPostTrainLevel
// PURPOSE:  Get the post train level.
//
static float GetPostTrainLevel(const ABFFileHeader *pFH, UINT uDAC, UINT uEpisode)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDAC < ABF_WAVEFORMCOUNT );

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDAC] &&
      (FH.nULParamToVary[uDAC] == ABF_CONDITPOSTTRAINLEVEL))
      return GetFloatEntry(pFH, uDAC, uEpisode);
   
   return FH.fPostTrainLevel[uDAC];
}

//===============================================================================================
// FUNCTION: GetEpochTrainPeriod
// PURPOSE:  Get the train period of an Epoch that corresponds to the given episode number.
//
static int GetEpochTrainPeriod(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDACChannel] &&
      (FH.nActiveDACChannel == (int)uDACChannel) &&       // User list must be active channel at present.
      (FH.nULParamToVary[uDACChannel] >= ABF_EPOCHTRAINPERIOD) &&
      (FH.nULParamToVary[uDACChannel] < ABF_EPOCHTRAINPULSEWIDTH) &&
      (nEpoch == FH.nULParamToVary[uDACChannel] - ABF_EPOCHTRAINPERIOD))
   {
      CUserList UL;
      UL.Initialize( &FH, uDACChannel );    
      return UL.GetItem( uEpisode, NULL ).n;
   }

   return (int)FH.lEpochPulsePeriod[uDACChannel][nEpoch];
}

//===============================================================================================
// FUNCTION: GetEpochTrainPulseWidth
// PURPOSE:  Get the train pulse width of an Epoch that corresponds to the given episode number.
//
static int GetEpochTrainPulseWidth(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nEpoch)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   if (FH.nULEnable[uDACChannel] &&
      (FH.nActiveDACChannel == (int)uDACChannel) &&       // User list must be active channel at present.
      (FH.nULParamToVary[uDACChannel] >= ABF_EPOCHTRAINPULSEWIDTH) &&
      (FH.nULParamToVary[uDACChannel] < ABF_EPOCHTRAINPULSEWIDTH + ABF_EPOCHCOUNT) &&
      (nEpoch == FH.nULParamToVary[uDACChannel] - ABF_EPOCHTRAINPULSEWIDTH))
   {
      CUserList UL;
      UL.Initialize( &FH, uDACChannel );    
      return UL.GetItem( uEpisode, NULL ).n;
   }

   return (int)FH.lEpochPulseWidth[uDACChannel][nEpoch];
}

//===============================================================================================
// FUNCTION: GetHoldingLength
// PURPOSE:  Get the duration of the first/last holding period.
//
static int _GetHoldingLength(int nSweepLength, int nNumChannels)
{
   ASSERT((nSweepLength % nNumChannels)==0);

   // Calculate holding count.
   int nHoldingCount = nSweepLength / ABFH_HOLDINGFRACTION;

   // Round down to nearest sequence length.
   nHoldingCount -= nHoldingCount % nNumChannels;

   // If less than one sequence, round up to one sequence.
   if (nHoldingCount < nNumChannels)
      nHoldingCount = nNumChannels;

   return nHoldingCount;
}

//===============================================================================================
// FUNCTION: ABFH_GetHoldingDuration
// PURPOSE:  Get the duration of the first holding period.
//
UINT WINAPI ABFH_GetHoldingDuration(const ABFFileHeader *pFH)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   // Only waveform files have first/last holding points.
   if( !FH.nWaveformEnable[0] &&
       !FH.nWaveformEnable[1] &&
       (FH.nWaveformSource[0] == ABF_WAVEFORMDISABLED) &&
       (FH.nWaveformSource[1] == ABF_WAVEFORMDISABLED) &&
       !FH.nDigitalEnable )
      return 0;

   // Old data files used a different number of holding points.
   if (FH.nFileType == ABF_CLAMPEX)
      return 6 * (UINT)(FH.lNumSamplesPerEpisode / 512L);

   return _GetHoldingLength(FH.lNumSamplesPerEpisode, FH.nADCNumChannels);
}

//===============================================================================================
// FUNCTION: ABFH_SweepLenFromUserLen
// PURPOSE:  Get the full sweep length given the length available to epochs.
//
int WINAPI ABFH_SweepLenFromUserLen(int nUserLength, int nNumChannels)
{
   ASSERT((nUserLength % nNumChannels)==0);

   // UserLen = SweepLen - 2 * HoldingLen
   // where HoldingLen = SweepLen/64 rounded to nearest sequence.

   // => UserLen = SweepLen * (1 - 2/64)
   // => UserLen = SweepLen * (31/32)
   // => SweepLen = UserLen * (32/31)
   // But this may not be exact because of the rounding of the holding level,
   // hence the iterative solution below:

   // We will start with the user length and keep adding sequence lengths until we get to 
   // the sweep length that corresponds.

   int nSweepLength = nUserLength;
   while (ABFH_UserLenFromSweepLen(nSweepLength, nNumChannels) < nUserLength)
      nSweepLength += nNumChannels;

   return nSweepLength;
}

//===============================================================================================
// FUNCTION: ABFH_UserLenFromSweepLen
// PURPOSE:  Get the length available to epochs given the full sweep length.
//
int WINAPI ABFH_UserLenFromSweepLen(int nSweepLength, int nNumChannels)
{
   ASSERT((nSweepLength % nNumChannels)==0);

   // UserLen = SweepLen - 2 * HoldingLen
   // where HoldingLen = SweepLen/64 rounded to nearest sequence.

   // Calculate holding count.
   int nHoldingCount = _GetHoldingLength(nSweepLength, nNumChannels);

   ASSERT(nSweepLength > nHoldingCount * 2);
   return nSweepLength - nHoldingCount * 2;
}

//===============================================================================================
// FUNCTION: GetHoldingLevel
// PURPOSE:  Calculate the holding level for this episode, allowing for user lists, 
//           presweep pulses and "use last epoch" holding strategies.
//
static float GetHoldingLevel(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   float fCurrentHolding = FH.fDACHoldingLevel[uDACChannel];

   if (FH.nConditEnable[uDACChannel])
   {
      if (GetPostTrainPeriod(pFH, uDACChannel, uEpisode) > 0.0F)
         return GetPostTrainLevel(pFH, uDACChannel, uEpisode);
      else 
         return fCurrentHolding;
   }

   if (!FH.nInterEpisodeLevel[uDACChannel])
      return fCurrentHolding;

   int i;
   for (i=ABF_EPOCHCOUNT-1; i>=0; i--)
      if (FH.nEpochType[uDACChannel][i])
         break;
   if ((i < 0) || (uEpisode < 2))
      return fCurrentHolding;

   return ABFH_GetEpochLevel(pFH, uDACChannel, uEpisode-1, i);
}

//===============================================================================================
// FUNCTION: GetDigitalHoldingLevel
// PURPOSE:  Calculate the digital holding level for this episode, allowing for user lists, 
//           and "use last epoch" holding strategies.
//
static UINT GetDigitalHoldingLevel(const ABFFileHeader *pFH, UINT uEpisode)
{
   ABFH_ASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   UINT uCurrentHolding = FH.nDigitalHolding;
   UINT uListNum = FH.nActiveDACChannel;
   if (FH.nULEnable[uListNum] &&
      (FH.nULParamToVary[uListNum] == ABF_DIGITALHOLDING))
      uCurrentHolding = GetBinaryEntry(pFH, uListNum, uEpisode);

   if (!FH.nDigitalInterEpisode)
      return uCurrentHolding;
   int i=0;
   for (i=ABF_EPOCHCOUNT-1; i>=0; i--)
      if (FH.nEpochType[uListNum][i])
         break;
   if ((i < 0) || (uEpisode < 2))
      return uCurrentHolding;

   return GetDigitalEpochLevel(pFH, uEpisode-1, i, FH.nActiveDACChannel);
}

//===============================================================================================
// FUNCTION: ABFH_GetEpochLimits
// PURPOSE:  Return the bounds of a given epoch in a given episode.
//           Values returned are ZERO relative.
//
BOOL WINAPI ABFH_GetEpochLimits(const ABFFileHeader *pFH, int nADCChannel, DWORD dwEpisode, 
                                int nEpoch, UINT *puEpochStart, UINT *puEpochEnd, int *pnError)
{
   return ABFH_GetEpochLimitsEx(pFH, nADCChannel, pFH->nActiveDACChannel, dwEpisode, 
                                nEpoch, puEpochStart, puEpochEnd, pnError);
}

BOOL WINAPI ABFH_GetEpochLimitsEx(const ABFFileHeader *pFH, int nADCChannel, UINT uDACChannel, DWORD dwEpisode, 
                                int nEpoch, UINT *puEpochStart, UINT *puEpochEnd, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(puEpochStart);
   WPTRASSERT(puEpochEnd);
   ASSERT(dwEpisode > 0);

   if( pFH->nOperationMode != ABF_WAVEFORMFILE )
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   if ((nADCChannel < 0) && (NewFH.nArithmeticEnable != 0))
      nADCChannel = NewFH.nArithmeticADCNumA;

   UINT uChannelOffset;
   if (!ABFH_GetChannelOffset(&NewFH, nADCChannel, &uChannelOffset))
      ERRORRETURN(pnError, ABFH_CHANNELNOTSAMPLED);

   if (NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED)
      ERRORRETURN(pnError, ABFH_EPOCHNOTPRESENT);

   UINT uHoldingDuration = ABFH_GetHoldingDuration(&NewFH);
   int nEpochStart, nEpochEnd;

   if (nEpoch == ABFH_FIRSTHOLDING)
   {
      if (uChannelOffset >= uHoldingDuration)
         ERRORRETURN(pnError, ABFH_EPOCHNOTPRESENT);

      nEpochStart = 0;
      nEpochEnd = uHoldingDuration - 1;
   }
   else
   {
      if ((nEpoch!=ABFH_LASTHOLDING) && (!NewFH.nEpochType[uDACChannel][nEpoch]))
         ERRORRETURN(pnError, ABFH_EPOCHNOTPRESENT);

      nEpochStart = uHoldingDuration;
      int nEpochDuration = 0;

      for (int i=0; i<=nEpoch; i++)
      {
         if (!NewFH.nEpochType[uDACChannel][i])
         {
            nEpochDuration = 0;
            continue;
         }

         nEpochDuration = ABFH_GetEpochDuration(&NewFH, uDACChannel, dwEpisode, i);
         nEpochDuration *= NewFH.nADCNumChannels;
         nEpochDuration = max(nEpochDuration, 0);
         if (i == nEpoch)
            break;

         nEpochStart += nEpochDuration;
      }

      if (nEpoch == ABFH_LASTHOLDING)
         nEpochEnd = (UINT)NewFH.lNumSamplesPerEpisode - 1;
      else if( nEpochDuration > 0 )
         nEpochEnd = nEpochStart + nEpochDuration - 1;
      else
         nEpochEnd = nEpochStart;
   }

   *puEpochStart = (UINT)(nEpochStart / NewFH.nADCNumChannels);
   *puEpochEnd   = (UINT)(nEpochEnd / NewFH.nADCNumChannels);

   if (*puEpochEnd < *puEpochStart)
      ERRORRETURN(pnError, ABFH_EPOCHNOTPRESENT);

   return TRUE;
}

//===============================================================================================
// FUNCTION: GenerateWaveform
// PURPOSE:  Build the waveform as an array of UU floats.
//
static BOOL GenerateWaveform(const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, float *pfBuffer)
{
   ABFH_ASSERT(pFH);
   WARRAYASSERT(pfBuffer, (UINT)pFH->lNumSamplesPerEpisode);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   UINT   uHoldingDuration = ABFH_GetHoldingDuration(pFH);
   double dHoldingLevel    = GetHoldingLevel(pFH, uDACChannel, uEpisode);
   double dLevel           = dHoldingLevel;
   double dStartLevel      = dHoldingLevel;
   int    nDuration        = 0;
   bool   bSetToHolding    = false;

   // Set every other episode to holding if we are in alternating DAC episodic mode
   if (FH.nAlternateDACOutputState != 0)
   {
      if( uEpisode % 2 == 0 && uDACChannel == 0 )
         bSetToHolding = true;
      else if( uEpisode % 2 != 0 && uDACChannel == 1 )
         bSetToHolding = true;
   }

   // If this sweep is not included in protocol (i.e. sweeps 
   // added in off-line analysis), simply fill with holding level.
   if( uEpisode > (UINT)FH.lEpisodesPerRun )
   {
      for( int nSample = 0; nSample < FH.lNumSamplesPerEpisode; nSample++ )
         *pfBuffer++ = (float)dHoldingLevel;
      return TRUE;
   }

   // Populate the first holding with the holding level
   int nPoints = uHoldingDuration;
   for (int nSample = 0; nSample < (int)uHoldingDuration; nSample++)
      *pfBuffer++ = (float)dHoldingLevel;
   int nEndHolding = (UINT)FH.lNumSamplesPerEpisode - uHoldingDuration;

   // Reconstruct all epochs
   for (int nEpoch = 0; nEpoch < ABF_EPOCHCOUNT; nEpoch++)
   {
      if (FH.nEpochType[uDACChannel][nEpoch] == ABF_EPOCHDISABLED )
         continue;

      nDuration   = ABFH_GetEpochDuration(pFH, uDACChannel, uEpisode, nEpoch) * FH.nADCNumChannels;

      if ( nDuration <= 0 )
         continue;

      dLevel      = ABFH_GetEpochLevel(pFH, uDACChannel, uEpisode, nEpoch);
      int nPeriod = GetEpochTrainPeriod( pFH, uDACChannel, uEpisode, nEpoch ) * FH.nADCNumChannels;
      int nWidth  = GetEpochTrainPulseWidth(  pFH, uDACChannel, uEpisode, nEpoch ) * FH.nADCNumChannels;
   
      nPoints += nDuration;
      if( nPoints > (int)FH.lNumSamplesPerEpisode )
         return FALSE;

      // Allocate doubles to build the waveform
      double * pdValue = new double[nDuration];
      WARRAYASSERT(pdValue, (UINT)nDuration);

      // Set to holding if alternating
      if( bSetToHolding )
      {
         PopulateStep( nDuration, dHoldingLevel, pdValue );
      }
      else
      {
         // Build the appropriate waveform
         switch (FH.nEpochType[uDACChannel][nEpoch]) 
         {
         case ABF_EPOCHDISABLED:
            break;
         case ABF_EPOCHSTEPPED:
            PopulateStep( nDuration, dLevel, pdValue );
            break;
         case ABF_EPOCHRAMPED:
            PopulateRamp( nDuration, dStartLevel, dLevel, pdValue );
            break;
         case ABF_EPOCH_TYPE_RECTANGLE:
            PopulateRectangle( nDuration, dStartLevel, dLevel, nPeriod, nWidth, pdValue );
            break;
         case ABF_EPOCH_TYPE_BIPHASIC:
            PopulateBiphasic( nDuration, dStartLevel, dLevel, nPeriod, nWidth, pdValue );
            break;
         case ABF_EPOCH_TYPE_TRIANGLE:
            PopulateTriangle( nDuration, dStartLevel, dLevel, nPeriod, nWidth, pdValue );
            break;
         case ABF_EPOCH_TYPE_COSINE:
            PopulateCosine( nDuration, dStartLevel, dLevel, nPeriod, pdValue );
            break;
         case ABF_EPOCH_TYPE_RESISTANCE:
            PopulateResistance( nDuration, dLevel, dHoldingLevel, pdValue );
            break;
         }
      }

      // Fill buffer with floats (from double)
      for ( int nSample = 0; nSample < nDuration; nSample++)
         *pfBuffer++ = (float)pdValue[nSample];
      delete [] pdValue;

      dStartLevel    = dLevel;
      nEndHolding   -= nDuration;
   }

   if (!FH.nInterEpisodeLevel[uDACChannel])
      dStartLevel = FH.fDACHoldingLevel[uDACChannel];

   nPoints += nEndHolding;
   if( nPoints > (int)FH.lNumSamplesPerEpisode )
      return FALSE;

   // Populate the last holding with the holding level
   for(int nSample = 0; nSample < nEndHolding; nSample++)
      *pfBuffer++ = (float)dStartLevel;

   return TRUE;
}

//===============================================================================================
// FUNCTION: GenerateDigitalWaveform
// PURPOSE:  Build the waveform as an array of DWORDs.
//
static void GenerateDigitalWaveform(const ABFFileHeader *pFH, UINT uEpisode, DWORD *pdwBuffer)
{
   ABFH_ASSERT(pFH);
   ARRAYASSERT(pdwBuffer, (UINT)pFH->lNumSamplesPerEpisode);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   UINT uHoldingDuration = ABFH_GetHoldingDuration(pFH);
   UINT uHoldingLevel    = GetDigitalHoldingLevel(pFH, uEpisode);
   UINT uEpochLevel      = uHoldingLevel;
   UINT uDigitalTrainLevel = uHoldingLevel;

   long nPeriod         = 0;
   long nWidth          = 0;
   long nEndOfLastPulse = 0;

   // If this sweep is not included in protocol (i.e. sweeps added in off-line analysis), 
   // simply fill with holding level.
   if( uEpisode > (UINT)FH.lEpisodesPerRun )
   {
      for(int j=0; j<FH.lNumSamplesPerEpisode; j++ )
         *pdwBuffer++ = uHoldingLevel;
      return;
   }

   for (UINT i=0; i<uHoldingDuration; i++)
      *pdwBuffer++ = uHoldingLevel;

   UINT uEndHolding = UINT(FH.lNumSamplesPerEpisode) - uHoldingDuration;

   // Select source of digital channel if we are in alternating mode 
   // otherwise the digital channel is always taken from the active DAC
   UINT uDigitalChannel = FH.nActiveDACChannel;
   if( FH.nAlternateDigitalOutputState )
   {
      if( uEpisode % 2 != 0 )
         uDigitalChannel = 0;
      else
         uDigitalChannel = 1;
   }


   for( UINT uEpoch = 0; uEpoch < ABF_EPOCHCOUNT; uEpoch++)
   {
      if (!FH.nEpochType[uDigitalChannel][uEpoch] )
         continue;

      UINT uEpochDuration = ABFH_GetEpochDuration(pFH, uDigitalChannel, uEpisode, uEpoch);
      uEpochDuration *= FH.nADCNumChannels;

      uEpochLevel        = GetDigitalEpochLevel(pFH, uEpisode, uEpoch, uDigitalChannel);
      uDigitalTrainLevel = GetDigitalTrainEpochLevel( pFH, uEpisode, uEpoch, uDigitalChannel);

      // Determine if digital wave should have trains.
      BOOL bDigitalTrains = FH.nEpochType[uDigitalChannel][uEpoch] > 0 &&  uDigitalTrainLevel > 0 && uEpochDuration > 0;

      if( bDigitalTrains )
      {
         nPeriod = GetEpochTrainPeriod( pFH, uDigitalChannel, uEpisode, uEpoch ) * FH.nADCNumChannels;
         nWidth  = GetEpochTrainPulseWidth(  pFH, uDigitalChannel, uEpisode, uEpoch ) * FH.nADCNumChannels;

         nEndOfLastPulse = uEpochDuration;

         // This if statement should only be accessed if digital trains are enabled.
         // nPeriod == 0 is not valid.
         ASSERT( nPeriod > 0 );

         long nSample = 0;

         // Fill as many pulses as we can
         for ( nSample = 0; nSample < nEndOfLastPulse; ++nSample) 
         {
            DWORD   dwValue;
            if ((nSample % nPeriod) < nWidth) 
            {
               // Invert the logic of the digital train if the Active High Logic is not selected,
               if( FH.nDigitalTrainActiveLogic )
                  dwValue = uDigitalTrainLevel | uEpochLevel;
               else
                  dwValue = uHoldingLevel | uEpochLevel;
            }
            else 
            {
               // Invert the logic of the digital train if the Active High Logic is not selected,
               if( FH.nDigitalTrainActiveLogic )
                  dwValue = uHoldingLevel | uEpochLevel;
               else
                  dwValue = uDigitalTrainLevel | uEpochLevel;
            }

            *pdwBuffer++ = dwValue;
         }

         uEndHolding -= uEpochDuration;
      }
      else
      {
         for (UINT j=0; j<uEpochDuration; j++)
            *pdwBuffer++ = uEpochLevel;

         uEndHolding -= uEpochDuration;     
      }
   }

   if (FH.nDigitalInterEpisode)
      uHoldingLevel = uEpochLevel;

   for (int i=0; i<uEndHolding; i++)
      *pdwBuffer++ = uHoldingLevel;
}

//===============================================================================================
// FUNCTION: GetChannelEntries
// PURPOSE:  Get the entries that correspond to a particular channel 
//           in the multiplexed data stream.
//
static void GetChannelEntries(const ABFFileHeader *pFH, UINT uChannelOffset, void *pvDestBuffer, 
                              void *pvSourceBuffer, UINT uElementSize)
{
   ABFH_ASSERT(pFH);
   BYTE *pbySrce = (BYTE *)pvSourceBuffer;
   BYTE *pbyDest = (BYTE *)pvDestBuffer;

   ARRAYASSERT(pbySrce, (UINT)pFH->lNumSamplesPerEpisode*uElementSize);
   ARRAYASSERT(pbyDest, (UINT)(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels)*uElementSize);
   UINT uNumSamples = (UINT)pFH->lNumSamplesPerEpisode;
   UINT uSkip = pFH->nADCNumChannels;

   // Set the source pointer to the first item.
   pbySrce += (uChannelOffset * uElementSize);

   // Loop through the source array picking out the items that
   // correspond to the given channel.
   for (UINT i=uChannelOffset; i<uNumSamples; i+=uSkip)
   {
      memcpy(pbyDest, pbySrce, uElementSize);
      pbyDest  += uElementSize;
      pbySrce += uSkip * uElementSize;
   }
}

//===============================================================================================
// FUNCTION: ABFH_GetWaveform
// PURPOSE:  This function forms the de-multiplexed DAC output waveform for the
//           particular channel in the pfBuffer, in DAC UserUnits.
//
// The required size of the passed buffer is:
// pfBuffer     -> FH.lNumSamplesPerEpisode / FH.nADCNumChannels (floats)
//
BOOL WINAPI ABFH_GetWaveform( const ABFFileHeader *pFH, int nADCChannel, DWORD dwEpisode, 
                              float *pfBuffer, int *pnError)
{
   // Note:  we now ignore the nADCChannel parameter. 
   return ABFH_GetWaveformEx( pFH, pFH->nActiveDACChannel, dwEpisode, 
                              pfBuffer, pnError);
}


BOOL WINAPI ABFH_GetWaveformEx( const ABFFileHeader *pFH, UINT uDACChannel, DWORD dwEpisode, 
                              float *pfBuffer, int *pnError)
{
   // Check that the buffer is as large as it should be.
   ARRAYASSERT(pfBuffer, UINT(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels));
   
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   if (dwEpisode > (DWORD)NewFH.lActualEpisodes)
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);

   if( !NewFH.nWaveformEnable[uDACChannel] ||
       (NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED) )
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);
   
   if (NewFH.nWaveformSource[uDACChannel] == ABF_DACFILEWAVEFORM)
      ERRORRETURN(pnError, ABFH_EDACFILEWAVEFORM);
   
   if (NewFH.nADCNumChannels == 1)
   {
      if( !GenerateWaveform(&NewFH, uDACChannel, dwEpisode, pfBuffer) )
         ERRORRETURN(pnError, ABFH_EBADWAVEFORM);
   }
   else
   {
      CArrayPtr<float> pfWorkBuffer(NewFH.lNumSamplesPerEpisode);
      if (!pfWorkBuffer)
         ERRORRETURN(pnError, ABFH_ENOMEMORY);
      if( !GenerateWaveform(&NewFH, uDACChannel, dwEpisode, pfWorkBuffer) )
         ERRORRETURN(pnError, ABFH_EBADWAVEFORM);

      GetChannelEntries(&NewFH, 0, pfBuffer, 
                        pfWorkBuffer, sizeof(*pfBuffer));
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABFH_GetDigitalWaveform
// PURPOSE:  This function forms the de-multiplexed Digital output waveform for the
//           particular channel in the pdwBuffer, as a bit mask. Digital OUT 0 is in bit 0.
//
// The required size of the passed buffer is:
// pdwBuffer     -> FH.lNumSamplesPerEpisode / FH.nADCNumChannels (floats)
//
BOOL WINAPI ABFH_GetDigitalWaveform( const ABFFileHeader *pFH, int nChannel, DWORD dwEpisode, 
                                     DWORD *pdwBuffer, int *pnError)
{
   ABFH_ASSERT(pFH);

   // Check that the buffer is as large as it should be.
   ARRAYASSERT(pdwBuffer, UINT(pFH->lNumSamplesPerEpisode/pFH->nADCNumChannels));
   
   UINT uChannelOffset = 0;
   if (dwEpisode > (DWORD)pFH->lActualEpisodes)
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);

   if (!ABFH_GetChannelOffset(pFH, nChannel, &uChannelOffset))
      ERRORRETURN(pnError, ABFH_CHANNELNOTSAMPLED);

   if (pFH->nDigitalEnable == FALSE)
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);
   
   if (pFH->nADCNumChannels == 1)
      GenerateDigitalWaveform(pFH, dwEpisode, pdwBuffer);
   else
   {
      CArrayPtr<DWORD> pdwWorkBuffer(pFH->lNumSamplesPerEpisode);
      if (!pdwWorkBuffer)
         ERRORRETURN(pnError, ABFH_ENOMEMORY);
      GenerateDigitalWaveform(pFH, dwEpisode, pdwWorkBuffer);
      GetChannelEntries(pFH, uChannelOffset, pdwBuffer, 
                        pdwWorkBuffer, sizeof(*pdwBuffer));
   }
   return TRUE;
}

//===============================================================================================
// FUNCTION: ABFH_GetWaveformVector
// PURPOSE:  Returns vector pairs for displaying a waveform made up of epochs.
//           The clipping limits are PER CHANNEL sample numbers.
//
BOOL WINAPI ABFH_GetWaveformVector(const ABFFileHeader *pFH, UINT uDACChannel, DWORD dwEpisode, UINT uStart, 
                                   UINT uFinish, float *pfLevels, float *pfTimes,
                                   int *pnVectors, int *pnError)
{
   ABFH_ASSERT(pFH);
   ERRORMSG("This function has not been implemented as yet.");
   ERRORRETURN(pnError, ABFH_ENOWAVEFORM);
}

// This function is no longer supported.
#if 0
BOOL WINAPI ABFH_GetWaveformVector(const ABFFileHeader *pFH, DWORD dwEpisode, UINT uStart, 
                                   UINT uFinish, float *pfLevels, float *pfTimes,
                                   int *pnVectors, int *pnError)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pnVectors);
   ARRAYASSERT(pfLevels, ABFH_MAXVECTORS);
   ARRAYASSERT(pfTimes, ABFH_MAXVECTORS);
   
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   int i = 0;
   int j = 0;
   if( !NewFH.nWaveformEnable[uDACChannel] ||
       (NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED) || 
       (dwEpisode > (DWORD)NewFH.lActualEpisodes) )
      ERRORRETURN(pnError, ABFH_ENOWAVEFORM);
      
   if (NewFH.nWaveformSource[uDACChannel] != ABF_EPOCHTABLEWAVEFORM)
      ERRORRETURN(pnError, ABFH_EDACFILEWAVEFORM);

   // First build the waveform vectors in terms of samples.

   int  nHere = 0;
   float *pfL = pfLevels;
   float *pfT = pfTimes;

   UINT uHoldingDuration = ABFH_GetHoldingDuration(&NewFH);
   float fHoldingLevel = GetHoldingLevel(&NewFH, uDACChannel, dwEpisode);

   *pfL++ = fHoldingLevel;
   *pfT++ = (float)nHere;

   *pfL++ = fHoldingLevel;
   *pfT++ = (float)(nHere+=uHoldingDuration);

   int nEndHolding = (UINT)NewFH.lNumSamplesPerEpisode - uHoldingDuration;

   float fEpochLevel = 0.0F;
   for (i=0; i<ABF_EPOCHCOUNT; i++)
   {
      if (!NewFH.nEpochType[uDACChannel][i])
         continue;

      int nEpochDuration = GetEpochDuration(&NewFH, uDACChannel, dwEpisode, i);
      nEpochDuration *= NewFH.nADCNumChannels;

      if (nEpochDuration == 0)
         continue;

      fEpochLevel = GetEpochLevel(&NewFH, uDACChannel, dwEpisode, i);

      if ((NewFH.nEpochType[uDACChannel][i] == 1) &&    // epoch step away from prev level
         (fEpochLevel != *(pfL-1)))
      {
         *pfL++ = fEpochLevel;
         *pfT++ = (float)nHere;
      }
      *pfL++ = fEpochLevel;               // epoch finish level
      *pfT++ = (float)(nHere+=nEpochDuration);
      nEndHolding -= nEpochDuration;
   }

   if (!NewFH.nInterEpisodeLevel[uDACChannel])
      fHoldingLevel = NewFH.fDACHoldingLevel[uDACChannel];
   else
      fHoldingLevel = fEpochLevel;

   if (fHoldingLevel != *(pfL-1))
   {
      *pfL++ = fHoldingLevel;
      *pfT++ = (float)nHere;
   }
   *pfL++ = fHoldingLevel;
   *pfT++ = (float)(nHere+=nEndHolding);

   int nVectors = (int)(pfL - pfLevels);

   // Do clipping.

   uStart--;
   uFinish--;
   uStart  *= NewFH.nADCNumChannels;
   uFinish *= NewFH.nADCNumChannels;

   // Clip start

   for (i=0; i<nVectors; i++)
   {
      if (pfTimes[i] >= uStart)
         break;
   }

   if (i==nVectors)
   {
      *pnVectors = 0;
      return TRUE;
   }

   if (i > 0) 
   {
      float fLevel = (pfLevels[i] - pfLevels[i-1]) / (pfTimes[i] - pfTimes[i-1]);
      fLevel *= uStart - pfTimes[i];
      fLevel += pfLevels[i];
      pfL = pfLevels;
      pfT = pfTimes;

      i--;
      pfTimes[0]  = (float)uStart;
      pfLevels[0] = fLevel;
      for (j=1; j<nVectors-i; j++)
      {
         pfTimes[j]  = pfTimes[i+j];
         pfLevels[j] = pfLevels[i+j];
      }
      nVectors -= i;
   }

   // Clip finish

   for (i=0; i<nVectors; i++)
   {
      if (pfTimes[i] >= uFinish)
         break;
   }

   if (i<nVectors)
   {
      float fLevel = (pfLevels[i] - pfLevels[i-1]) / (pfTimes[i] - pfTimes[i-1]);
      fLevel *= uFinish - pfTimes[i];
      fLevel += pfLevels[i];

      pfTimes[i]  = (float)uFinish;
      pfLevels[i] = fLevel;
      nVectors = i+1;
   }

   UINT uClockChange = ABFH_GetClockChange(&NewFH);
   BOOL bSplitClock  = (uClockChange < UINT(NewFH.lNumSamplesPerEpisode));

   // If a vector passes through a split clock transition, add an extra point
   // at the transition.

   if (bSplitClock)
   {
      for (i=0; i<nVectors; i++)
      {
         if (pfTimes[i] > uClockChange)
            break;
      }
      if ((i < nVectors) && (pfTimes[i-1] < uClockChange))
      {
         float fLevel = (pfLevels[i] - pfLevels[i-1]) / (pfTimes[i] - pfTimes[i-1]);
         fLevel *= uClockChange - pfTimes[i];
         fLevel += pfLevels[i];

         for (j=nVectors; j>i; j--)
         {
            pfTimes[j]  = pfTimes[j-1];
            pfLevels[j] = pfLevels[j-1];
         }
         pfTimes[i]  = (float)uClockChange;
         pfLevels[i] = fLevel;
         nVectors++;
      }
   }

   // return the number of vectors generated.

   *pnVectors = nVectors;

   // Convert samples to time offsets.

   float fTimeInc = NewFH.fADCSampleInterval / 1E3F;
   for (i=0; i<nVectors; i++)
   {
      float fSample = pfTimes[i];

      if (bSplitClock && (fSample > uClockChange))
         break;
      pfTimes[i] = (fSample - uHoldingDuration) * fTimeInc;
   }

   if (i < nVectors)
   {
      float fSplitTime = (uClockChange - uHoldingDuration) * fTimeInc;
      fTimeInc = NewFH.fADCSecondSampleInterval / 1E3F;
      for (j=i; j<nVectors; j++)
         pfTimes[j] = fSplitTime + (pfTimes[j] - uClockChange) * fTimeInc;
   }

   return TRUE;
}
#endif

//===============================================================================================
// FUNCTION: ABFH_GetDigitalWaveformVector
// PURPOSE:  Returns vector pairs for displaying a waveform made up of epochs.
//           The clipping limits are PER CHANNEL sample numbers.
//
BOOL WINAPI ABFH_GetDigitalWaveformVector(const ABFFileHeader *pFH, DWORD dwEpisode, UINT uStart, 
                                          UINT uFinish, DWORD *pdwLevels, float *pfTimes,
                                          int *pnVectors, int *pnError)
{
   ABFH_ASSERT(pFH);
   ERRORMSG("This function has not been implemented as yet.");
   ERRORRETURN(pnError, ABFH_ENOWAVEFORM);
}
*/
//===============================================================================================
// FUNCTION: GetTimebase
// PURPOSE:  Calculates the timebase array for the file.
//
template <class T>
void GetTimebase(const ABFFileHeader *pFH, T TimeOffset, T *pBuffer, UINT uBufferSize);

template <class T>
void GetTimebase(const ABFFileHeader *pFH, T TimeOffset, T *pBuffer, UINT uBufferSize)
{
   ABFH_ASSERT(pFH);
   ARRAYASSERT(pBuffer, uBufferSize);
   UINT i, j;

   UINT uSamplesPerSweep = UINT(pFH->lNumSamplesPerEpisode);
   UINT uClockChange     = ABFH_GetClockChange(pFH);

   double dTime = (T)TimeOffset;
   double dTimeInc = ABFH_GetFirstSampleInterval(pFH) * pFH->nADCNumChannels;
   dTimeInc = dTimeInc / 1E3;
   for (i=0; i<uSamplesPerSweep; i+=pFH->nADCNumChannels)
   {
      if (i >= uClockChange)
         break;
      if (uBufferSize-- == 0)
         return;
      *pBuffer++ = (T)dTime;
      dTime += dTimeInc;
   }
   
   if (i < uSamplesPerSweep)
   {
      dTimeInc = ABFH_GetSecondSampleInterval(pFH) * pFH->nADCNumChannels;
      dTimeInc = dTimeInc / 1E3;
      for (j=i; j<uSamplesPerSweep; j+=pFH->nADCNumChannels)
      {
         if (uBufferSize-- == 0)
            return;
         *pBuffer++ = (T)dTime;
         dTime += dTimeInc;
      }
   }
}


//===============================================================================================
// FUNCTION: ABFH_GetTimebase
// PURPOSE:  Calculates the timebase array for the file.
//
void WINAPI ABFH_GetTimebase(const ABFFileHeader *pFH, float fTimeOffset, float *pfBuffer, UINT uBufferSize)
{
   GetTimebase( pFH, fTimeOffset, pfBuffer, uBufferSize );
}

//===============================================================================================
// FUNCTION: ABFH_GetTimebaseEx
// PURPOSE:  Calculates the timebase array (in doubles) for the file.
//
void WINAPI ABFH_GetTimebaseEx(const ABFFileHeader *pFH, double dTimeOffset, double *pdBuffer, UINT uBufferSize)
{
   GetTimebase( pFH, dTimeOffset, pdBuffer, uBufferSize );
}
/*
//==============================================================================================
// FUNCTION: ABFH_GetNumberOfChangingSweeps
// PURPOSE:  Count the number of changing sweeps.
//           In waveform preview we restrict the number of displayed sweeps
//           to the number of changing sweeps so as to decrease redraw time.
//
UINT WINAPI ABFH_GetNumberOfChangingSweeps( const ABFFileHeader *pFH )
{
   ABFH_ASSERT(pFH);
   
   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );
   
   UINT uNumChangingSweeps = 1;
   UINT uMaxChangingSweeps = 1;
   UINT uNumEpisodes = NewFH.lEpisodesPerRun;
   if( uNumEpisodes > 1 )
   {
      for( int nDAC = 0; nDAC < ABF_WAVEFORMCOUNT; nDAC++ )
      {
         // Just two sweeps change if alternating output is set
         if( NewFH.nAlternateDACOutputState || NewFH.nAlternateDigitalOutputState )
              uNumChangingSweeps = 2;

         // Get number of entries of user list
         if( NewFH.nULEnable[ nDAC ] )
         {
            CUserList UserList;
            UserList.Initialize( &NewFH, nDAC );
            UINT uListEntries = UserList.GetNumEntries();
            ASSERT( uListEntries > 0 );
            if( uListEntries > uNumChangingSweeps )
               uNumChangingSweeps = min(uListEntries, uNumEpisodes );
         }

         // All sweeps are assumed to change if increments are set
         for( int nEpoch = 0; nEpoch < ABF_EPOCHCOUNT; nEpoch++ )
         {
            if( (NewFH.fEpochLevelInc[ nDAC ][ nEpoch ] != 0.0F) || 
                (NewFH.lEpochDurationInc[ nDAC ][ nEpoch ] != 0) )
            {
               uNumChangingSweeps = NewFH.lEpisodesPerRun;
            }
         }
         uMaxChangingSweeps = max(uMaxChangingSweeps, uNumChangingSweeps);
      }
   }
   return uMaxChangingSweeps;
}

//===============================================================================================
// FUNCTION: ABFH_IsConstantWaveform
// PURPOSE:  Checks whether the waveform varies from episode to episode.
//
BOOL WINAPI ABFH_IsConstantWaveform(const ABFFileHeader *pFH)
{
   return ABFH_IsConstantWaveformEx(pFH, pFH->nActiveDACChannel);
}

BOOL WINAPI ABFH_IsConstantWaveformEx(const ABFFileHeader *pFH, UINT uDACChannel)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   int i=0;

   if( !NewFH.nWaveformEnable[uDACChannel] ||
        NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED)
      return TRUE;
      
   if (NewFH.nWaveformSource[uDACChannel] == ABF_DACFILEWAVEFORM)
      return ( NewFH.lDACFileNumEpisodes[uDACChannel] == 1 );

   if (NewFH.nConditEnable[uDACChannel] &&
       NewFH.nULEnable[uDACChannel])
   {
      if ((NewFH.nULParamToVary[uDACChannel] == ABF_CONDITPOSTTRAINLEVEL) &&
         (NewFH.fPostTrainPeriod[uDACChannel] > 0.0F))
         return FALSE;

      if (NewFH.nULParamToVary[uDACChannel] == ABF_CONDITPOSTTRAINDURATION)
         return FALSE;

      if ((NewFH.nULParamToVary[uDACChannel] == ABF_CONDITSTEPLEVEL) &&
         (NewFH.fPostTrainPeriod[uDACChannel] == 0.0F))
         return FALSE;
   }

   for (i=0; i<ABF_EPOCHCOUNT; i++)
   {
      if (NewFH.nEpochType[uDACChannel][i] == 0)
         continue;

      if (NewFH.nULEnable[uDACChannel] &&
         (NewFH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITLEVEL) &&
         (NewFH.nULParamToVary[uDACChannel] < ABF_EPOCHINITDURATION) &&
         (i == NewFH.nULParamToVary[uDACChannel] - ABF_EPOCHINITLEVEL))
         return FALSE;

      if (NewFH.nULEnable[uDACChannel] &&
         (NewFH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITDURATION) &&
         (i == NewFH.nULParamToVary[uDACChannel] - ABF_EPOCHINITDURATION))
         return FALSE;

      if (NewFH.nULEnable[uDACChannel] &&
         (NewFH.nULParamToVary[uDACChannel] >= ABF_EPOCHTRAINPERIOD) &&
         (i == NewFH.nULParamToVary[uDACChannel] - ABF_EPOCHTRAINPERIOD))
         return FALSE;

      if (NewFH.nULEnable[uDACChannel] &&
         (NewFH.nULParamToVary[uDACChannel] >= ABF_EPOCHTRAINPULSEWIDTH) &&
         (i == NewFH.nULParamToVary[uDACChannel] - ABF_EPOCHTRAINPULSEWIDTH))
         return FALSE;

      if (NewFH.fEpochLevelInc[uDACChannel][i] != 0.0F)
         return FALSE;
      if (NewFH.lEpochDurationInc[uDACChannel][i] != 0)
         return FALSE;
   }
   
   if( NewFH.nAlternateDACOutputState )
      return FALSE;

   return TRUE;
}


//===============================================================================================
// FUNCTION: ABFH_IsConstantDigitalOutput
// PURPOSE:  Checks whether the waveform varies from episode to episode.
//
BOOL WINAPI ABFH_IsConstantDigitalOutput(const ABFFileHeader *pFH)
{
   return ABFH_IsConstantDigitalOutputEx(pFH, pFH->nActiveDACChannel);
}

BOOL WINAPI ABFH_IsConstantDigitalOutputEx(const ABFFileHeader *pFH, UINT uDACChannel)
{
   ABFH_ASSERT(pFH);
   ASSERT( uDACChannel < ABF_WAVEFORMCOUNT );

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   int i=0;
   if( !(NewFH.nDigitalEnable) ||
         NewFH.nWaveformSource[uDACChannel] == ABF_WAVEFORMDISABLED)
      return TRUE;
      
   if (NewFH.nWaveformSource[uDACChannel] == ABF_DACFILEWAVEFORM)
      return ( NewFH.lDACFileNumEpisodes[uDACChannel] == 1 );

   for (i=0; i<ABF_EPOCHCOUNT; i++)
   {
      if (NewFH.nEpochType[uDACChannel][i] == 0)
         continue;

      if (NewFH.nULEnable[uDACChannel] &&
         (NewFH.nULParamToVary[uDACChannel] >= ABF_EPOCHINITDURATION) &&
         (i == NewFH.nULParamToVary[uDACChannel] - ABF_EPOCHINITDURATION))
         return FALSE;

      if (NewFH.lEpochDurationInc[uDACChannel][i] != 0)
         return FALSE;
   }
   if( NewFH.nAlternateDigitalOutputState )
      return FALSE;

   return TRUE;
}

//==============================================================================================
// FUNCTION: _IsMultipleOf
// PURPOSE:  Determines whether a number is a multiple of another number.
//
static BOOL _IsMultipleOf(double dSampleInterval, double dGranularity)
{
   double dDelta = fmod(dSampleInterval, dGranularity);
   if (dDelta > dGranularity/2)
      dDelta -= dGranularity;
   double dCompare = max(dSampleInterval, dGranularity)/1E6;
   return (fabs(dDelta) < dCompare);
}

//==============================================================================================
// FUNCTION: _Granularity
// PURPOSE:  Returns the granularity at the given interval.
//
static double _Granularity(double dSampleInterval, float fClockResolution)
{
   // If less than 20us allow full board resolution.
   // >20us but <=200us must be multiple of 5us.
   // >200us but <=400us must be multiple of 10us.
   // >400us must be multiple of 20us.
   // This assumes that the digitizer resolution is at least 0.5us

   if (dSampleInterval < 20.0)
      return fClockResolution;

   if (dSampleInterval < 200.0)
      return 5.0;

   if (dSampleInterval < 400.0)
      return 10.0;

   return 20.0;
}

//==============================================================================================
// FUNCTION: ABFH_CheckSampleIntervals
// PURPOSE:  Checks that the sample intervals in the header are valid.
//{{
// All intervals are in PROTOCOL units.
// (100 uS, 2 channels == 50uS)
//}}

BOOL WINAPI ABFH_CheckSampleIntervals(const ABFFileHeader *pFH, float fClockResolution, int *pnError)
{
   if (!_IsMultipleOf(pFH->fADCSampleInterval, fClockResolution))
      ERRORRETURN(pnError, ABFH_BADSAMPLEINTERVAL);

   if (pFH->nOperationMode!=ABF_WAVEFORMFILE)
      return TRUE;

   double dFirstInterval  = pFH->fADCSampleInterval;
   double dSecondInterval = pFH->fADCSecondSampleInterval;
   double dMinInterval    = dFirstInterval;

   // If the second interval comes into play, check that it is valid.
   if (dSecondInterval != 0.0)
   {
      if (!_IsMultipleOf(dSecondInterval, fClockResolution))
         ERRORRETURN(pnError, ABFH_BADSECONDSAMPLEINTERVAL);

      dMinInterval        = min(dFirstInterval, dSecondInterval);
      double dMaxInterval = max(dFirstInterval, dSecondInterval);
      if (!_IsMultipleOf(dMaxInterval, dMinInterval))
         ERRORRETURN(pnError, ABFH_BADSAMPLEINTERVALS);
   }
   else
      dSecondInterval = dFirstInterval;

   double dGranularity = _Granularity(dMinInterval, fClockResolution);
   if (!_IsMultipleOf(dMinInterval, dGranularity))
   {
      if (dFirstInterval <= dSecondInterval)
         ERRORRETURN(pnError, ABFH_BADSAMPLEINTERVAL);
      ERRORRETURN(pnError, ABFH_BADSECONDSAMPLEINTERVAL);
   }
   return TRUE;
}

//==============================================================================================
// FUNCTION: ClipInterval
// PURPOSE:  Clips the interval to be within the proscribed range.
//
static float ClipInterval(double dInterval, float fMax,  float fMin)
{
   if (dInterval > fMax)
      return fMax;
   if (dInterval < fMin)
      return fMin;
   return float(dInterval);
}

//==============================================================================================
// FUNCTION: ABFH_GetClosestSampleIntervals
// PURPOSE:  Gets the closest sample intervals higher and lower than the passed interval.
//
void WINAPI ABFH_GetClosestSampleIntervals(float fSampleInterval, float fClockResolution, 
                                           int nOperationMode, float fMinPeriod, float fMaxPeriod,
                                           float *pfHigher, float *pfLower)
{
   double dSampleInterval = fSampleInterval;
   double dGranularity = fClockResolution;
   if (nOperationMode==ABF_WAVEFORMFILE)
      dGranularity = _Granularity(dSampleInterval, fClockResolution);

   double dRemainder = fmod(dSampleInterval, dGranularity);
   double dMin = dSampleInterval - dRemainder;
   double dMax = dMin + dGranularity;
   if (pfLower)
      *pfLower = ClipInterval(dMin, fMaxPeriod, fMinPeriod);
   if (pfHigher)
      *pfHigher = ClipInterval(dMax, fMaxPeriod, fMinPeriod);
}


//===============================================================================================
// FUNCTION: ABFH_SetupSamplingList
// PURPOSE:  Sets up the list for the spinner to drive the sampling interval through.
// RETURNS:  The number of items added to the list.
//
UINT WINAPI ABFH_SetupSamplingList(UINT uNumChannels, float fMinPeriod, float fMaxPeriod, 
                                   float *pfIntervalList, UINT uListEntries)
{
   float fMultiplier = 0.0F;
   switch (uNumChannels)
   {
      case 1:
      case 2:
      case 5:
      case 10:
         fMultiplier = 1.0F;
         break;
      case 4:
      case 8:
         fMultiplier = 2.0F;
         break;
      case 3:
      case 6:
         fMultiplier = 3.0F;
         break;
      default:
         fMultiplier = float(uNumChannels);
         break;
   }

   // Build the list of sampling intervals.
   const UINT uMAX_DECADES = 7;
   UINT uEntries = min(uMAX_DECADES*3, uListEntries);
   uEntries -= uEntries % 3;
   float fDecade = 1.0F;
   for (UINT i=0; i<uEntries/3; i++)
   {
      pfIntervalList[i*3]   = fDecade*1.0F * fMultiplier;
      pfIntervalList[i*3+1] = fDecade*2.0F * fMultiplier;
      pfIntervalList[i*3+2] = fDecade*5.0F * fMultiplier;
      fDecade *= 10.0F;
   }

   // Calculate the extremes of the range.
   float fMinInterval = fMinPeriod*uNumChannels;
   float fMaxInterval = fMaxPeriod*uNumChannels;

   // If any entries in the list exceed the allowable range, edit them out.
   UINT uShift = 0;
   while (pfIntervalList[uShift] < fMinInterval)
   {
      uShift++;
      uEntries--;
   }
   while (pfIntervalList[uEntries-1] > fMaxInterval)
      uEntries--;

   // Put the minimum interval in at the start of the list if it is not there.
   if ((uShift > 0) &&
       (pfIntervalList[uShift] > fMinInterval * 1.01))
      pfIntervalList[--uShift] = fMinInterval;

   // If any entries were taken out of the front of the list,
   // pack the list down.
   if (uShift > 0)
      for (int i=0; i<uEntries; i++)
         pfIntervalList[i] = pfIntervalList[i+uShift];

   // Add the max interval at the end of the list if it is not there.
   if ((uEntries < uListEntries) && 
       (pfIntervalList[uEntries-1] < fMaxInterval * 0.99))
      pfIntervalList[uEntries++] = fMaxInterval;

   // Pad out the remaining list entries with the last value.
   for (int i=uEntries; i<uListEntries; i++)
      pfIntervalList[i] = pfIntervalList[uEntries-1];

   return uEntries;
}

*/
//==============================================================================================
// FUNCTION: ABFH_GetClockChange
// PURPOSE:  Gets the point at which the sampling interval changes if split clock.
//
UINT WINAPI ABFH_GetClockChange(const ABFFileHeader *pFH)
{
   ABFH_ASSERT(pFH);
   UINT uSamplesPerSweep = UINT(pFH->lNumSamplesPerEpisode);

   // If no change, return the full sweep length.
   if( (pFH->fADCSecondSampleInterval==0.0F) ||
       (pFH->fADCSampleInterval==pFH->fADCSecondSampleInterval ) ||
      (pFH->nOperationMode!=ABF_WAVEFORMFILE))
      return uSamplesPerSweep;

   // If the clock change point is zero it means change at the halfway point.
   UINT uClockChange = (pFH->lClockChange > 0) ? UINT(pFH->lClockChange) : uSamplesPerSweep/2;

   // Round it to the next lowest input sequence boundary.
   uClockChange -= uClockChange % UINT(pFH->nADCNumChannels);
   return uClockChange;
}


//==============================================================================================
// FUNCTION: ABFH_GetEpisodeDuration
// PURPOSE:  Gets the duration of the Waveform Episode (in us), allowing for split clock etc.
//
void WINAPI ABFH_GetEpisodeDuration(const ABFFileHeader *pFH, double *pdEpisodeDuration)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdEpisodeDuration);

   UINT uClockChange = ABFH_GetClockChange(pFH);
   *pdEpisodeDuration = uClockChange * ABFH_GetFirstSampleInterval(pFH) + 
                        UINT(pFH->lNumSamplesPerEpisode-uClockChange) * ABFH_GetSecondSampleInterval(pFH);
}


//==============================================================================================
// FUNCTION: ABFH_GetPNDuration
// PURPOSE:  Gets the duration of a P/N sequence (in us), including settling times.
//
void WINAPI ABFH_GetPNDuration(const ABFFileHeader *pFH, double *pdPNDuration)
{
   ABFH_ASSERT(pFH);

   ABFH_GetPNDurationEx( pFH, pFH->nActiveDACChannel, pdPNDuration);
}

#pragma warning( disable : 4189) //For the unused int pFH below
void WINAPI ABFH_GetPNDurationEx(const ABFFileHeader *pFH, UINT uDAC, double *pdPNDuration)
{
   WPTRASSERT(pdPNDuration);
   ABFH_ASSERT(pFH);
   ASSERT( uDAC < ABF_WAVEFORMCOUNT );

   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );
   {
      // Prevent use of pFH
      int pFH = 0;
      
      *pdPNDuration = 0.0;
      if ((NewFH.nOperationMode!=ABF_WAVEFORMFILE) || !NewFH.nPNEnable[uDAC])
         return;
      
      // Get sweep duration in us.
      double dEpisodeDuration = 0.0;
      ABFH_GetEpisodeDuration( &NewFH, &dEpisodeDuration );
      
      // Convert to ms.
      dEpisodeDuration /= 1000.0;
      
      // Get the P/N start to start time in ms.
      double dPNEpisodeDuration = fmax(dEpisodeDuration, double(NewFH.fPNInterpulse));
      
      // Calculate the total duration in ms.
      double dPNDuration = (NewFH.nPNNumPulses-1) * dPNEpisodeDuration + dEpisodeDuration + 
                           2 * NewFH.fPNSettlingTime;
      
      // Return the total duration in us.
      *pdPNDuration = dPNDuration * 1000.0;
   }
}
#pragma warning( default : 4189)


//==============================================================================================
// FUNCTION: ABFH_GetTrainDuration
// PURPOSE:  Gets the duration of a presweep train in us.
//
void WINAPI ABFH_GetTrainDuration(const ABFFileHeader *pFH, double *pdTrainDuration)
{
   ABFH_ASSERT(pFH);

   ABFH_GetTrainDurationEx (pFH, pFH->_nConditChannel, pdTrainDuration);
}

#pragma warning( disable : 4189) //For the unused int pFH below
void WINAPI ABFH_GetTrainDurationEx (const ABFFileHeader *pFH, UINT uDAC, double *pdTrainDuration)
{

   WPTRASSERT(pdTrainDuration);
   ABFH_ASSERT(pFH);
   ASSERT( uDAC < ABF_WAVEFORMCOUNT );

   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   {
      // Protect against accidental use of pFH.
      int pFH = 0;

      *pdTrainDuration = 0.0;
      if ((NewFH.nOperationMode!=ABF_WAVEFORMFILE) || !NewFH.nConditEnable[uDAC])
         return;

      // Calculate pulse duration in us.
      double dPulseDuration = double(NewFH.fBaselineDuration[uDAC] + NewFH.fStepDuration[uDAC]) * 1000.0;

      // Return train duration in us.
      *pdTrainDuration = NewFH.lConditNumPulses[uDAC] * dPulseDuration + NewFH.fPostTrainPeriod[uDAC] * 1000.0;
   }
}
#pragma warning( default : 4189)

//==============================================================================================
// FUNCTION: ABFH_GetMetaEpisodeDuration
// PURPOSE:  Gets the duration of a whole meta-episode (in us).
//
void WINAPI ABFH_GetMetaEpisodeDuration(const ABFFileHeader *pFH, double *pdMetaEpisodeDuration)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdMetaEpisodeDuration);

   double dTrainDuration=0.0, dPNDuration=0.0, dEpisodeDuration=0.0;
   double dTrainDur[ABF_WAVEFORMCOUNT];
   double dPNDur[ABF_WAVEFORMCOUNT];

   for( UINT i=0; i<ABF_WAVEFORMCOUNT; i++ )
   {
      ABFH_GetTrainDurationEx(pFH, i, &dTrainDur[i]);
      ABFH_GetPNDurationEx(pFH, i, &dPNDur[i]);
   }

   dTrainDuration = fmax( dTrainDur[0], dTrainDur[1] );
   dPNDuration    = fmax( dPNDur[0], dPNDur[1] );

   ABFH_GetEpisodeDuration(pFH, &dEpisodeDuration);

   *pdMetaEpisodeDuration = dTrainDuration + dPNDuration + dEpisodeDuration;
}

//==============================================================================================
// FUNCTION: ABFH_GetEpisodeStartToStart
// PURPOSE:  Gets the start to start period for the episode in us.
//
void WINAPI ABFH_GetEpisodeStartToStart(const ABFFileHeader *pFH, double *pdEpisodeStartToStart)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pdEpisodeStartToStart);

   *pdEpisodeStartToStart = 0.0;
   if (pFH->nOperationMode!=ABF_WAVEFORMFILE)
      return;

   double dMetaEpisodeDuration = 0.0;
   ABFH_GetMetaEpisodeDuration(pFH, &dMetaEpisodeDuration);
   *pdEpisodeStartToStart = fmax(dMetaEpisodeDuration, double(pFH->fEpisodeStartToStart)*1E6);
}
/*
//===============================================================================================
// METHOD:     CheckDACLevel
// TYPE:       Public Function
// PURPOSE:    Returns TRUE if the user value is within range.
// ARGUMENTS:  nChannel - The channel.
//             fValue - The value.
// RETURNS:    TRUE if the user value is within range.
//
BOOL CheckDACLevel( const ABFFileHeader *pFH,
                    int nChannel,
                    float fValue )
{
   // Clip the DAC value.
   float fClippedValue = fValue;
   ABFH_ClipDACUUValue( pFH, nChannel, &fClippedValue );
   
   BOOL bSame = (fValue == fClippedValue);
   return bSame;
}

//==============================================================================================
// FUNCTION: CheckMetaDuration
// PURPOSE:  Check that the meta episode duration doesn't exceed the requested episode start to start time.
// RETURNS:  TRUE if meta episode duration is OK
//
BOOL CheckMetaDuration( const ABFFileHeader *pFH )
{
   // If start to start time is minimum, its OK.
   if( pFH->fEpisodeStartToStart == 0.0F )
      return TRUE;

   double dMetaDuration = 0.0;
   ABFH_GetMetaEpisodeDuration( pFH, &dMetaDuration);

   return dMetaDuration <= pFH->fEpisodeStartToStart * 1E6;
}

//==============================================================================================
// FUNCTION: CheckEpochLength
// PURPOSE:  Check that the epoch length doesn't exceed the allowable maximum.
// RETURNS:  TRUE if epoch length is OK
//
BOOL CheckEpochLength( const ABFFileHeader *pFH, UINT uDACChannel, UINT uEpisode, int nListEpoch )
{
   ABFH_ASSERT( pFH);
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   // Get the maximum allowable epoch length
   int nMaxEpochLen = ABFH_UserLenFromSweepLen( FH.lNumSamplesPerEpisode, FH.nADCNumChannels );
   nMaxEpochLen /= FH.nADCNumChannels;

   // Calculate the duration of all epochs combined
   int nEpochDuration = 0;
   for( int nEpoch = 0; nEpoch < ABF_EPOCHCOUNT; nEpoch++ )
   {
      if( FH.nEpochType[uDACChannel][nEpoch] != ABF_EPOCHDISABLED )
      {
         int nDuration = FH.lEpochInitDuration[uDACChannel][nEpoch];
         //The duration must be > 0
         if( nDuration < 0 )
            return FALSE;

         if ((nEpoch != nListEpoch) && (uEpisode > 1))
            nDuration += uEpisode * FH.lEpochDurationInc[uDACChannel][nEpoch];
         if (nDuration > 0)
            nEpochDuration += nDuration;
      }
   }

   return nEpochDuration <= nMaxEpochLen;
}

//==============================================================================================
// FUNCTION: IsBinaryList
// PURPOSE:  Check that the User List is valid for binary items.
//
BOOL IsBinaryList( const ABFFileHeader *pFH, UINT uListNum ) 
{
   ABFH_ASSERT( pFH );
   ASSERT( uListNum < ABF_USERLISTCOUNT );

   char szUserList[ABF_USERLISTLEN+1];
   ABF_GET_STRING(szUserList, pFH->sULParamValueList[uListNum], sizeof(szUserList));
   return ( strspn( szUserList, "01*, " ) == strlen(szUserList) );
}

//==============================================================================================
// FUNCTION: AreListCharsOK
// PURPOSE:  Check that the User List only contains valid characters.
//
BOOL AreListCharsOK( const ABFFileHeader *pFH, UINT uListNum ) 
{
   ABFH_ASSERT( pFH );
   ASSERT( uListNum < ABF_USERLISTCOUNT );

   char szUserList[ABF_USERLISTLEN+1];
   ABF_GET_STRING(szUserList, pFH->sULParamValueList[uListNum], sizeof(szUserList));
   
   // a '*' is allowed for digital bit patterns.
   BOOL bStarInUserList = strspn( szUserList, "*0123456789, Ee.+-" ) == strlen( szUserList );
   BOOL bUserListParamIsDigitalPattern = pFH->nULParamToVary[uListNum] >= ABF_PARALLELVALUE && pFH->nULParamToVary[uListNum] < ABF_EPOCHINITLEVEL;
   if( bStarInUserList  && bUserListParamIsDigitalPattern )
       return TRUE;

   // Check the first character.
   BOOL bFirstCharInvalid = ( szUserList[ 0 ] < '0' || szUserList[ 0 ] > '9' );
   if ( ( bFirstCharInvalid ) &&
        szUserList[ 0 ] != '.' &&
        szUserList[ 0 ] != '+' &&
        szUserList[ 0 ] != '-' )
      return FALSE;

   // Check all the other characters.
   return ( strspn( szUserList, "0123456789, Ee.+-" ) == strlen(szUserList) );
}

//==============================================================================================
// FUNCTION: ABFH_CheckUserList
// PURPOSE:  Checks that the user list contains valid entries for the protocol.
//
BOOL WINAPI ABFH_CheckUserList(const ABFFileHeader *pFH, int *pnError)
{
   ABFH_ASSERT(pFH);

   return ABFH_CheckUserListEx(pFH, pFH->nActiveDACChannel, pnError);
}

BOOL WINAPI ABFH_CheckUserListEx(const ABFFileHeader *pFH, UINT uListNum, int *pnError)
{
   ABFH_ASSERT(pFH);
   ASSERT( uListNum < ABF_USERLISTCOUNT );

   // Take a copy of the ABF Header for testing User List values.
   ABFFileHeader FH;
   ABFH_PromoteHeader( &FH, pFH );

   // The user list is only defined for Stimulus Waveform acquisitions.
   if (FH.nOperationMode!=ABF_WAVEFORMFILE)
      return TRUE;
   
   // Only check if user list is enabled.
   if( FH.nULEnable[uListNum] == 0)
      return TRUE;

   // Check the characters in the user list.
   if (!AreListCharsOK( &FH, uListNum ))
      ERRORRETURN( pnError, ABFH_EINVALIDCHARS );

   
   // Create the User List
   CUserList UserList;
   
   // Make sure we are initialised.
   UserList.Initialize( &FH, uListNum );

   UINT uListEntries = UserList.GetNumEntries();
   ASSERT( uListEntries > 0 );

   // Prevent compiler warnings.
   uListEntries = uListEntries;

   short nParamToVary = FH.nULParamToVary[uListNum];

   // loop thru each episode
   for( int nEpisode = 1; nEpisode <= FH.lEpisodesPerRun; nEpisode++ )
   {
      UserList.UpdateHeader( &FH, uListNum, nEpisode );

      int   nValue = 0;
      float fValue = 0.0F;

      // Check things applicable to Presweep Trains
      if( nParamToVary == ABF_CONDITNUMPULSES ||
          nParamToVary == ABF_CONDITBASELINEDURATION ||
          nParamToVary == ABF_CONDITBASELINELEVEL ||
          nParamToVary == ABF_CONDITSTEPDURATION ||
          nParamToVary == ABF_CONDITSTEPLEVEL ||
          nParamToVary == ABF_CONDITPOSTTRAINDURATION ||
          nParamToVary == ABF_CONDITPOSTTRAINLEVEL )
      {
         // Check that Presweep Trains are enabled.
         if ( FH.nConditEnable[uListNum] == 0 )
            ERRORRETURN( pnError, ABFH_ENOCONDITTRAINS );

         // Check we don't make the meta episode too long
         if( !CheckMetaDuration( &FH ) )
            ERRORRETURN( pnError, ABFH_EMETADURATION );
      }

      // check each entry in the list for a safe value, return FALSE if not
      switch (nParamToVary)
      {
         case ABF_CONDITNUMPULSES:
            nValue = FH.lConditNumPulses[uListNum];
            if( nValue < 0 || nValue > ABF_CTPULSECOUNT_MAX )
               ERRORRETURN( pnError, ABFH_ECONDITNUMPULSES );
            break;
         
         case ABF_CONDITBASELINEDURATION:
            fValue = FH.fBaselineDuration[uListNum];
            if( fValue < 0.0F || fValue > ABF_CTBASELINEDURATION_MAX ) 
               ERRORRETURN( pnError, ABFH_ECONDITBASEDUR );
            break;
         
         case ABF_CONDITSTEPLEVEL:
            fValue = FH.fStepLevel[uListNum];
            if( !CheckDACLevel( &FH, uListNum, fValue ) ) 
               ERRORRETURN( pnError, ABFH_ECONDITSTEPLEVEL );
            // Fall through to test baseline level as well
         
         case ABF_CONDITBASELINELEVEL:
            fValue = FH.fBaselineLevel[uListNum];
            if( !CheckDACLevel( &FH, uListNum, fValue ) ) 
               ERRORRETURN( pnError, ABFH_ECONDITBASELEVEL );
            break;
         
         case ABF_CONDITSTEPDURATION:
            fValue = FH.fStepDuration[uListNum];
            if( fValue < 0.0F || fValue > ABF_CTSTEPDURATION_MAX ) 
               ERRORRETURN( pnError, ABFH_ECONDITSTEPDUR );
            break;
         
         case ABF_CONDITPOSTTRAINDURATION:
            fValue = FH.fPostTrainPeriod[uListNum];
            if( fValue < 0.0F || fValue > ABF_CTPOSTTRAINDURATION_MAX ) 
               ERRORRETURN( pnError, ABFH_ECONDITPOSTTRAINDUR );
            break;
         
         case ABF_CONDITPOSTTRAINLEVEL:
            fValue = FH.fPostTrainLevel[uListNum];
            if( !CheckDACLevel( &FH, uListNum, fValue ) ) 
               ERRORRETURN( pnError, ABFH_ECONDITPOSTTRAINLEVEL );
            break;
         
         case ABF_EPISODESTARTTOSTART:
            if( !CheckMetaDuration( &FH ) )
               ERRORRETURN( pnError, ABFH_EMETADURATION );
            break;
         
         case ABF_INACTIVEHOLDING:
            fValue = FH.fDACHoldingLevel[1-FH.nActiveDACChannel] ;	
            if( !CheckDACLevel( &FH, 1-FH.nActiveDACChannel, fValue ) ) 
               ERRORRETURN( pnError, ABFH_EINACTIVEHOLDING );
            break;
         
         case ABF_DIGITALHOLDING:
            if (!IsBinaryList( &FH, uListNum ))
               ERRORRETURN( pnError, ABFH_EINVALIDBINARYCHARS );
            if( FH.nDigitalEnable == 0 )
               ERRORRETURN( pnError, ABFH_ENODIG );
            nValue = FH.nDigitalHolding;
            if( nValue < 0 || nValue > ABF_DIGITALVALUE_MAX )
               ERRORRETURN( pnError, ABFH_EDIGHOLDLEVEL );
            break;
         
         case ABF_PNNUMPULSES:
            if( FH.nPNEnable == 0 )
               ERRORRETURN( pnError, ABFH_ENOPNPULSES );
            nValue = FH.nPNNumPulses;
            if( nValue < 1 || nValue > ABF_PNPULSECOUNT_MAX ) 
               ERRORRETURN( pnError, ABFH_EPNNUMPULSES );
            if( !CheckMetaDuration( &FH ) )
               ERRORRETURN( pnError, ABFH_EMETADURATION );
            break;
         
         default:
            // Epoch related parameter to vary
            
            int nListEpoch = UserList.GetActiveEpoch();

            ASSERT(nListEpoch >= 0);
            ASSERT(nListEpoch < ABF_EPOCHCOUNT);
            
            // Check that the waveform is enabled.
            if( (!FH.nWaveformEnable[uListNum] || 
                 FH.nWaveformSource[uListNum] == ABF_WAVEFORMDISABLED) &&
                 FH.nDigitalEnable == 0 )
               ERRORRETURN( pnError, ABFH_ENOWAVEFORM );
            
            // Make sure that the epoch chosen is enabled.
            if ( FH.nEpochType[uListNum][nListEpoch] == ABF_EPOCHDISABLED )
               ERRORRETURN( pnError, ABFH_ENOEPOCH);
            
            if (nParamToVary >= ABF_EPOCHINITDURATION)
            {
               if( !CheckEpochLength( &FH, uListNum, nEpisode, nListEpoch ) )
                  ERRORRETURN( pnError, ABFH_EEPOCHLEN );
            }
            else if (nParamToVary >= ABF_EPOCHINITLEVEL)
            {
               fValue = FH.fEpochInitLevel[uListNum][nListEpoch];
               if( !CheckDACLevel( &FH, uListNum, fValue ) ) 
                  ERRORRETURN( pnError, ABFH_EEPOCHINITLEVEL );
            }
            else if (nParamToVary >= ABF_PARALLELVALUE)
            {
               if (!IsBinaryList( &FH, uListNum ))
                  ERRORRETURN( pnError, ABFH_EINVALIDBINARYCHARS);
               nValue = FH.nDigitalValue[nListEpoch];
               if( nValue < 0 || nValue > ABF_EPOCHDIGITALVALUE_MAX )
                  ERRORRETURN( pnError, ABFH_EDIGLEVEL );
               
               // Check the digital train value.
               int nTrainValue = FH.nDigitalTrainValue[nListEpoch];
               if( nTrainValue < 0 || nTrainValue > ABF_EPOCHDIGITALVALUE_MAX )
                  ERRORRETURN( pnError, ABFH_EDIGLEVEL );
            }
            break;
         }         
   }

   // if we got here, everything's OK
   return TRUE;
}
*/
