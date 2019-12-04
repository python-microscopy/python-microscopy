//***********************************************************************************************
//
//    Copyright (c) 1993-2000 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// This is ABFHEADR.CPP; the routines that cope with reading the data file
// parameters block for all AXON pCLAMP binary file formats.
//
// An ANSI C compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL ABFHEADR.C)

#include "../Common/wincpp.hpp"

#include "abfheadr.h"                  // header definition & constants
#include "oldheadr.h"                  // old header conversion prototypes
#include "abfutil.h"                   // Large memory allocation/free
/*
#include "StringResource.h"            // Access to string resources.
*/
#define A_VERY_SMALL_NUMBER      1E-10
#define DEFAULT_LEVEL_HYSTERESIS 64    // Two LSBits of level hysteresis.
#define DEFAULT_TIME_HYSTERESIS  1     // Two sequences of time hysteresis.

#if defined(__UNIX__) || defined(__STF__)
	#define max(a,b)   (((a) > (b)) ? (a) : (b))
	#define min(a,b)   (((a) < (b)) ? (a) : (b))
#endif

const char c_szValidOperators[] = "+-*/";

const long c_lMaxShort          = 30000;

//-----------------------------------------------------------------------------------------------
// Uncomment the following line to display interface structure sizes.
//#define SHOW_STRUCT_SIZES 1

//-----------------------------------------------------------------------------------------------
// Macros and functions to deal with returning error return codes through a pointer if given.

#define ERRORRETURN(p, e)  return ErrorReturn(p, e);
static BOOL ErrorReturn(int *pnError, int nErrorNum)
{
   if (pnError)
      *pnError = nErrorNum;
   return FALSE;
}

//===============================================================================================
// FUNCTION: ABFH_Initialize
// PURPOSE:  Initialize an ABFFileHeader structure to a consistent set of parameters
//
void WINAPI ABFH_Initialize( ABFFileHeader *pFH )
{
//   ABFH_WASSERT(pFH);
   int i;
   
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );

   // Zero fill all to start with.
   memset(&NewFH, '\0', sizeof(NewFH));
   
   // Blank fill all strings.
   ABF_BLANK_FILL(NewFH._sParamValueList);
   ABF_BLANK_FILL(NewFH.sADCChannelName);
   ABF_BLANK_FILL(NewFH.sADCUnits);
   ABF_BLANK_FILL(NewFH.sDACChannelName);
   ABF_BLANK_FILL(NewFH.sDACChannelUnits);
   ABF_BLANK_FILL(NewFH.sDACFilePath[0]);
   ABF_BLANK_FILL(NewFH.sDACFilePath[1]);
   ABF_SET_STRING(NewFH.sArithmeticOperator, "+");
   ABF_BLANK_FILL(NewFH.sArithmeticUnits);
      
   NewFH.lFileSignature        = ABF_NATIVESIGNATURE;
   NewFH.fFileVersionNumber    = ABF_CURRENTVERSION;
   NewFH.fHeaderVersionNumber  = ABF_CURRENTVERSION;
   NewFH.nOperationMode        = ABF_GAPFREEFILE;
   NewFH.nADCNumChannels       = 1;
   NewFH.fADCSampleInterval    = 100.0F;
   NewFH.lNumSamplesPerEpisode = 512;
   NewFH.lEpisodesPerRun       = 1;
   NewFH.lDataSectionPtr       = sizeof(ABFFileHeader) / ABF_BLOCKSIZE;
      
   NewFH.nDrawingStrategy      = ABF_DRAW_REALTIME;
   NewFH.nTiledDisplay         = ABF_DISPLAY_TILED;
   NewFH.nEraseStrategy        = 1;
   NewFH.nDataDisplayMode      = ABF_DRAW_LINES;
   NewFH.nMultiColor           = TRUE;
   NewFH.nFileType             = ABF_ABFFILE;
   NewFH.nAutoTriggerStrategy  = 1;   // Allow auto triggering.
   NewFH.nChannelStatsStrategy = 0;   // Don't calculate channel statistics.
   NewFH.fStatisticsPeriod     = 1.0F;
   NewFH.lCalculationPeriod    = long(NewFH.fStatisticsPeriod / NewFH.fADCSampleInterval * 1E3F);
   NewFH.lStatisticsMeasurements = ABF_STATISTICS_ABOVETHRESHOLD | ABF_STATISTICS_MEANOPENTIME;
   
   NewFH.lSamplesPerTrace      = 16384;
   NewFH.lPreTriggerSamples    = 16;    // default to 16

   NewFH.fADCRange             = 10.24F;
   NewFH.fDACRange             = 10.24F;
   NewFH.lADCResolution        = 32768L;
   NewFH.lDACResolution        = 32768L;
   NewFH.nExperimentType       = ABF_SIMPLEACQUISITION;
   
      
   ABF_BLANK_FILL(NewFH.sCreatorInfo);
   ABF_BLANK_FILL(NewFH.sModifierInfo);
   ABF_BLANK_FILL(NewFH.sFileComment);
      
   // ADC channel data
   for (i=0; i<ABF_ADCCOUNT; i++)
   {
      char szName[13];      
      sprintf(szName, "AI #%-8d", i);
      strncpy(NewFH.sADCChannelName[i], szName, ABF_ADCNAMELEN);
      strncpy(NewFH.sADCUnits[i], "pA        ", ABF_ADCUNITLEN);
      
      NewFH.nADCPtoLChannelMap[i]       = short(i);
      NewFH.nADCSamplingSeq[i]          = ABF_UNUSED_CHANNEL;
      NewFH.fADCProgrammableGain[i]     = 1.0F;
      NewFH.fADCDisplayAmplification[i] = 1.0F;
      NewFH.fInstrumentScaleFactor[i]   = 0.1F;
      NewFH.fSignalGain[i]              = 1.0F;
      NewFH.fSignalLowpassFilter[i]     = ABF_FILTERDISABLED;

// FIX FIX FIX PRC DEBUG Telegraph changes - check !
      NewFH.fTelegraphAdditGain[i]      = 1.0F;
      NewFH.fTelegraphFilter[i]         = 100000.0F;
   }
   NewFH.nADCSamplingSeq[0] = 0;
      
   // DAC channel data
   for (i=0; i<ABF_DACCOUNT; i++)
   {
      char szName[13];
      sprintf(szName, "AO #%-8d", i);
      strncpy(NewFH.sDACChannelName[i], szName, ABF_DACNAMELEN);
      strncpy(NewFH.sDACChannelUnits[i], "mV        ", ABF_ADCUNITLEN);
      NewFH.fDACScaleFactor[i] = 20.0F;
   }
   
   // DAC file settings
   for (i=0; i<ABF_WAVEFORMCOUNT; i++)
   {
      NewFH.fDACFileScale[i] = 1.0F;
      NewFH.nPNPolarity[i]   = ABF_PN_SAME_POLARITY;
   }

   NewFH.nPNNumPulses        = 2;
   NewFH.fPNInterpulse       = 0;

   // Initialize as non-zero to avoid glitch in first holding
   NewFH.fPNSettlingTime     = 10;
   for (i=0; i<ABF_WAVEFORMCOUNT; i++)
      NewFH.fPostTrainPeriod[i] = 10;

   // Initialize statistics variables.
   NewFH.nStatsSearchRegionFlags = ABF_PEAK_SEARCH_REGION0;
   NewFH.nStatsBaseline          = ABF_PEAK_BASELINE_SPECIFIED;
   NewFH.nStatsSelectedRegion    = 0;
   NewFH.nStatsSmoothing         = 1;
   NewFH.nStatsActiveChannels    = 0;
   for( int nStatsRegionID = 0; nStatsRegionID < ABF_STATS_REGIONS; nStatsRegionID++ )
   {
      NewFH.nStatsSearchMode[ nStatsRegionID ]       = ABF_PEAK_SEARCH_SPECIFIED;
      NewFH.lStatsMeasurements[ nStatsRegionID ]     = ABF_PEAK_MEASURE_PEAK | ABF_PEAK_MEASURE_PEAKTIME;
      NewFH.nRiseBottomPercentile[ nStatsRegionID ]  = 10;
      NewFH.nRiseTopPercentile[ nStatsRegionID ]     = 90;
      NewFH.nDecayBottomPercentile[ nStatsRegionID ] = 10;
      NewFH.nDecayTopPercentile[ nStatsRegionID ]    = 90;   
   }

   for ( UINT uChannel = 0; uChannel < ABF_ADCCOUNT; uChannel++ )
      NewFH.nStatsChannelPolarity[uChannel] = ABF_PEAK_ABSOLUTE;

   NewFH.fArithmeticUpperLimit = 100.0F;
   NewFH.fArithmeticLowerLimit = -100.0F;
   NewFH.fArithmeticK1         = 1.0F;
   NewFH.fArithmeticK3         = 1.0F;

   for (i=0; i<ABF_BELLCOUNT; i++)   
   {
      NewFH.nBellEnable[i] = 0;
      NewFH.nBellLocation[i] = 1;
      NewFH.nBellRepetitions[i] = 1;
   }
   NewFH.nLevelHysteresis    = DEFAULT_LEVEL_HYSTERESIS;   // Two LSBits of level hysteresis.
   NewFH.lTimeHysteresis     = DEFAULT_TIME_HYSTERESIS;    // Two sequences of time hysteresis.
   NewFH.fAverageWeighting   = 0.1F;                       // Add 10% of trace to 90% of average.
   NewFH.nTrialTriggerSource = ABF_TRIALTRIGGER_NONE;
   NewFH.nExternalTagType    = ABF_EXTERNALTAG;

   NewFH.lHeaderSize         = ABF_HEADERSIZE;
   NewFH.nAutoAnalyseEnable  = ABF_AUTOANALYSE_DEFAULT;

   for( i=0; i<ABF_USERLISTCOUNT; i++ )
      ABF_BLANK_FILL( NewFH.sULParamValueList[i] );

   // DAC Calibration Factors.
   for( i=0; i<ABF_DACCOUNT; i++ )
   {
      NewFH.fDACCalibrationFactor[i] = 1.0F;
      NewFH.fDACCalibrationOffset[i] = 0.0F;
   }

   // Digital train params.
   NewFH.nDigitalTrainActiveLogic = 1;
   for( i = 0; i < ABF_EPOCHCOUNT; i ++ )
   {
      NewFH.nDigitalTrainValue[ i ] = 0;
   }

   // Initialize LTP type.
   NewFH.nLTPType = ABF_LTP_TYPE_NONE;
   for( i=0; i<ABF_WAVEFORMCOUNT; i++ )
   {
      NewFH.nLTPUsageOfDAC[ i ] = ABF_LTP_DAC_USAGE_NONE;
      NewFH.nLTPPresynapticPulses[ i ] = 0;
   }

   // Epoch resistance
   for( i = 0; i < ABF_WAVEFORMCOUNT; i++ )
   {
      sprintf( NewFH.sEpochResistanceSignalName[ i ], "IN #%d", i);
      NewFH.nEpochResistanceState[ i ] = 0; 
   }

   // Alternating Outputs 
   NewFH.nAlternateDACOutputState = 0;
   NewFH.nAlternateDigitalOutputState = 0;
   for( int nEpoch = 0; nEpoch  < ABF_EPOCHCOUNT; nEpoch ++ )
   {
      NewFH.nAlternateDigitalValue[ nEpoch ] = 0;
      NewFH.nAlternateDigitalTrainValue[ nEpoch ] = 0;
   }

   //Post-processing values.
   for( i=0; i<ABF_ADCCOUNT; i++)
   {
      NewFH.fPostProcessLowpassFilter[i] = ABF_FILTERDISABLED;
      NewFH.nPostProcessLowpassFilterType[i] = ABF_POSTPROCESS_FILTER_NONE;
   }

   ABFH_DemoteHeader( pFH, &NewFH );
}
/*
//===============================================================================================
// FUNCTION: _GetDefaultScopeMode
// PURPOSE:  Gets the Scope mode appropriate to the acquisition mode.
//   
static WORD _GetDefaultScopeMode(const ABFFileHeader *pFH)
{
   switch (pFH->nOperationMode)
   {
      case ABF_FIXLENEVENTS:
      case ABF_VARLENEVENTS:
      case ABF_GAPFREEFILE:
         return ABF_CONTINUOUSMODE;

      case ABF_HIGHSPEEDOSC:
      case ABF_WAVEFORMFILE:
         return ABF_EPISODICMODE;

      default:
         ERRORMSG1("Unexpected operation mode '%d'.", pFH->nOperationMode);
         return ABF_EPISODICMODE;
   }
}

//===============================================================================================
// FUNCTION: SetupSignal
// PURPOSE:  Sets up a signal descriptor from display parameters in an old ABF header.
//   
static void _SetupSignal(const ABFFileHeader *pFH, ABFSignal *pS, UINT uMxOffset, UINT uTraces)
{
   memset(pS, 0, sizeof(ABFSignal));
   int nChannel = pFH->nADCSamplingSeq[uMxOffset];
   ABFU_GetABFString(pS->szName, ABF_ADCNAMELEN+1, pFH->sADCChannelName[nChannel], ABF_ADCNAMELEN);
   pS->nMxOffset = short(uMxOffset);
   pS->rgbColor = RGB_BLUE;
   pS->nPenWidth = 1;
   pS->bDrawPoints = char(!pFH->nDataDisplayMode);
   pS->bHidden = FALSE;
   pS->bFloatData = FALSE;
   pS->fVertProportion = 1.0F / float(uTraces);
   pS->fDisplayGain   = pFH->fADCDisplayAmplification[nChannel];
   pS->fDisplayOffset = pFH->fADCDisplayOffset[nChannel];
}

//===============================================================================================
// FUNCTION: SetupMathSignal
// PURPOSE:  Sets up a signal descriptor for a math channel from parameters in an ABF header.
//   
static void _SetupMathSignal(const ABFFileHeader *pFH, ABFSignal *pS, UINT uTraces)
{
   memset(pS, 0, sizeof(ABFSignal));
   LoadString(g_hInstance, IDS_MATHCHANNEL, pS->szName, sizeof(pS->szName));
   pS->nMxOffset = 0;
   pS->rgbColor = RGB_BLUE;
   pS->nPenWidth = 1;
   pS->bDrawPoints = char(!pFH->nDataDisplayMode);
   pS->bHidden = FALSE;
   pS->bFloatData = TRUE;
   pS->fVertProportion = 1.0F / float(uTraces);
   pS->fDisplayGain   = 1.0F;
   pS->fDisplayOffset = 0.0F;
}

//===============================================================================================
// FUNCTION: MathChannelEnabled
// PURPOSE:  Returns TRUE if the math channel is enabled.
//   
static BOOL MathChannelEnabled(const ABFFileHeader *pFH)
{
   return (pFH->nArithmeticEnable && (pFH->nADCNumChannels < ABF_ADCCOUNT));
}

//===============================================================================================
// FUNCTION: _GetTraceCount
// PURPOSE:  Gets the number of traces that should be displayed for this acquisition.
//   
static UINT _GetTraceCount(const ABFFileHeader *pFH, BOOL *pbMathChannel=NULL)
{
   UINT uTraces = UINT(pFH->nADCNumChannels);
   BOOL bMathChannel = MathChannelEnabled(pFH);
   if (bMathChannel)
      uTraces++;
   if (pbMathChannel)
      *pbMathChannel = bMathChannel;
   return uTraces;
}

//===============================================================================================
// FUNCTION: ABFH_InitializeScopeConfig
// PURPOSE:  Sets up a scope configuration structure from the old header locations for display
//           parameters.
//   
void WINAPI ABFH_InitializeScopeConfig(const ABFFileHeader *pFH, ABFScopeConfig *pCfg)
{
   // Zapp everything!
   memset(pCfg, 0, sizeof(*pCfg));
   
   // Enable SC_CLIPPING
   pCfg->dwFlags = 0x00000008;

   // Set the scope mode.
   pCfg->wScopeMode = _GetDefaultScopeMode(pFH);
   
   // Set up the first ScopeConfig structure.
   pCfg->rgbColor[ABF_BACKGROUNDCOLOR] = RGB_WHITE;
   pCfg->rgbColor[ABF_GRIDCOLOR]       = RGB_LTGRAY;
   pCfg->rgbColor[ABF_THRESHOLDCOLOR]  = RGB_DKRED;
   pCfg->rgbColor[ABF_EVENTMARKERCOLOR]= RGB_RED;
   pCfg->rgbColor[ABF_SEPARATORCOLOR]  = RGB_BLACK;
   pCfg->rgbColor[ABF_AVERAGECOLOR]    = RGB_RED;
   pCfg->rgbColor[ABF_OLDDATACOLOR]    = RGB_LTGRAY;
   pCfg->rgbColor[ABF_TEXTCOLOR]       = RGB_BLACK;
   pCfg->rgbColor[ABF_AXISCOLOR]       = GetSysColor(COLOR_3DFACE);
   pCfg->rgbColor[ABF_ACTIVEAXISCOLOR] = GetSysColor(COLOR_3DSHADOW);
   pCfg->fDisplayStart                 = 0.0F;   // If fDisplayStart==fDisplayEnd a default
   pCfg->fDisplayEnd                   = 0.0F;   // display scaling will be used.
   pCfg->nYAxisWidth                   = -1;
   pCfg->nEraseStrategy                = ABF_ERASE_EACHRUN;

   // Set the Signals in the first scope configuration to match the 
   // old entries in the header.
   BOOL bMathChannel = FALSE;
   UINT uTraces = _GetTraceCount(pFH, &bMathChannel);
   UINT uNormalTraces = uTraces;
   if (bMathChannel)
      uNormalTraces--;
   UINT i;
   for (i=0; i<uNormalTraces; i++)
      _SetupSignal(pFH, pCfg->TraceList+i, i, uTraces);
   if (bMathChannel)
      _SetupMathSignal(pFH, pCfg->TraceList+i, uTraces);

   pCfg->nTraceCount = short(uTraces);

   // Initialize the extended ABFScopeConfig fields for ABF file version 1.68.
   pCfg->nAutoZeroState = 0;
   pCfg->nSizeofOldStructure = pCfg->nSizeofOldStructure;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION0 ] =  RGB_BLACK;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION1 ] =  RGB_DKRED;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION2 ] =  RGB_DKGREEN;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION3 ] =  RGB_DKYELLOW;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION4 ] =  RGB_DKBLUE;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION5 ] =  RGB_MAUVE;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION6 ] =  RGB_BLUEGRAY;
   pCfg->rgbColorEx[ ABF_STATISTICS_REGION7 ] =  RGB_DKGRAY;
   pCfg->rgbColorEx[ ABF_BASELINE_REGION ]    =  RGB_RED;
   pCfg->rgbColorEx[ ABF_STOREDSWEEPCOLOR ]   =  RGB_BLUEGRAY;

} 

//===============================================================================================
// FUNCTION: SetMxOffset
// PURPOSE:  Returns TRUE if the passed signal is still in the sampling sequence.
//   
static BOOL SetMxOffset(const ABFFileHeader *pFH, ABFSignal *pS)
{
   //  Initialize to failure case.
   pS->nMxOffset = -1;

   if (!pS->bFloatData)
   {
      char szName[ABF_ADCNAMELEN+1];
      for (int i=0; i<int(pFH->nADCNumChannels); i++)
      {
         int nChannel = pFH->nADCSamplingSeq[i];
         ABFU_GetABFString(szName, sizeof(szName), pFH->sADCChannelName[nChannel], ABF_ADCNAMELEN);

         if (strcmp(pS->szName, szName)==0)
         {
            pS->nMxOffset = short(i);
            break;
         }
      }
   }
   else if (MathChannelEnabled(pFH))
      pS->nMxOffset = 0;

   return (pS->nMxOffset != -1);
}

//===============================================================================================
// FUNCTION: ABFH_CheckScopeConfig
// PURPOSE:  Checks a scope configuration structure against an ABFFileHeader structure, making
//           sure that both reference the same signals, in the same multiplex sequence.
// RETURNS:  FALSE if the configuration structure specified an innappropriate Scope mode.
//
BOOL WINAPI ABFH_CheckScopeConfig(ABFFileHeader *pFH, ABFScopeConfig *pCfg)
{
   ASSERT(pFH->nADCNumChannels > 0);
   
   // Search through the ABFScopeConfig trace list to see if all signals are
   // included in the sampling list.
   BOOL bMathChannel = FALSE;
   UINT uNewTraceCount = _GetTraceCount(pFH, &bMathChannel);
   UINT uOldTraceCount = UINT(pCfg->nTraceCount);
   UINT uOldVisibleTraceCount = uOldTraceCount;

   UINT uGaps = 0;
   ABFSignal *pS = pCfg->TraceList;
   for (UINT i=0; i<uOldTraceCount; i++, pS++)
   {
      // Decrement the visible count for each hidden trace.
      if (pS->bHidden)
         uOldVisibleTraceCount--;

      // If the signal is not is the sampling list, mark it as being free for replacement.
      if ( !SetMxOffset(pFH, pS) )
      {
         pS->szName[0] = '\0';
         ++uGaps;
      }
   }
   ASSERT(uOldVisibleTraceCount > 0);
   
   // Scale back the proportions of any remaining channels so that new channels can be added in
   // with a proportion of 1.0 and get 1/n th of the pane allocated.
   if (uOldTraceCount != uNewTraceCount)
   {
      pS = pCfg->TraceList;
      for (UINT j=0; j<uOldTraceCount; j++, pS++)
         if (pS->szName[0] != '\0')
            pS->fVertProportion *= float(uOldVisibleTraceCount)/uNewTraceCount;
   }

   // Go through the list of acquired channels and insert any ones that are not
   // currently in the scope config list into the list.
   for (int i=0; i<UINT(pFH->nADCNumChannels); i++)
   {
      // Get the name of the next signal in the sampling sequence.
      char szName[ABF_ADCNAMELEN+1];
      int nChannel = pFH->nADCSamplingSeq[i];
      ABFU_GetABFString(szName, sizeof(szName), pFH->sADCChannelName[nChannel], ABF_ADCNAMELEN);

      // Search for the name in the list.
      pS = pCfg->TraceList;
      BOOL bFound = FALSE;
      for (UINT j=0; j<uOldTraceCount; j++, pS++)
      {
         bFound = (strcmp(pS->szName, szName)==0);
         if (bFound)
            break;
      }

      // If it was found, go on to the next channel in the sampling sequence.
      if (bFound)
         continue;

      // If gaps were left in the list, fill one in with this signal.
      pS = pCfg->TraceList;
      if (uGaps > 0)
      {
         for (UINT k=0; k<uOldTraceCount; k++, pS++)
            if (pS->szName[0]=='\0')
               break;
         --uGaps;
      }
      else
      {
         // Add a new signal on the end.
         pS += uOldTraceCount;
         ++uOldTraceCount;
      }
      _SetupSignal(pFH, pS, i, uNewTraceCount);
   }

   // If the math channel is enabled, check whether it is already there, if not add it.
   if (bMathChannel)
   {
      char szName[ABF_ADCNAMELEN+1];
      LoadString(g_hInstance, IDS_MATHCHANNEL, szName, sizeof(szName));

      pS = pCfg->TraceList;
      BOOL bFound = FALSE;
      for (UINT j=0; j<uOldTraceCount; j++, pS++)
      {
         bFound = (strcmp(pS->szName, szName)==0);
         if (bFound)
            break;
      }
      if (!bFound)
      {
         // If gaps were left in the list, fill one in with this signal.
         pS = pCfg->TraceList;
         if (uGaps > 0)
         {
            for (UINT k=0; k<uOldTraceCount; k++, pS++)
               if (pS->szName[0]=='\0')
                  break;
            --uGaps;
         }
         else
         {
            // Add a new signal on the end.
            pS += uOldTraceCount;
            ++uOldTraceCount;
         }
         _SetupMathSignal(pFH, pS, uNewTraceCount);
      }
   }

   // Strip out any remaining gaps.
   if (uGaps)
   {
      ABFSignal *pDest = pCfg->TraceList;
      ABFSignal *pSrce = pCfg->TraceList;

      for (int i=0; i<uNewTraceCount; i++)
      {
         // Skip any gaps.
         while (pSrce->szName[0]=='\0')
         {
            pSrce++;
            uGaps--;
            uOldTraceCount--;
         }
         *pDest++ = *pSrce++;
      }
   }
   ASSERT(uOldTraceCount-uNewTraceCount==uGaps);
      
   pCfg->nTraceCount = short(uNewTraceCount);
   memset(pCfg->TraceList+uNewTraceCount, 0, sizeof(pCfg->TraceList) - uNewTraceCount * sizeof(ABFSignal));
   
   // Check recommended scope mode against the value in the Scope configuration.      
   WORD wScopeMode = _GetDefaultScopeMode(pFH);
   BOOL bRval = (wScopeMode == pCfg->wScopeMode);
   pCfg->wScopeMode = wScopeMode;
   return bRval;
}
*/
//==============================================================================================
// FUNCTION:   GetADCtoUUFactors
// PURPOSE:    Calculates the scaling factors used to convert ADC values to UserUnits.
// PARAMETERS:
//    nChannel        - The physical channel number to get the factors for.
//    pfADCToUUFactor - Pointers to return locations for scale and offset.
//    pfADCToUUShift    UserUnits = ADCValue * fADCToUUFactor + fADCToUUShift;
//
void WINAPI ABFH_GetADCtoUUFactors( const ABFFileHeader *pFH, int nChannel, 
                                    float *pfADCToUUFactor, float *pfADCToUUShift )
{
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(pfADCToUUFactor);
//   WPTRASSERT(pfADCToUUShift);
   ASSERT(nChannel < ABF_ADCCOUNT);

   float fTotalScaleFactor = pFH->fInstrumentScaleFactor[nChannel] *
                             pFH->fADCProgrammableGain[nChannel];
   if (pFH->nSignalType != 0)
      fTotalScaleFactor *= pFH->fSignalGain[nChannel];

   // Adjust for the telegraphed gain.
   if( pFH->nTelegraphEnable[nChannel] )
      fTotalScaleFactor *= pFH->fTelegraphAdditGain[nChannel];

   ASSERT(fTotalScaleFactor != 0.0F);
   if (fTotalScaleFactor==0.0F)
      fTotalScaleFactor = 1.0F;

   // InputRange and InputOffset is the range and offset of the signal in
   // user units when it hits the Analog-to-Digital converter

   float fInputRange = pFH->fADCRange / fTotalScaleFactor;
   float fInputOffset= -pFH->fInstrumentOffset[nChannel];
   if (pFH->nSignalType != 0)
      fInputOffset += pFH->fSignalOffset[nChannel];

   *pfADCToUUFactor = fInputRange / pFH->lADCResolution;
   *pfADCToUUShift  = -fInputOffset;
}
/*
//==============================================================================================
// FUNCTION:   ABFH_GetADCDisplayRange
// PURPOSE:    Calculates the upper and lower limits of the display given the display
//             amplification and offset in the ABF header for this channel.
// PARAMETERS:
//    nChannel   - The physical channel number to get the factors for.
//    pfUUTop    - Pointers to return locations for top and bottom of the display.
//    pfUUBottom
//
void WINAPI ABFH_GetADCDisplayRange( const ABFFileHeader *pFH, int nChannel, 
                                     float *pfUUTop, float *pfUUBottom)
{
   // Verify that parameters are reasonable.
   ABFH_ASSERT(pFH);
   WPTRASSERT(pfUUTop);
   WPTRASSERT(pfUUBottom);
   ASSERT(nChannel < ABF_ADCCOUNT);

   ABFH_GainOffsetToDisplayRange( pFH, nChannel, 
                                  pFH->fADCDisplayAmplification[nChannel],
                                  pFH->fADCDisplayOffset[nChannel],
                                  pfUUTop, pfUUBottom);
}

//==============================================================================================
// FUNCTION: GetInputRangeAndOffset
// PURPOSE:  Returns the input range and offset for the channel.
// NOTES:    If channel < 0 the values for the math channel are returned.
//
static void GetInputRangeAndOffset(const ABFFileHeader *pFH, int nChannel, 
                                   float *pfInputRange, float *pfInputOffset)
{
   float fInputRange  = 0.0F;
   float fInputOffset = 0.0F;

   if (nChannel >= 0)
   {
      ASSERT(nChannel < ABF_ADCCOUNT);

      float fADCToUUFactor, fADCToUUShift;
      ABFH_GetADCtoUUFactors(pFH, nChannel, &fADCToUUFactor, &fADCToUUShift);

      fInputRange  = fADCToUUFactor * (pFH->lADCResolution * 2 - 1);
      fInputOffset = -fADCToUUShift;
   }
   else  // if nChannel < 0 it refers to the math channel.
   {
      ASSERT(pFH->nArithmeticEnable);
      fInputRange  = pFH->fArithmeticUpperLimit - pFH->fArithmeticLowerLimit;
      fInputOffset = (pFH->fArithmeticUpperLimit + pFH->fArithmeticLowerLimit)/2;
   }
   *pfInputRange  = fInputRange;
   *pfInputOffset = fInputOffset;
}

//==============================================================================================
// FUNCTION:   ABFH_GainOffsetToDisplayRange
// PURPOSE:    Converts a display range to the equivalent gain and offset factors.
// PARAMETERS:
//    pFH            - The file header that contains scaling factors etc.
//    nChannel       - The physical channel number to get the factors for.
//    fDisplayGain   - The gain & offset pair to convert.
//    fDisplayOffset
//    pfUUTop        - Pointers to return locations for top and bottom of the display.
//    pfUUBottom
//
void WINAPI ABFH_GainOffsetToDisplayRange( const ABFFileHeader *pFH, int nChannel, 
                                           float fDisplayGain, float fDisplayOffset,
                                           float *pfUUTop, float *pfUUBottom)
{
   // Verify that parameters are reasonable.
   ABFH_ASSERT(pFH);
   WPTRASSERT(pfUUTop);
   WPTRASSERT(pfUUBottom);
   ASSERT(nChannel < ABF_ADCCOUNT);

   float fInputRange, fInputOffset;
   GetInputRangeAndOffset(pFH, nChannel, &fInputRange, &fInputOffset);

   // DisplayRange and DisplayOffset are the range and offset of the signal
   // in user units when it is about to be displayed.

   float fUURange  = (float)fabs(fInputRange) / fDisplayGain;
   float fUUOffset = fInputOffset + fDisplayOffset;

   *pfUUTop    = fUUOffset + fUURange / 2;
   *pfUUBottom = fUUOffset - fUURange / 2;
}

//==============================================================================================
// FUNCTION:   ABFH_DisplayRangeToGainOffset
// PURPOSE:    Converts a display range to the equivalent gain and offset factors.
// PARAMETERS:
//    pFH             - The file header that contains scaling factors etc.
//    nChannel        - The physical channel number to get the factors for.
//    fUUTop          - The current values of the top and bottom of the display.
//    fUUBottom
//    pfDisplayGain   - Pointers to return locations for the gain and offset factors.
//    pfDisplayOffset
//
void WINAPI ABFH_DisplayRangeToGainOffset( const ABFFileHeader *pFH, int nChannel, 
                                           float fUUTop, float fUUBottom,
                                           float *pfDisplayGain, float *pfDisplayOffset)
{
   // Verify that parameters are reasonable.
   ABFH_ASSERT(pFH);
   WPTRASSERT(pfDisplayGain);
   WPTRASSERT(pfDisplayOffset);
   ASSERT(nChannel < ABF_ADCCOUNT);

   float fUURange  = (fUUTop - fUUBottom);
   float fUUOffset = (fUUTop + fUUBottom) / 2;

   float fInputRange, fInputOffset;
   GetInputRangeAndOffset(pFH, nChannel, &fInputRange, &fInputOffset);

   *pfDisplayGain   = (float)fabs(fInputRange) / fUURange;
   *pfDisplayOffset = fUUOffset - fInputOffset;
}

//==============================================================================================
// FUNCTION:   ClipADCUUValue
// PURPOSE:    Limiting the UU value to the range of the A/D converter.
//
void WINAPI ABFH_ClipADCUUValue(const ABFFileHeader *pFH, int nChannel, float *pfUUValue)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pfUUValue);
   ASSERT(nChannel < ABF_ADCCOUNT);

   // Get the conversion factors.
   float fADCToUUFactor, fADCToUUShift;
   ABFH_GetADCtoUUFactors(pFH, nChannel, &fADCToUUFactor, &fADCToUUShift);
   
   // Calculate the extremes of the range.   
   float fUUMax = (pFH->lADCResolution-1) * fADCToUUFactor + fADCToUUShift;
   float fUUMin = (-pFH->lADCResolution) * fADCToUUFactor + fADCToUUShift;
   if (fUUMax < fUUMin)
   {
      float fTemp = fUUMax;
      fUUMax = fUUMin;
      fUUMin = fTemp;
   }
   
   // Clip the value to the range of the A/D converter.
   float fUUValue = *pfUUValue;
   if (fUUValue > fUUMax)
      fUUValue = fUUMax;
   if (fUUValue < fUUMin)
      fUUValue = fUUMin;
   *pfUUValue = fUUValue;
}
*/
//==============================================================================================
// FUNCTION:   GetDACtoUUFactors
// PURPOSE:    Calculates the scaling factors used to convert DAC values to UserUnits.
// PARAMETERS:
//    nChannel        - The physical channel number to get the factors for.
//    pfDACToUUFactor - Pointers to return locations for scale and offset.
//    pfDACToUUShift    UserUnits = DACValue * fDACToUUFactor + fDACToUUShift;
//
void WINAPI ABFH_GetDACtoUUFactors( const ABFFileHeader *pFH, int nChannel, 
                                    float *pfDACToUUFactor, float *pfDACToUUShift )
{
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(pfDACToUUShift);
//   WPTRASSERT(pfDACToUUFactor);
   ASSERT(nChannel < ABF_DACCOUNT);

   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );
   {
      // Prevent accidental use of pFH.
      int pFH = 0;   pFH = pFH;

      float fScaleFactor       = NewFH.fDACScaleFactor[nChannel];
      float fCalibrationFactor = NewFH.fDACCalibrationFactor[nChannel];
      float fCalibrationOffset = NewFH.fDACCalibrationOffset[nChannel];

      // OutputRange and OutputOffset is the range and offset of the signal in
      // user units when it hits the Digital-to-Analog converter

      float fOutputRange  = NewFH.fDACRange * fScaleFactor;
      float fOutputOffset = 0.0F;  // NewFH.fWaveformOffset;

      // nCalibratedDAC  = nDAC * NewFH.fDACCalibrationFactor[i] + NewFH.fDACCalibrationOffset[i];
      // fUU             = nCalibratedDAC * pfDACToUUFactor + pfDACToUUShift;

      float fDACToUUFactor = fOutputRange / NewFH.lDACResolution;
      float fDACToUUShift  = fOutputOffset;

      *pfDACToUUFactor = fDACToUUFactor * fCalibrationFactor;
      *pfDACToUUShift  = fDACToUUShift + fDACToUUFactor * fCalibrationOffset;
   }
}
/*
//==============================================================================================
// FUNCTION:   ABFH_ClipDACUUValue
// PURPOSE:    Limiting the UU value to the range of the D/A converter.
//
void WINAPI ABFH_ClipDACUUValue(const ABFFileHeader *pFH, int nChannel, float *pfUUValue)
{
   ABFH_ASSERT(pFH);
   WPTRASSERT(pfUUValue);
   ASSERT(nChannel < ABF_DACCOUNT);

   // Get the conversion factors.
   float fDACToUUFactor, fDACToUUShift;
   ABFH_GetDACtoUUFactors(pFH, nChannel, &fDACToUUFactor, &fDACToUUShift);
   
   // Calculate the extremes of the range.   
   float fUUMax = (pFH->lDACResolution-1) * fDACToUUFactor + fDACToUUShift;
   float fUUMin = (-pFH->lDACResolution) * fDACToUUFactor + fDACToUUShift;
   if (fUUMax < fUUMin)
   {
      float fTemp = fUUMax;
      fUUMax = fUUMin;
      fUUMin = fTemp;
   }
   
   // Clip the value to the range of the D/A converter.
   float fUUValue = *pfUUValue;
   if (fUUValue > fUUMax)
      fUUValue = fUUMax;
   if (fUUValue < fUUMin)
      fUUValue = fUUMin;
   *pfUUValue = fUUValue;
}
*/
#define AVERYBIGNUMBER 3.402823466E+38
//===============================================================================================
// FUNCTION: ABFH_GetMathValue
// PURPOSE:  Evaluate the Math expression for the given UU values.
// RETURNS:  TRUE if the expression could be evaluated OK.
//           FALSE if a divide by zero occurred.
//
BOOL WINAPI ABFH_GetMathValue(const ABFFileHeader *pFH, float fA, float fB, float *pfRval)
{
//   ABFH_ASSERT(pFH);
//   WPTRASSERT(pfRval);
   double dResult = 0.0;          // default return response
   double dLeftVal, dRightVal;
   BOOL bRval = TRUE;

   if (pFH->nArithmeticExpression == ABF_SIMPLE_EXPRESSION)
   {
      dLeftVal  = pFH->fArithmeticK1 * fA + pFH->fArithmeticK2;
      dRightVal = pFH->fArithmeticK3 * fB + pFH->fArithmeticK4;
   }
   else
   {
      double dRatio;
      if (fB + pFH->fArithmeticK6 != 0.0F)
         dRatio = (fA + pFH->fArithmeticK5) / (fB + pFH->fArithmeticK6);
      else if (fA + pFH->fArithmeticK5 > 0.0F)
      {
         dRatio = AVERYBIGNUMBER;
         bRval = FALSE;
      }
      else
      {
         dRatio = -AVERYBIGNUMBER;
         bRval = FALSE;
      }
      dLeftVal  = pFH->fArithmeticK1 * dRatio + pFH->fArithmeticK2;
      dRightVal = pFH->fArithmeticK3 * dRatio + pFH->fArithmeticK4;
   }

   switch (pFH->sArithmeticOperator[0])
   {
      case '+':
         dResult = dLeftVal + dRightVal;
         break;
      case '-':
         dResult = dLeftVal - dRightVal;
         break;
      case '*':
         dResult = dLeftVal * dRightVal;
         break;
      case '/':
         if (dRightVal != 0.0)
            dResult = dLeftVal / dRightVal;
         else if (dLeftVal > 0)
         {
            dResult = pFH->fArithmeticUpperLimit;
            bRval = FALSE;
         }
         else
         {
            dResult = pFH->fArithmeticLowerLimit;
            bRval = FALSE;
         }
         break;

      default:
         //ERRORMSG1("Unexpected operator '%c'.", pFH->sArithmeticOperator[0]);
         break;
   }

   if (dResult < pFH->fArithmeticLowerLimit)
      dResult = pFH->fArithmeticLowerLimit;
   else if (dResult > pFH->fArithmeticUpperLimit)
      dResult = pFH->fArithmeticUpperLimit;

   if (pfRval)
      *pfRval = (float)dResult;
   return bRval;
}
/*

//===============================================================================================
// FUNCTION: ABFH_GetMathChannelName
// PURPOSE:  Gets the name to be used when displaying the math channel. This is exported here
//           for lack of a better place to put it. It is important that all signal names are
//           unique, so the user has to be prevented for choosing a name that matches a predefined
//           signal name.
// RETURNS:  The number of non-zero characters copied to the passed buffer.
//
int WINAPI ABFH_GetMathChannelName(char *pszName, UINT uNameLen)
{
   return LoadString(g_hInstance, IDS_MATHCHANNEL, pszName,  uNameLen);
}
*/
//===============================================================================================
// FUNCTION: ABFH_ParamReader
// PURPOSE:  Read parameters from all versions of ABF files, converting them to the current
//           header format.
// PARAMETERS:
//    hFile - DOS handle to an open data file.
//    pFH   - Address of an ABF file header structure to fill in.
// RETURNS:  TRUE  = Parameters read OK
//           FALSE = Parameters were invalid - error number returned in pnError
//                   (see ABFHEADR.h for symbolic constants defining error values)
BOOL WINAPI ABFH_ParamReader(FILEHANDLE hFile, ABFFileHeader *pFH, int *pnError)
{
//   ABFH_WASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader NewFH;
   ABFH_PromoteHeader( &NewFH, pFH );
   
   // Get the file version information.
   UINT  uFileType;
   BOOL  bMSBinFormat;
   float fFileVersion;
   if (!OLDH_GetFileVersion(hFile, &uFileType, &fFileVersion, &bMSBinFormat))
      ERRORRETURN(pnError, ABFH_EUNKNOWNFILETYPE);

   // Get the file length for parameter validation, then seek back to the start of the file.
   long lFileLength = c_SetFilePointer(hFile, 0, NULL, FILE_END);
   c_SetFilePointer(hFile, 0L, NULL, FILE_BEGIN);

   // If the file is not an ABF file, read in the file header and convert it to the current
   // header structure.
   if (uFileType != ABF_ABFFILE)
   {
      BOOL bRet = OLDH_ReadOldHeader(hFile, uFileType, bMSBinFormat, &NewFH, lFileLength, pnError);
      ABFH_DemoteHeader( pFH, &NewFH );
      return bRet;
   }

   // Files after this major version are not supported.
   if ((int)fFileVersion > (int)ABF_CURRENTVERSION)
      ERRORRETURN(pnError, ABFH_EINVALIDFILE);

   if ((int)fFileVersion < (int)ABF_CURRENTVERSION)
   {
      // Major release number is behind the current version, some rearranging
      // of parameters and fixups may be required.
      // (not necessary at present as there was no ABF version 0.x)
      ERRORRETURN(pnError, ABFH_EINVALIDFILE);
   }

   // Read the file header for the file directly into the passed data
   // structure - only read in 2k if old file or old header

   UINT uHeaderSize = ABF_OLDHEADERSIZE;
   if( ABFH_IsNewHeader(&NewFH) )
      uHeaderSize = ABF_HEADERSIZE;

   if( fFileVersion < ABF_V16 )
      uHeaderSize = ABF_OLDHEADERSIZE;

   BOOL bReadOK = ABFU_ReadFile(hFile, &NewFH, uHeaderSize);

   // Check various crucial parameters for sensible values.
   if ((NewFH.lSynchArraySize <= 0) || (NewFH.lSynchArrayPtr <= 0))
   {
      NewFH.lSynchArraySize = 0;
      NewFH.lSynchArrayPtr  = 0;
   }
   if (NewFH.fADCRange<=A_VERY_SMALL_NUMBER)
      NewFH.fADCRange = 10.0F;
   if (NewFH.fDACRange<=A_VERY_SMALL_NUMBER)
      NewFH.fDACRange = 10.0F;

   if (!bReadOK)
      ERRORRETURN(pnError, ABFH_EHEADERREAD);

   // Initialize parameters added in minor version revisions if necessary.
   if (fFileVersion < ABF_CURRENTVERSION)
      OLDH_ABFtoCurrentVersion(&NewFH);

   NewFH.fHeaderVersionNumber = ABF_CURRENTVERSION;
   NewFH.lHeaderSize          = ABF_HEADERSIZE;
   NewFH.nFileType            = ABF_ABFFILE;
   NewFH.nMSBinFormat         = FALSE;

   // Sanity check on various problematic parameters.
   if ((NewFH.nTrialTriggerSource != ABF_TRIALTRIGGER_EXTERNAL) &&
       (NewFH.nTrialTriggerSource != ABF_TRIALTRIGGER_SPACEBAR))
      NewFH.nTrialTriggerSource = ABF_TRIALTRIGGER_NONE;

   if (NewFH.fAverageWeighting < 0.001F)
      NewFH.fAverageWeighting = 0.1F;

   for( UINT i=0; i<ABF_WAVEFORMCOUNT; i++ )
   {
      if (NewFH.nPNPolarity[i] == 0)
         NewFH.nPNPolarity[i] = ABF_PN_SAME_POLARITY;

      // ATF stimulus files assume time in the first column, so ABF_DACFILE_SKIPFIRSTSWEEP is 
      // no longer allowed - replace it with ABF_DACFILE_USEALLSWEEPS.
      if( NewFH.lDACFileEpisodeNum[i] == ABF_DACFILE_SKIPFIRSTSWEEP )
         NewFH.lDACFileEpisodeNum[i] = ABF_DACFILE_USEALLSWEEPS;

      // Ensure the waveform source makes sense.
      if( (NewFH.nWaveformEnable[i] == FALSE) && 
          (NewFH.nWaveformSource[i] == ABF_WAVEFORMDISABLED) )
         NewFH.nWaveformSource[i] = ABF_EPOCHTABLEWAVEFORM;
   }

   if (NewFH.nStatsSmoothing < 1)
      NewFH.nStatsSmoothing = 1;

   if ((NewFH.nLevelHysteresis == 0) && (NewFH.lTimeHysteresis == 0))
   {
      NewFH.nLevelHysteresis = DEFAULT_LEVEL_HYSTERESIS;
      NewFH.lTimeHysteresis  = DEFAULT_TIME_HYSTERESIS;
   }

   if ( !memchr(c_szValidOperators, NewFH.sArithmeticOperator[0], sizeof(c_szValidOperators)-1) )
   {
      NewFH.sArithmeticOperator[0] = '+';
      NewFH.sArithmeticOperator[1] = ' ';
   }
   if (UINT(NewFH.nArithmeticExpression) > 1)
      NewFH.nArithmeticExpression = 0;


   
   NewFH.lFileStartDate = ABFU_FixFileStartDate( NewFH.lFileStartDate );

   ABFH_DemoteHeader( pFH, &NewFH );

   return TRUE;
}
/*
//===============================================================================================
// FUNCTION: ABFH_ParamWriter
// PURPOSE:  Write file header parameters to an ABF file.
// PARAMETERS:
//    hFile - DOS handle to an open data file.
//    pFH   - Address of an ABF file header structure to write out.
// RETURNS:  TRUE  = Parameters written OK
//           FALSE = Write error - error number returned in pnError
//                   (see ABFHEADR.h for symbolic constants defining error values)

BOOL WINAPI ABFH_ParamWriter(HANDLE hFile, ABFFileHeader *pFH, int *pnError)
{
   ABFH_WASSERT(pFH);

   // Take a copy of the passed in header to ensure it is 6k long.
   ABFFileHeader CopyFH;
   ABFH_PromoteHeader( &CopyFH, pFH );

   // Set the parameter and data file version number to the current version number
   CopyFH.lFileSignature       = ABF_NATIVESIGNATURE;
   CopyFH.fFileVersionNumber   = ABF_CURRENTVERSION;
   CopyFH.fHeaderVersionNumber = ABF_CURRENTVERSION;
   CopyFH.nFileType            = ABF_ABFFILE;
   CopyFH.nMSBinFormat         = FALSE;

   // If there is data in the file, ensure the header remains the same size as the original.
   if( CopyFH.lDataSectionPtr > 0 )
      CopyFH.lHeaderSize = min( CopyFH.lHeaderSize, CopyFH.lDataSectionPtr * ABF_BLOCKSIZE );

   ABFH_DemoteHeader( pFH, &CopyFH );
   
   // Seek to the start of the file.
   SetFilePointer(hFile, 0L, NULL, FILE_BEGIN);

   // Write the header out to the file.
   DWORD dwBytesWritten = 0;

   if (!WriteFile(hFile, pFH, pFH->lHeaderSize, &dwBytesWritten, NULL))   
      ERRORRETURN(pnError, ABFH_EHEADERWRITE);

   return TRUE;
}
*/
//===============================================================================================
// FUNCTION: ABFH_GetErrorText
// PURPOSE:  Returns a textual description of an error number returned by another function in
//           this module.
// RETURNS:  Number of characters copied to the buffer.
//
BOOL WINAPI ABFH_GetErrorText( int nError, char *sTxtBuf, UINT uMaxLen)
{
//   ARRAYASSERT(sTxtBuf, uMaxLen);
   if (uMaxLen < 2)
   {
//      ERRORMSG("String too short!");
      return FALSE;
   }

   BOOL rval = TRUE;        // OK return value
   if (!c_LoadString(g_hInstance, nError, sTxtBuf, uMaxLen))
   {
      char szTemplate[80];
      c_LoadString(g_hInstance, IDS_ENOMESSAGESTR, szTemplate, sizeof(szTemplate));

      char szErrorMsg[128];
      sprintf(szErrorMsg, szTemplate, nError);
//      ERRORMSG(szErrorMsg);

      strncpy(sTxtBuf, szErrorMsg, uMaxLen-1);
      sTxtBuf[uMaxLen-1] = '\0';
      rval = FALSE;
   }
   return rval;
}

//===============================================================================================
// FUNCTION: ABFH_SynchTimeToMS
// PURPOSE:  Converts the count in a synch entry to time in milli-seconds.
//
void WINAPI ABFH_SynchCountToMS(const ABFFileHeader *pFH, UINT uCount, double *pdTimeMS)
{
   ABFH_ASSERT(pFH);
   double dSynchTime = (pFH->fSynchTimeUnit > 0.0F ? pFH->fSynchTimeUnit : ABFH_GetFirstSampleInterval(pFH));

   // us -> ms
   dSynchTime /= 1E3;
   *pdTimeMS = uCount * dSynchTime;
}

//===============================================================================================
// FUNCTION: ABFH_MSToSynchCount
// PURPOSE:  Converts a time value to a synch time count.
//
UINT WINAPI ABFH_MSToSynchCount(const ABFFileHeader *pFH, double dTimeMS)
{
   ABFH_ASSERT(pFH);
   double dSynchTimeMS = ( pFH->fSynchTimeUnit > 0.0F ? pFH->fSynchTimeUnit : ABFH_GetFirstSampleInterval(pFH) )/1E3;
   return UINT(dTimeMS / dSynchTimeMS + 0.5);
}

//===============================================================================================
// FUNCTION: ABFH_IsNewHeader
// PURPOSE:  Checks if the ABFFileHeader is a new (6k) or old (2k) header.
//
BOOL WINAPI ABFH_IsNewHeader(const ABFFileHeader *pFH)
{
   ASSERT(pFH);

   BOOL bNewHeader = ( pFH->lFileSignature == ABF_NATIVESIGNATURE &&
                       pFH->fHeaderVersionNumber == ABF_CURRENTVERSION &&
                       pFH->lHeaderSize == ABF_HEADERSIZE );
   return bNewHeader;
}

//===============================================================================================
// FUNCTION: ClipToShort
// PURPOSE:  Forces a long value to a short, clipping it if needed.
//
static short ClipToShort( long lValue )
{
   long lClipped = min( lValue , c_lMaxShort );

   return short(lClipped);
}

//===============================================================================================
// FUNCTION: ABFH_DemoteHeader
// PURPOSE:  Demotes a 1.6 file header to a 1.5 file header.
//
void WINAPI ABFH_DemoteHeader(ABFFileHeader *pOut, const ABFFileHeader *pIn )
{
//   ABFH_ASSERT(pIn );
//   ABFH_WASSERT(pOut);

   // If both are new headers.
   if( ABFH_IsNewHeader(pIn) && ABFH_IsNewHeader(pOut) )
   {
      // Perform a simple copy
      *pOut = *pIn;
   }
   else
   {
      // We are copying from a new header to an old header.

      // Copy the first 2k and demote to ABF v1.5
      memcpy( pOut, pIn, ABF_OLDHEADERSIZE );
   
      // Demote the file version.
      pOut->fFileVersionNumber   = ABF_PREVIOUSVERSION;
      pOut->fHeaderVersionNumber = ABF_PREVIOUSVERSION;
      pOut->lHeaderSize          = ABF_OLDHEADERSIZE;
   }

   // Demote regardless of whether the header is an old or current version.
   // If the header version is current we preserve the integrity of old data fields for 
   // backwards compatiblity ... GRB
   
   // If there are no outputs on DAC 1, then set nActiveDACChannel to 0.
   UINT bForceToDAC0 = FALSE;
   if( !pIn->nWaveformEnable[1] &&
       !pIn->nDigitalEnable )
      bForceToDAC0 = TRUE;
   
   if( bForceToDAC0 &&
       (pIn->nActiveDACChannel != 0) )
   {
      pOut->nActiveDACChannel = 0;
//      TRACE2( "ABFH_DemoteHeader: nActiveDACChannel changed from %d to %d.\n", 
//              pIn->nActiveDACChannel, pOut->nActiveDACChannel );
   }
   UINT uDAC = pOut->nActiveDACChannel;
   
   pOut->_lDACFilePtr         = pIn->lDACFilePtr[uDAC];
   pOut->_lDACFileNumEpisodes = pIn->lDACFileNumEpisodes[uDAC];
   pOut->_nWaveformSource     = pIn->nWaveformSource[uDAC];
   if( pIn->nWaveformEnable[uDAC] == FALSE )
      pOut->_nWaveformSource = ABF_WAVEFORMDISABLED;
 		
   pOut->_nInterEpisodeLevel  = pIn->nInterEpisodeLevel[uDAC];
   
   // Waveform epoch parameters.
   for( UINT i=0; i<ABF_EPOCHCOUNT; i++ )
   {
      pOut->_nEpochType[i]         = pIn->nEpochType[uDAC][i];
      pOut->_fEpochInitLevel[i]    = pIn->fEpochInitLevel[uDAC][i];
      pOut->_fEpochLevelInc[i]     = pIn->fEpochLevelInc[uDAC][i];
      pOut->_nEpochInitDuration[i] = ClipToShort( pIn->lEpochInitDuration[uDAC][i] );
      pOut->_nEpochDurationInc[i]  = ClipToShort( pIn->lEpochDurationInc[uDAC][i] );
   }
   
   // Stimulus file parameters.
   pOut->_fDACFileScale      = pIn->fDACFileScale[uDAC];
   pOut->_fDACFileOffset     = pIn->fDACFileOffset[uDAC];
   pOut->_nDACFileEpisodeNum = ClipToShort( pIn->lDACFileEpisodeNum[uDAC] );
   pOut->_nDACFileADCNum     = pIn->nDACFileADCNum[uDAC];
   strncpy( pOut->_sDACFilePath, pIn->sDACFilePath[uDAC], ABF_DACFILEPATHLEN );

   // Presweep Trains (formerly called Conditioning Train)
   ASSERT( pOut->_nConditChannel >= 0 );
   ASSERT( pOut->_nConditChannel < ABF_WAVEFORMCOUNT );

   pOut->_nConditEnable     = pIn->nConditEnable[0] || pIn->nConditEnable[1];
   pOut->_nConditChannel    = short( pIn->nConditEnable[0] == 1 ? 0 : 1 );
   pOut->_fBaselineDuration = pIn->fBaselineDuration[pOut->_nConditChannel];
   pOut->_fBaselineLevel    = pIn->fBaselineLevel[pOut->_nConditChannel];
   pOut->_fStepDuration     = pIn->fStepDuration[pOut->_nConditChannel];
   pOut->_fStepLevel        = pIn->fStepLevel[pOut->_nConditChannel];
   pOut->_fPostTrainLevel   = pIn->fPostTrainLevel[pOut->_nConditChannel];

   
   // P/N Leak subtraction parameters.
   pOut->_nPNEnable       = pIn->nPNEnable[uDAC];
   pOut->_nPNPolarity     = pIn->nPNPolarity[uDAC];
   pOut->_nPNADCNum       = pIn->nPNADCSamplingSeq[uDAC][0];
   pOut->_fPNHoldingLevel = pIn->fPNHoldingLevel[uDAC];

   // User list parameters.
   pOut->_nListEnable      = pIn->nULEnable[uDAC];
   pOut->_nParamToVary     = pIn->nULParamToVary[uDAC];
   strncpy( pOut->_sParamValueList, pIn->sULParamValueList[uDAC], ABF_VARPARAMLISTLEN );

   // FIX FIX FIX PRC DEBUG Telegraph changes - check !
   // Telegraph information.
   pOut->_nAutosampleEnable      = pIn->nTelegraphEnable[pOut->_nAutosampleADCNum];
   pOut->_nAutosampleInstrument  = pIn->nTelegraphInstrument[pOut->_nAutosampleADCNum];
   pOut->_fAutosampleAdditGain   = pIn->fTelegraphAdditGain[pOut ->_nAutosampleADCNum];
   pOut->_fAutosampleFilter      = pIn->fTelegraphFilter[pOut->_nAutosampleADCNum];
   pOut->_fAutosampleMembraneCap = pIn->fTelegraphMembraneCap[pOut->_nAutosampleADCNum];
   
   // File Comment.
   strncpy( pOut->_sFileComment, pIn->sFileComment, ABF_OLDFILECOMMENTLEN );
   
   // Demoting the statistics regions
   pOut->_nAutopeakEnable       = pIn->nStatsEnable;
   pOut->_nAutopeakPolarity     = pIn->nStatsChannelPolarity[0];
   pOut->_nAutopeakSearchMode   = pIn->nStatsSearchMode[0];
   pOut->_lAutopeakStart        = pIn->lStatsStart[0];
   pOut->_lAutopeakEnd          = pIn->lStatsEnd[0];
   pOut->_nAutopeakSmoothing    = pIn->nStatsSmoothing;
   pOut->_nAutopeakBaseline     = pIn->nStatsBaseline;
   pOut->_lAutopeakBaselineStart= pIn->lStatsBaselineStart;
   pOut->_lAutopeakBaselineEnd  = pIn->lStatsBaselineEnd;
   pOut->_lAutopeakMeasurements = pIn->lStatsMeasurements[0];

   // Accept only the first channel selection.
   UINT nStatsADCNum = pIn->nStatsActiveChannels;
   unsigned short uADCChannel = 0; //16 Channels
   while(  uADCChannel < ABF_ADCCOUNT )
   {
      if ( nStatsADCNum & 0x0001 )
      {
         break;
      }
      //Shift zero bit by 1 to the left.
      nStatsADCNum = nStatsADCNum >> 1;
      uADCChannel++;
   }
   pOut->_nAutopeakADCNum = uADCChannel;
}

//===============================================================================================
// FUNCTION: ABFH_PromoteHeader
// PURPOSE:  Promotes a 1.5 file header to a 1.6 file header.
//
void WINAPI ABFH_PromoteHeader(ABFFileHeader *pOut, const ABFFileHeader *pIn )
{
//   ABFH_ASSERT(pIn );
//  WPTRASSERT(pOut);
   
   // If both are new headers.
   if( ABFH_IsNewHeader(pIn) && ABFH_IsNewHeader(pOut) )
   {
      // Perform a simple copy
      *pOut = *pIn;
      return;
   }

   // We are copying from an old header to a new header.
   // Copy the first 2k and clear the rest.
   memset( pOut, 0, ABF_HEADERSIZE);
   memcpy( pOut, pIn, ABF_OLDHEADERSIZE );

   // Promote ABF header parameters.
   UINT uDAC = (UINT)pIn->nActiveDACChannel;
   if( uDAC >= ABF_WAVEFORMCOUNT )
      uDAC = 0;

   pOut->lDACFilePtr[uDAC]         = pIn->_lDACFilePtr;
   pOut->lDACFileNumEpisodes[uDAC] = pIn->_lDACFileNumEpisodes;
   pOut->nInterEpisodeLevel[uDAC]  = pIn->_nInterEpisodeLevel;

   pOut->nWaveformSource[uDAC]     = short( (pIn->_nWaveformSource == ABF_DACFILEWAVEFORM) 
                                     ? ABF_DACFILEWAVEFORM : ABF_EPOCHTABLEWAVEFORM );
   
   pOut->nWaveformEnable[uDAC]     = (pIn->_nWaveformSource != ABF_WAVEFORMDISABLED);

   for( UINT i=0; i<ABF_EPOCHCOUNT; i++ )
   {
      pOut->nEpochType[uDAC][i]         = pIn->_nEpochType[i];
      pOut->fEpochInitLevel[uDAC][i]    = pIn->_fEpochInitLevel[i];
      pOut->fEpochLevelInc[uDAC][i]     = pIn->_fEpochLevelInc[i];
      pOut->lEpochInitDuration[uDAC][i] = pIn->_nEpochInitDuration[i];
      pOut->lEpochDurationInc[uDAC][i]  = pIn->_nEpochDurationInc[i];
   }

   pOut->fDACFileScale[uDAC]      = pIn->_fDACFileScale;
   pOut->fDACFileOffset[uDAC]     = pIn->_fDACFileOffset;
   pOut->lDACFileEpisodeNum[uDAC] = pIn->_nDACFileEpisodeNum;
   pOut->nDACFileADCNum[uDAC]     = pIn->_nDACFileADCNum;
   strncpy( pOut->sDACFilePath[uDAC], pIn->_sDACFilePath, ABF_DACFILEPATHLEN );

   // If this is a valid header, then check the presweep trains.
   if( (pIn->lFileSignature == ABF_NATIVESIGNATURE) &&
       (pIn->nFileType == ABF_ABFFILE) )
   {
      ASSERT( pIn->_nConditChannel >= 0 );
      ASSERT( pIn->_nConditChannel < ABF_WAVEFORMCOUNT );
   }

   if( uDAC == (UINT)pIn->_nConditChannel )
   {
      pOut->nConditEnable[pIn->_nConditChannel]     = pIn->_nConditEnable;
      pOut->lConditNumPulses[pIn->_nConditChannel]  = pIn->_lConditNumPulses;
      pOut->fBaselineDuration[pIn->_nConditChannel] = pIn->_fBaselineDuration;
      pOut->fBaselineLevel[pIn->_nConditChannel]    = pIn->_fBaselineLevel;
      pOut->fStepDuration[pIn->_nConditChannel]     = pIn->_fStepDuration;
      pOut->fStepLevel[pIn->_nConditChannel]        = pIn->_fStepLevel;
      pOut->fPostTrainLevel[pIn->_nConditChannel]   = pIn->_fPostTrainLevel;
   }

   if( uDAC == (UINT)pIn->nActiveDACChannel )
   {
      // P/N Leak subtraction parameters.
      pOut->nPNEnable[uDAC]            = pIn->_nPNEnable;
      pOut->nPNPolarity[uDAC]          = pIn->_nPNPolarity;
      pOut->fPNHoldingLevel[uDAC]      = pIn->_fPNHoldingLevel;
      pOut->nPNADCSamplingSeq[uDAC][0] = LOBYTE( pIn->_nPNADCNum );

      // Sanity check.
      ASSERT( pOut->nPNADCSamplingSeq[uDAC][0] == pIn->_nPNADCNum );

      // User list parameters.
      pOut->nULEnable[uDAC]        = pIn->_nListEnable;
      pOut->nULParamToVary[uDAC]   = pIn->_nParamToVary;
      strncpy( pOut->sULParamValueList[uDAC], pIn->_sParamValueList, ABF_VARPARAMLISTLEN );
   }

   // DAC Calibration Factors.
   for(int i=0; i<ABF_DACCOUNT; i++ )
   {
      pOut->fDACCalibrationFactor[i] = 1.0F;
      pOut->fDACCalibrationOffset[i] = 0.0F;
   }

   // File Comment.
   strncpy( pOut->sFileComment, pIn->_sFileComment, ABF_OLDFILECOMMENTLEN );

   // Extra 'enable' fields.
   pOut->nCommentsEnable = (pOut->nManualInfoStrategy != ABF_ENV_DONOTWRITE);

   // FIX FIX FIX PRC DEBUG Telegraph changes - check !
   // Telegraph information.
   pOut->nTelegraphEnable[pIn->_nAutosampleADCNum]      = pIn->_nAutosampleEnable;
   pOut->nTelegraphInstrument[pIn->_nAutosampleADCNum]  = pIn->_nAutosampleInstrument;
   pOut->fTelegraphAdditGain[pIn->_nAutosampleADCNum]   = pIn->_fAutosampleAdditGain;
   pOut->fTelegraphFilter[pIn->_nAutosampleADCNum]      = pIn->_fAutosampleFilter;
   pOut->fTelegraphMembraneCap[pIn->_nAutosampleADCNum] = pIn->_fAutosampleMembraneCap;
   
   // Promote the file version.
   pOut->fHeaderVersionNumber = ABF_CURRENTVERSION;
   pOut->lHeaderSize          = ABF_HEADERSIZE;
   
   // Promote statistics regions
   pOut->nStatsEnable          = pIn->_nAutopeakEnable;
   pOut->nStatsSearchMode[0]   = pIn->_nAutopeakSearchMode;
   pOut->lStatsStart[0]        = pIn->_lAutopeakStart;
   pOut->lStatsEnd[0]          = pIn->_lAutopeakEnd;
   pOut->nStatsSmoothing       = pIn->_nAutopeakSmoothing;
   pOut->nStatsBaseline        = pIn->_nAutopeakBaseline;
   pOut->lStatsBaselineStart   = pIn->_lAutopeakBaselineStart;
   pOut->lStatsBaselineEnd     = pIn->_lAutopeakBaselineEnd;
   pOut->lStatsMeasurements[0] = pIn->_lAutopeakMeasurements;

   // Polarity is channel specific
   for ( UINT uChannel = 0; uChannel < ABF_ADCCOUNT; uChannel++ )
      pOut->nStatsChannelPolarity[uChannel] = pIn->_nAutopeakPolarity;;

   // Convert the old channel selection to an active channel.
   UINT nStatsADCNum = pIn->_nAutopeakADCNum;

   // Shift the uBitmask by nStatsADCNum of bits.
   UINT uBitMask = 0x01 << nStatsADCNum;
   pOut->nStatsActiveChannels |= uBitMask;

   // Select statistics region zero.
   pOut->nStatsSearchRegionFlags = ABF_PEAK_SEARCH_REGION0;
   pOut->nStatsSelectedRegion    = 0;
}

//===============================================================================================
// FUNCTION: GetSampleInterval
// PURPOSE:  Gets the sample interval expressed as a double.
//           This prevents round off errors in modifiable ABF files, 
//           where sample intervals are not constrained to be in multiples of 0.5 us.
//
static double GetSampleInterval( const ABFFileHeader *pFH, const UINT uInterval )
{
//   ABFH_ASSERT( pFH );
   ASSERT( uInterval == 1 ||
           uInterval == 2 );

   float fInterval = 0;
   if( uInterval == 1 )
      fInterval = pFH->fADCSampleInterval;
   else if( uInterval == 2 ) 
      fInterval = pFH->fADCSecondSampleInterval;
   else ;
      //ERRORMSG( "ABFH_GetSampleInterval called with invalid parameters !\n" );

   
   // Modifiable ABF allows sample intervals which are not multiples of 0.5 us
   // Attempt to reconstruct the original sample interval to 0.1 us resolution
   // This has no adverse effect for acquisition files and prevents rounding errors in modifable ABF files.
   double dInterval = int((fInterval * pFH->nADCNumChannels) * 10 + 0.5);
   dInterval /= 10 * pFH->nADCNumChannels;

   return dInterval;
}

//===============================================================================================
// FUNCTION: ABFH_GetFirstSampleInterval
// PURPOSE:  Gets the first sample interval expressed as a double.
double WINAPI ABFH_GetFirstSampleInterval( const ABFFileHeader *pFH )
{
   return GetSampleInterval( pFH, 1 );
}

//===============================================================================================
// FUNCTION: ABFH_GetSecondSampleInterval
// PURPOSE:  Gets the second sample interval expressed as a double.
double WINAPI ABFH_GetSecondSampleInterval( const ABFFileHeader *pFH ) 
{
   return GetSampleInterval( pFH, 2 );
}
/*


// **********************************************************************************************
// **********************************************************************************************
//

#ifdef SHOW_STRUCT_SIZES

#define SHOW_STRUCT_SIZE(S) \
static int u##S = AXODBG_printf( "sizeof(%s) = %d\n", #S, sizeof(S))

SHOW_STRUCT_SIZE(ABFFileHeader);
SHOW_STRUCT_SIZE(ABFLogFont);
SHOW_STRUCT_SIZE(ABFSignal);
SHOW_STRUCT_SIZE(ABFScopeConfig);
SHOW_STRUCT_SIZE(ABFSynch);
SHOW_STRUCT_SIZE(ABFTag);
SHOW_STRUCT_SIZE(ABFVoiceTagInfo);
SHOW_STRUCT_SIZE(ABFDelta);

#else
// If there is a compiler error here, it means the struct in question is the wrong size
// - fix it in ABFHEADR.H

char FH     [sizeof(ABFFileHeader) == ABF_HEADERSIZE ? 1 : 0];
char LogF   [sizeof(ABFLogFont) == 40 ? 1 : 0];
char Signal [sizeof(ABFSignal) == 34 ? 1 : 0];
char SC     [sizeof(ABFScopeConfig) == 769 ? 1 : 0];
char Synch  [sizeof(ABFSynch) == 8 ? 1 : 0];
char Tag    [sizeof(ABFTag) == 64 ? 1 : 0];
char VTag   [sizeof(ABFVoiceTagInfo) == 32 ? 1 : 0];
char Delta  [sizeof(ABFDelta) == 12 ? 1 : 0];

#endif   // SHOW_STRUCT_SIZES
*/
