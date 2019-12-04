//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// This is OLDHEADR.CPP; the routines that cope with reading the data file
// parameters block for old AXON pCLAMP binary file formats.
//
// An ANSI C compiler should be used for compilation.
// Compile with the large memory model option.
// (e.g. CL -c -AL ABFHEADR.C)

#include "../Common/wincpp.hpp"

#include "abffiles.h"               // header definition & constants
#include "oldheadr.h"               // prototypes for this file.
#include "abfoldnx.h"               // Indices for old pCLAMP files (< V6.0)
#include "msbincvt.h"               // Conversion routines MSBIN <==> IEEE float
#include "abfutil.h"                // Utility functions.
#include <math.h>
#include <float.h>
 

#define ABF_OLDPARAMSIZE      260   // size of old acquisition parameter array
//#define (sz)   OemToCharBuff(sz, sz, sizeof(sz))

#pragma pack(1)                     // pack structure on byte boundaries
struct ABFTopOfFile
{
   long     lFileSignature;
   float    fFileVersionNumber;
   short    nOperationMode;
   long     lActualAcqLength;
   short    nNumPointsIgnored;
   long     lActualEpisodes;
   long     lFileStartDate;
   long     lFileStartTime;
   long     lStopwatchTime;
   float    fHeaderVersionNumber;
   short    nFileType;
   short    nMSBinFormat;
};

typedef union
{
   struct ABFTopOfFile  ABF;
   float Param[10];
} TopOfFile;
#pragma pack()                      // return to default packing

//
// #defines for diagnosing ATF files.
//
/*
#ifdef BIGENDIAN
#error Big endian computers are no longer supported by ABF.
#endif
*/

#define ATF_MASK      0x00FFFFFF
#define ATF_SIGNATURE 0x00465441 // "FTA"

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
// FUNCTION: IsValidFloat
// PURPOSE:  Check if the number is a valid float, to protect against math exceptions.
// RETURNS:  TRUE = number is a valid float.
//
static BOOL IsValidFloat( double dNumber, float fMaxMantissa, int nMaxExponent )
{
   int nExponent = 0;
   double dMantissa = 0.0;
   
   // Get the mantissa and exponent.
   dMantissa = frexp( dNumber, &nExponent );
   
   // Check they are OK.
   if( dMantissa > fMaxMantissa || nExponent > nMaxExponent )
	   return FALSE;
   
   return TRUE;
}

//===============================================================================================
// FUNCTION: GetFileVersion
// PURPOSE:  Find out the version number and data format of a data file, given a DOS file handle.
// RETURNS:  TRUE = File is an ABF file, file version retrieved OK.
//
BOOL OLDH_GetFileVersion( FILEHANDLE hFile, UINT *puFileType, float *pfFileVersion,
                          BOOL *pbMSBinFormat)
{
   TopOfFile TOF;

   *puFileType = 0L;
   *pfFileVersion = 0.0F;
   *pbMSBinFormat = FALSE;

   // Seek to the start of the file.
   c_SetFilePointer(hFile, 0L, NULL, FILE_BEGIN);

   // Read top of file, to determine the file version
   if (!ABFU_ReadFile(hFile, &TOF, sizeof(TOF)))
      return FALSE;

   // If the file is byte swapped, return as invalid file.
   // Big-endian computers are no longer supported.
   if (TOF.ABF.lFileSignature == ABF_REVERSESIGNATURE)
      return FALSE;

   // Check if it is an ABF file.
   if (TOF.ABF.lFileSignature == ABF_NATIVESIGNATURE)
   {
      *puFileType = ABF_ABFFILE;
      *pfFileVersion = TOF.ABF.fFileVersionNumber;
      return TRUE;
   }

   // check for AXON ATF text file format.
   if ((TOF.ABF.lFileSignature & ATF_MASK) == ATF_SIGNATURE)
      return FALSE;

   // Now we must determine if the file is an old pCLAMP file (< V6.0).
   // Check whether the file is in old MS binary format. This was a floating point format
   // with a different exponent and mantissa length than IEEE.
   
   // Make sure the experiment type and file version are valid floating point numbers.
   if( !IsValidFloat( TOF.Param[F53_EXPERIMENTTYPE], 10.0F, 4 ) ||
       !IsValidFloat( TOF.Param[F53_FILEVERSIONNUMBER], 10.0F, 4 ) ||
       !IsValidFloat( TOF.Param[F53_ADCNUMCHANNELS], 10.0F, 4 ) ||
       !IsValidFloat( TOF.Param[F53_SAMPLESPEREPISODE], 10.0F, 50 ) )
	   return FALSE;
 
   if( !IsValidFloat( TOF.Param[F53_FILEVERSIONNUMBER], 10.0F, 4 ) )
      return FALSE;
   
   if ((TOF.Param[F53_EXPERIMENTTYPE] < 0.0F) || (TOF.Param[F53_FILEVERSIONNUMBER] < 0.0F))
   {
      for (int i=0; i < 10; i++)
         fMSBintoIeee(&(TOF.Param[i]), &(TOF.Param[i]));
      *pbMSBinFormat = TRUE;
   }

   // Set type for very old data files
   if (TOF.Param[F53_EXPERIMENTTYPE] == 0.0F)
      TOF.Param[F53_EXPERIMENTTYPE] = 10.0F;
   
   // Set return file type.
   if (TOF.Param[F53_EXPERIMENTTYPE] == 1.0F)
      *puFileType = ABF_CLAMPEX;
   else if (TOF.Param[F53_EXPERIMENTTYPE] == 10.0F)
      *puFileType = ABF_FETCHEX;
   else 
      return FALSE;

   // Check for minimal sanity in critical parameters.
   if ((TOF.Param[F53_ADCNUMCHANNELS] < 1.0F) ||
       (TOF.Param[F53_ADCNUMCHANNELS] > 8.0F) ||
       (TOF.Param[F53_SAMPLESPEREPISODE] < 0.0F) ||
       (TOF.Param[F53_FILEVERSIONNUMBER] < 0.0F) ||
       (TOF.Param[F53_FILEVERSIONNUMBER] > 10.0F) )
      return FALSE;

   // Return the file version.
   *pfFileVersion = TOF.Param[F53_FILEVERSIONNUMBER];
   return TRUE;
}

//==============================================================================================
// FUNCTION: CorrectDACFilePath
// PURPOSE:  Combines the old DACFileName with the old DACFilePath into the new DACFilePath.
//
static void CorrectDACFilePath(ABFFileHeader *pFH)
{
   // Get the old DACFileName from the header.
   char szOldName[ABF_OLDDACFILENAMELEN+1];
   ABFU_GetABFString(szOldName, sizeof(szOldName), 
                     pFH->_sDACFilePath, ABF_OLDDACFILENAMELEN);

   // Split it into filename and extension components.
   char szName[_MAX_FNAME];
   char szExt[_MAX_EXT];
   _splitpath( szOldName, NULL, NULL, szName, szExt );

   // pCLAMP 6 substituted DAT for a non-existent extension.
   if (szExt[0] == '\0')
      strcpy(szExt, ".DAT");

   // Get the old DACFilePath from the header.
   char szOldPath[ABF_OLDDACFILEPATHLEN+1];
   ABFU_GetABFString(szOldPath, sizeof(szOldPath), 
                     pFH->_sDACFilePath+ABF_OLDDACFILENAMELEN, ABF_OLDDACFILEPATHLEN);

   // Build the complete path and set it in the header again.
   char szPath[_MAX_PATH];
   _makepath( szPath, NULL, szOldPath, szName, szExt );
   ABF_SET_STRING(pFH->_sDACFilePath, szPath);
}


//===============================================================================================
// FUNCTION: CorrectDACScaling
// PURPOSE:  Correct the DAC scaling factors in the header.
// NOTES:    Previously the DAC file scale factor scaled 2-byte ADC values
//           from the DAC file directly into DAC values for output.
//           Now the scale and offset convert the UU values in the input file 
//           to UU output values, so old scale & offset values must be adjusted.
//
static void CorrectDACScaling(ABFFileHeader *pFH)
{
   // Now:
   // DACuu = ADCuu x S + O
   //
   // Previously:
   // DAC = ADC x S + O
   //
   // DACuu = (ADC x S + O) x Sdac + Odac
   // DACuu = ((ADCuu - Oadc)/Sadc x S + O) x Sdac + Odac
   // DACuu = ADCuu x (S/Sadc x Sdac)
   //       + Odac + O x Sdac - Oadc x (S/Sadc x Sdac) 
   //
   // => fNewScale  = fOldScale * Sdac / Sadc
   // => fNewOffset = fOldOffset * Sdac + Odac
   //               - Oadc x fNewScale

   // Initialize a header structure to use on the DAC file.
   ABFFileHeader DH;
   ABFH_Initialize(&DH);

   // Try and open the file as an ABF binary file, if this fails, 
   // just return, the user will have to sort things out for themselves.
   {
      char szFilename[_MAX_PATH];
      ABF_GET_STRING(szFilename, pFH->_sDACFilePath, sizeof(szFilename));

      UINT uMaxSamples = 0;
      DWORD dwMaxEpi = 0;
      int hFile;
      if (!ABF_ReadOpen(szFilename, &hFile, ABF_DATAFILE, 
                        &DH, &uMaxSamples, &dwMaxEpi, NULL))
         return;
      ABF_Close(hFile, NULL);
   }

   // Check that the channel was acquired.
   UINT uChannelOffset = 0;
   if (!ABFH_GetChannelOffset( &DH, pFH->_nDACFileADCNum, &uChannelOffset ))
      return;

   // Save the old sacle and offset.
   float fOldScale  = pFH->_fDACFileScale;
   float fOldOffset = pFH->_fDACFileOffset;

   // Get the ADC-to-UU factors for the ADC channel in the DAC file.
   float fSadc, fOadc;
   ABFH_GetADCtoUUFactors( &DH, pFH->_nDACFileADCNum, &fSadc, &fOadc);

   // Get the DAC-to-UU factors for target DAC channel.
   float fSdac, fOdac;
   ABFH_GetDACtoUUFactors( pFH, pFH->nActiveDACChannel, &fSdac, &fOdac );

   // Update the scaling factors.
   float fNewScale  = fOldScale * fSdac / fSadc;
   float fNewOffset = fOldOffset * fSdac + fOdac - fOadc * fNewScale;

   pFH->fDACFileScale[pFH->nActiveDACChannel]  = fNewScale;
   pFH->fDACFileOffset[pFH->nActiveDACChannel] = fNewOffset;
}

//===============================================================================================
// FUNCTION: OLDH_ABFtoABF15
// PURPOSE:  Brings an ABF file header up to ABF version 1.5.
//   
void OLDH_ABFtoABF15(ABFFileHeader *pFH)
{
   for (int i=0; i<ABF_BELLCOUNT; i++)   
   {
      pFH->nBellEnable[i] = 0;
      pFH->nBellLocation[i] = 1;
      pFH->nBellRepetitions[i] = 1;
   }

   ASSERT(pFH->lFileSignature==ABF_NATIVESIGNATURE);

   //  Convert DOS strings to ANSI char set.
   static char s_szAxEngine[] = "AXENGN";
   if ((pFH->fFileVersionNumber < 1.4) && 
       (_strnicmp(pFH->sCreatorInfo, s_szAxEngine, strlen(s_szAxEngine))!=0))
   {
/*      ABF_OEM_TO_ANSI(pFH->_sFileComment);
      for (int i=0; i<ABF_ADCCOUNT; i++)
      {
         ABF_OEM_TO_ANSI(pFH->sADCChannelName[i]);
         ABF_OEM_TO_ANSI(pFH->sADCUnits[i]);
      }
      for (int i=0; i<ABF_DACCOUNT; i++)
      {
         ABF_OEM_TO_ANSI(pFH->sDACChannelName[i]);
         ABF_OEM_TO_ANSI(pFH->sDACChannelUnits[i]);
      }
      ABF_OEM_TO_ANSI(pFH->sArithmeticUnits);
*/   }
   
   if (pFH->fFileVersionNumber < 1.4)
   {
      // Fix trigger source parameter for FETCHEX files pre-ABF1.3.
      if ((pFH->nOperationMode==ABF_GAPFREEFILE) && (pFH->nTriggerSource > 0))
         pFH->nTriggerSource = 0;

      // Synthesize statistics period from old lCalculationPeriod.         
      pFH->fStatisticsPeriod = pFH->lCalculationPeriod * pFH->fADCSampleInterval / 1E6F;

      // Fill in the new filter type fields.
      for (int i=0; i<ABF_ADCCOUNT; i++)
      {
         if (pFH->fSignalLowpassFilter[i] != ABF_FILTERDISABLED)
            pFH->nLowpassFilterType[i] = ABF_FILTER_EXTERNAL;
         if (pFH->fSignalHighpassFilter[i] != 0.0F)
            pFH->nHighpassFilterType[i] = ABF_FILTER_EXTERNAL;
      }

      // Synthesize the new trial trigger field (new for 1.4).
      if ((pFH->nOperationMode==ABF_WAVEFORMFILE) && 
          (pFH->nTriggerAction==ABF_TRIGGER_STARTTRIAL))
      {
         switch (pFH->nTriggerSource)
         {
            case ABF_TRIGGERSPACEBAR:
               pFH->nTrialTriggerSource = ABF_TRIALTRIGGER_SPACEBAR;
               break;
            case ABF_TRIGGEREXTERNAL:
               pFH->nTrialTriggerSource = ABF_TRIALTRIGGER_EXTERNAL;
               break;
            default:
               pFH->nTrialTriggerSource = ABF_TRIALTRIGGER_NONE;
               break;
         }
         pFH->nTriggerAction = ABF_TRIGGER_STARTEPISODE;
         pFH->nTriggerSource = 0;
      }

      // Correct the DAC file scaling parameters.
      if ((pFH->nOperationMode==ABF_WAVEFORMFILE) && 

          (pFH->_nWaveformSource == ABF_DACFILEWAVEFORM))
      {
         CorrectDACFilePath(pFH);
         CorrectDACScaling(pFH);
      }
      UINT nStatsRegionID = 0; // Temporary statistics region ID.
      pFH->lStatisticsMeasurements = ABF_STATISTICS_ABOVETHRESHOLD | ABF_STATISTICS_MEANOPENTIME;
      pFH->lStatsMeasurements[ nStatsRegionID ] = ABF_PEAK_MEASURE_PEAK | ABF_PEAK_MEASURE_PEAKTIME;
   }

   if (pFH->fFileVersionNumber < 1.5F)
   {
      // Changes for V1.5:
      // * Change ABFSignal parameters from UUTop & UUBottom to fDisplayGain & fDisplayOffset.
      // * Addition of the external tag type parameter.
      pFH->nExternalTagType = ABF_EXTERNALTAG;
   }

}

//===============================================================================================
// FUNCTION: OLDH_ABFtoCurrentVersion
// PURPOSE:  Brings an ABF file header up to the current ABF version.
//   
void OLDH_ABFtoCurrentVersion(ABFFileHeader *pFH)
{
   if( (pFH->fFileVersionNumber < ABF_V16) || (pFH->nFileType != ABF_ABFFILE) )
   {
      // Changes for V1.6:
      // 1. Expanded header to 6k bytes
      // 2. New entries to accommodate 2 waveform channels.
      // 3. Date changed to 4 digit year
      // 4. New entries to accommodate 2 presweep trains, user lists and P/N leak subtractions.
      // 5. Longer file comment.
      // 6. Extra 'enable' fields.
      // 7. Protocol name in header.
	  
      // Copy the waveform info to the new header parameters.
      UINT uDAC = pFH->nActiveDACChannel;

      pFH->lDACFilePtr[uDAC]           = pFH->_lDACFilePtr;
      pFH->lDACFilePtr[uDAC]           = 0;
      pFH->lDACFileNumEpisodes[uDAC]   = pFH->_lDACFileNumEpisodes;
      pFH->lDACFileNumEpisodes[1-uDAC] = 0;
      pFH->nWaveformEnable[uDAC]       = (pFH->_nWaveformSource != ABF_WAVEFORMDISABLED);
      pFH->nWaveformEnable[1-uDAC]     = 0;
      pFH->nWaveformSource[uDAC]       = pFH->_nWaveformSource;
      pFH->nWaveformSource[1-uDAC]     = ABF_EPOCHTABLEWAVEFORM;
      pFH->nInterEpisodeLevel[uDAC]    = pFH->_nInterEpisodeLevel;
      pFH->nInterEpisodeLevel[1-uDAC]  = 0;
      
      for( UINT i=0; i<ABF_EPOCHCOUNT; i++ )
      {
         pFH->nEpochType[uDAC][i]           = pFH->_nEpochType[i];
         pFH->nEpochType[1-uDAC][i]         = 0;
         pFH->fEpochInitLevel[uDAC][i]      = pFH->_fEpochInitLevel[i];
         pFH->fEpochInitLevel[1-uDAC][i]    = 0.0F;
         pFH->fEpochLevelInc[uDAC][i]       = pFH->_fEpochLevelInc[i];
         pFH->fEpochLevelInc[1-uDAC][i]     = 0.0F;
         pFH->lEpochInitDuration[uDAC][i]   = pFH->_nEpochInitDuration[i];
         pFH->lEpochInitDuration[1-uDAC][i] = 0;
         pFH->lEpochDurationInc[uDAC][i]    = pFH->_nEpochDurationInc[i];
         pFH->lEpochDurationInc[1-uDAC][i]  = 0;
      }

      pFH->fDACFileScale[uDAC]        = pFH->_fDACFileScale;
      pFH->fDACFileScale[1-uDAC]      = 0.0F;
      pFH->fDACFileOffset[uDAC]       = pFH->_fDACFileOffset;
      pFH->fDACFileOffset[1-uDAC]     = 0.0F;
      pFH->lDACFileEpisodeNum[uDAC]   = pFH->_nDACFileEpisodeNum;
      pFH->lDACFileEpisodeNum[1-uDAC] = 0;
      pFH->nDACFileADCNum[uDAC]       = pFH->_nDACFileADCNum;
      pFH->nDACFileADCNum[1-uDAC]     = 0;
      strncpy( pFH->sDACFilePath[uDAC], pFH->_sDACFilePath, ABF_DACFILEPATHLEN );
      strncpy( pFH->sDACFilePath[1-uDAC], "", ABF_DACFILEPATHLEN );

	   // Telegraph settings
      pFH->nTelegraphEnable[pFH->_nAutosampleADCNum]      = pFH->_nAutosampleEnable;
      pFH->nTelegraphInstrument[pFH->_nAutosampleADCNum]  = pFH->_nAutosampleInstrument;
      pFH->fTelegraphAdditGain[pFH->_nAutosampleADCNum]   = pFH->_fAutosampleAdditGain;
      pFH->fTelegraphFilter[pFH->_nAutosampleADCNum]      = pFH->_fAutosampleFilter;
      pFH->fTelegraphMembraneCap[pFH->_nAutosampleADCNum] = pFH->_fAutosampleMembraneCap;

      // Presweep trains.
      if( uDAC == (UINT)pFH->_nConditChannel )
      {
         pFH->nConditEnable[uDAC]        = pFH->_nConditEnable;
         pFH->lConditNumPulses[uDAC]     = pFH->_lConditNumPulses;
         pFH->fBaselineDuration[uDAC]    = pFH->_fBaselineDuration;
         pFH->fBaselineLevel[uDAC]       = pFH->_fBaselineLevel;
         pFH->fStepDuration[uDAC]        = pFH->_fStepDuration;
         pFH->fStepLevel[uDAC]           = pFH->_fStepLevel;
         pFH->fPostTrainLevel[uDAC]      = pFH->_fPostTrainLevel;
      }

      if( uDAC == (UINT)pFH->nActiveDACChannel )
      {
         // P/N Leak subtraction parameters.
         pFH->nPNEnable[uDAC]            = pFH->_nPNEnable;
         pFH->nPNPolarity[uDAC]          = pFH->_nPNPolarity;
         pFH->fPNHoldingLevel[uDAC]      = pFH->_fPNHoldingLevel;
         pFH->nPNADCSamplingSeq[uDAC][0] = LOBYTE( pFH->_nPNADCNum );

         // Sanity check the channel number.
         ASSERT( pFH->nPNADCSamplingSeq[uDAC][0] == pFH->_nPNADCNum );

         // User list parameters.
         pFH->nULEnable[uDAC]            = pFH->_nListEnable;
         pFH->nULParamToVary[uDAC]       = pFH->_nParamToVary;
         strncpy( pFH->sULParamValueList[uDAC], pFH->_sParamValueList, ABF_VARPARAMLISTLEN );
      }

      // DAC Calibration Factors.
      for(int i=0; i<ABF_DACCOUNT; i++ )
      {
         pFH->fDACCalibrationFactor[i] = 1.0F;
         pFH->fDACCalibrationOffset[i] = 0.0F;
      }

      // File Comment.
      strncpy( pFH->sFileComment, pFH->_sFileComment, ABF_OLDFILECOMMENTLEN );

      // Fix the date if needed.
      pFH->lFileStartDate = ABFU_FixFileStartDate ( pFH->lFileStartDate );

      // Extra 'enable' fields.
      pFH->nCommentsEnable = (pFH->nManualInfoStrategy != ABF_ENV_DONOTWRITE);

      // Turn on the 'Analyse Me' flag.
      pFH->nAutoAnalyseEnable  = ABF_AUTOANALYSE_DEFAULT;
   }   
   
   if( (pFH->fFileVersionNumber < ABF_V170) || (pFH->nFileType != ABF_ABFFILE) )
   {
      // Set stats variables
      for ( UINT uRegion = 0; uRegion < ABF_STATS_REGIONS; uRegion++ )
      {
         pFH->lStatsMeasurements[uRegion]     = pFH->_lAutopeakMeasurements; // Copy stats measurements across all regions.
         pFH->nRiseBottomPercentile[uRegion]  = 10;
         pFH->nRiseTopPercentile[uRegion]     = 90;
         pFH->nDecayBottomPercentile[uRegion] = 10;
         pFH->nDecayTopPercentile[uRegion]    = 90;   
      }

      pFH->nStatsEnable        = pFH->_nAutopeakEnable;
      pFH->nStatsSmoothing     = pFH->_nAutopeakSmoothing;
      pFH->nStatsBaseline      = pFH->_nAutopeakBaseline;
      pFH->lStatsBaselineStart = pFH->_lAutopeakBaselineStart;
      pFH->lStatsBaselineEnd   = pFH->_lAutopeakBaselineEnd;

      // Polarity is channel specific
      for ( UINT uChannel = 0; uChannel < ABF_ADCCOUNT; uChannel++ )
         pFH->nStatsChannelPolarity[uChannel] = pFH->_nAutopeakPolarity;;

      // Convert the old channel selection to an active channel.
      UINT nStatsADCNum = pFH->_nAutopeakADCNum;
      
      // Shift the uBitmask by nStatsADCNum of bits.
      UINT uBitMask = 0x01 << nStatsADCNum;
      pFH->nStatsActiveChannels = (short)uBitMask;

      // Convert the old search region into the statistics regions.
      for ( UINT uRegion = 0; uRegion < ABF_STATS_REGIONS; uRegion++ )
      {
         pFH->nStatsSearchMode[ uRegion ] = pFH->_nAutopeakSearchMode;
         pFH->lStatsStart[ uRegion ]      = pFH->_lAutopeakStart;
         pFH->lStatsEnd[ uRegion ]        = pFH->_lAutopeakEnd;
      }

      // Select statistics region zero.
      pFH->nStatsSearchRegionFlags           = ABF_PEAK_SEARCH_REGION0;
      pFH->nStatsSelectedRegion              = 0;

	   // Telegraph settings
      pFH->nTelegraphEnable[pFH->_nAutosampleADCNum]      = pFH->_nAutosampleEnable;
      pFH->nTelegraphInstrument[pFH->_nAutosampleADCNum]  = pFH->_nAutosampleInstrument;
      pFH->fTelegraphAdditGain[pFH->_nAutosampleADCNum]   = pFH->_fAutosampleAdditGain;
      pFH->fTelegraphFilter[pFH->_nAutosampleADCNum]      = pFH->_fAutosampleFilter;
      pFH->fTelegraphMembraneCap[pFH->_nAutosampleADCNum] = pFH->_fAutosampleMembraneCap;
   }

   if( (pFH->fFileVersionNumber < ABF_V171) || (pFH->nFileType != ABF_ABFFILE) )
   {
      for( int i = 0; i < ABF_WAVEFORMCOUNT; i++ )
      {
         sprintf( pFH->sEpochResistanceSignalName[ i ], "IN #%d", i);
         pFH->nEpochResistanceState[ i ] = 0; 
      }
   }

   if( (pFH->fFileVersionNumber < ABF_V172) || (pFH->nFileType != ABF_ABFFILE) )
   {
      pFH->nAlternateDACOutputState = 0;
      for( int nEpoch = 0; nEpoch  < ABF_EPOCHCOUNT; nEpoch ++ )
      {
         pFH->nAlternateDigitalValue[ nEpoch ] = 0;
         pFH->nAlternateDigitalTrainValue[ nEpoch ] = 0;
      }
   }

   if( (pFH->fFileVersionNumber < ABF_V173) || (pFH->nFileType != ABF_ABFFILE) )
   {
      //Post-processing values.
      for( int i=0; i<ABF_ADCCOUNT; i++)
      {
         pFH->fPostProcessLowpassFilter[i] = ABF_FILTERDISABLED;
         pFH->nPostProcessLowpassFilterType[i] = ABF_POSTPROCESS_FILTER_NONE;
      }
   }

   if( (pFH->fFileVersionNumber < ABF_V174) || (pFH->nFileType != ABF_ABFFILE) )
      pFH->channel_count_acquired = 0; 

   if( (pFH->fFileVersionNumber < ABF_V175) || (pFH->nFileType != ABF_ABFFILE) )
   {
      for( int i=0; i<ABF_ADCCOUNT; i++)
         pFH->nStatsChannelPolarity[ i ] = ABF_PEAK_ABSOLUTE;
   }
   
   if( (pFH->fFileVersionNumber < ABF_V176) || (pFH->nFileType != ABF_ABFFILE) )
      pFH->nDD132xTriggerOut = 0;

   if( (pFH->fFileVersionNumber < ABF_V177) || (pFH->nFileType != ABF_ABFFILE) )
   {
      pFH->nCreatorMajorVersion  = 0;
      pFH->nCreatorMinorVersion  = 0;
      pFH->nCreatorBugfixVersion = 0;
      pFH->nCreatorBuildVersion  = 0; 
   }

   if( (pFH->fFileVersionNumber < ABF_V178) || (pFH->nFileType != ABF_ABFFILE) )
      pFH->nAlternateDigitalOutputState = 0;

   if( (pFH->fFileVersionNumber < ABF_V180 && pFH->fFileVersionNumber > ABF_V174) || (pFH->nFileType != ABF_ABFFILE) )
   {
      for( UINT uRegion = 0; uRegion < ABF_STATS_REGIONS; uRegion++ )
         pFH->nStatsSearchMode[ uRegion ] = pFH->_nStatsSearchMode;
   }

   // When updating the header version copy this else if statement but use your parameters and the current version number
   if( (pFH->fFileVersionNumber < ABF_V180) || (pFH->nFileType != ABF_ABFFILE) )
   {
      // Clear the annotations since they will not be copied.
      pFH->lAnnotationSectionPtr = 0;
      pFH->lNumAnnotations       = 0;
   }
}
/*
//===============================================================================================
// FUNCTION: OLDH_CorrectScopeConfig
// PURPOSE:  Corrects the display gain and offset in the scope parameters.
//   
void OLDH_CorrectScopeConfig(ABFFileHeader *pFH, ABFScopeConfig *pCfg)
{
   ABFSignal *pT = pCfg->TraceList;
   for (int i=0; i<pCfg->nTraceCount; i++, pT++)
   {
      if (pT->bFloatData)
      {
         float fUUTop    = pT->fDisplayGain;
         float fUUBottom = pT->fDisplayOffset;

         float fDisplayRange = fUUTop - fUUBottom;
         float fDisplayOffset = (fUUTop + fUUBottom)/2;
      
         // FIX FIX FIX FIX FIX
         float fInputRange  = pFH->fArithmeticUpperLimit - pFH->fArithmeticLowerLimit;
         float fInputOffset = (pFH->fArithmeticUpperLimit + pFH->fArithmeticLowerLimit)/2;

         pT->fDisplayGain   = fInputRange / fDisplayRange;
         pT->fDisplayOffset = fDisplayOffset - fInputOffset;
      }
      else
      {
         ABFH_DisplayRangeToGainOffset( pFH, pFH->nADCSamplingSeq[pT->nMxOffset], 
                                        pT->fDisplayGain, pT->fDisplayOffset,
                                        &pT->fDisplayGain, &pT->fDisplayOffset);
      }
   }
}

*/
//***********************************************************************************************
// The following three utility functions work for both old FETCHEX and old
// CLAMPEX files, the array indices being the same for each case.

//===============================================================================================
// FUNCTION: GetOldDACUnits
// PURPOSE:  Extract the units-of-measure string for DAC channel 0 from the second label.
//   
static void GetOldDACUnits(char *Label, char *DACLabel)
{
   strncpy(DACLabel, Label+ABF_OLDUNITLEN, ABF_DACUNITLEN);
}

//===============================================================================================
// FUNCTION: GetOldADCUnits
// PURPOSE:  Extract the units-of-measure strings for ADC channels 0 and 1 from the
//           fourth and fifth labels, respectively.
//   
static void GetOldADCUnits(char *Label, char *ADCLabel)
{
   strncpy(ADCLabel, Label+3*ABF_OLDUNITLEN, ABF_ADCUNITLEN);
   strncpy(ADCLabel+(ABF_ADCCOUNT-2)*ABF_ADCUNITLEN, Label+3*ABF_OLDUNITLEN, ABF_ADCUNITLEN);

   strncpy(ADCLabel+8, Label+4*ABF_OLDUNITLEN, ABF_ADCUNITLEN);
   strncpy(ADCLabel+(ABF_ADCCOUNT-1)*ABF_ADCUNITLEN, Label+4*ABF_OLDUNITLEN, ABF_ADCUNITLEN);
}

//===============================================================================================
// FUNCTION: ReadADCInfo
// PURPOSE:  Read ADC channel scaling factors etc.
//   
static BOOL ReadADCInfo(FILEHANDLE hFile, float *Param, char *ADCLabel )
{
   // Seek to the start of the ADC channel information.
	c_SetFilePointer(hFile, 640L, NULL, FILE_BEGIN);

   // Read the ADC channel information and display parameters (97-160).

   // 4 arrays of ABF_ADCCOUNT x 4 byte floats
   if (!ABFU_ReadFile(hFile, Param+F53_INSTOFFSET, 4 * ABF_ADCCOUNT * sizeof(float)))
      return FALSE;

   // Read the ADC channel units strings into a single string
   return ABFU_ReadFile(hFile, ADCLabel, ABF_ADCCOUNT * ABF_ADCUNITLEN);
}

//===============================================================================================
// Code for translation of old FETCHEX parameter and data file headers.
// It will handle two different file formats:
//
// 1. Data files (program versions before 5.1):
//
//   320  80 single-precision (4-byte) values in Microsoft BASIC binary format
//    77  77-byte ASCII character string for file comment
//    80  5 16-byte ASCII character strings for units-of-measure:
//          time-scale "us"
//          DAC channel #0
//          (unused)
//          ADC channel #0
//          (unused)
//    35  35 bytes of environment data
//   512  bytes of unused header
//  ----  ---------------------------
//  1024  bytes total for file header
//
// 2. Parameter and data files (program versions 5.1 and later):
//
//   320  80 single-precision (4-byte) values in IEEE floating-point format
//    77  77-byte ASCII character string for file comment
//    80  5 16-byte ASCII character strings (80 bytes) for units-of-measure:
//          time-scale "us"
//          DAC channel #0
//          DAC channel #1
//          (unused)
//          (unused)
//     3  3 bytes of unused space
//   160  16 10-byte strings (160 bytes) of channel names
//   256  64 single-precision (4-byte) values in IEEE floating-point format
//   128  16 8-byte ASCII character strings (128 bytes) for units-of-measure:
//          ADC channel #0
//               .
//               .
//               .
//          ADC channel #15
//  ----  ---------------------------
//  1024  bytes total for file header
//
// See also the extensive documentation for the format of the Version 5.1
// parameter array.
//
// NOTE: the parameter version number (Param[9]) is NOT updated here.
//
//-----------------------------------------------------------------------------

//===============================================================================================
// FUNCTION: ReadADCNames
// PURPOSE:  Read the channel names array.
//   
static BOOL ReadADCNames(FILEHANDLE hFile, char *ChannelName)
{
   // Seek to the channel names area.
   c_SetFilePointer(hFile, 480L, NULL, FILE_BEGIN);

   // Read the ADC channel name strings into a single string.
   return ABFU_ReadFile(hFile, ChannelName, ABF_ADCCOUNT * ABF_ADCNAMELEN);
}

//===============================================================================================
// FUNCTION: FetchexToV5_3
// PURPOSE:  Convert fetchex parameters prior to file version 5.3 to the V5.3 format.
//   
static BOOL FetchexToV5_3( float *Param )
{
   // Correct the acquisition mode.
   if (Param[F53_OPERATIONMODE] == 1.0F)
      Param[F53_OPERATIONMODE] = 2.0F;

   // Convert old parameter 22 to event-triggered mode.
   if (Param[22] == -1.0F)
   {
      Param[22] = 0.0F;
      Param[F53_OPERATIONMODE] = 0.0F;
   }

   // Set the default DAC channel 0 gain.
   if (Param[F53_GAINDACTOCELL] == 0.0F)
      Param[F53_GAINDACTOCELL] = 1.0F;

   // Convert the DAC channel 0 gain from milli-volts to volts.
   Param[F53_GAINDACTOCELL] = Param[F53_GAINDACTOCELL] * 1000.0F;

   // Initialize the amplification factors
   for (int i=0; i<ABF_ADCCOUNT; i++)
   {
      Param[F53_INSTSCALEFACTOR+i] = 1.0F;
      Param[F53_ADCDISPLAYGAIN+i]  = 1.0F;
   }

   // Move the ADC gain parameters to their correct positions in the parameter
   // array, converting from milli-volts to volts along the way.
   if (Param[F53_CHANNEL0GAIN] != 0.0F)
   {
      Param[F53_INSTSCALEFACTOR] = Param[F53_CHANNEL0GAIN] / 1000.0F;
      Param[F53_INSTSCALEFACTOR+ABF_ADCCOUNT-1] = Param[F53_INSTSCALEFACTOR];
   }

   Param[F53_ADCFIRSTLOGICALCHANNEL] = 0.0F;

   // Handle the case of a parameter file (pre-version 5.1).

   // Set the default ADC channel amplification factor.
   // Move the ADC display parameters to their correct positions.
   if (Param[F53_AMPLIFICATIONFACTOR] != 0.0F)
   {
      Param[F53_ADCDISPLAYGAIN] = Param[F53_AMPLIFICATIONFACTOR];
      Param[F53_ADCDISPLAYGAIN+ABF_ADCCOUNT-1] = Param[F53_ADCDISPLAYGAIN];
   }
   Param[F53_AMPLIFICATIONFACTOR] = 0.0F;

   // ADC channel 0 offset => 0 & 15
   Param[F53_ADCDISPLAYOFFSET] = Param[F53_VERTICALOFFSET];
   Param[F53_ADCDISPLAYOFFSET+ABF_ADCCOUNT-1] = Param[F53_ADCDISPLAYOFFSET];

   // Check the number of 512-sample segments per episode (old versions
   // allowed values of 8, 16, 24, ..., 2048), and adjust if necessary.

   int nSamplesPerEpisode = short(Param[F53_OLDSAMPLESPEREPISODE]);

   // if FETCHEX can't read the data, return an error.
   if ((nSamplesPerEpisode == 0) || (nSamplesPerEpisode % 512 != 0))
      return FALSE;

   Param[F53_SEGMENTSPEREPISODE] = (float)(nSamplesPerEpisode / 512);

   // Check that ADC range and resolution were set properly

   if (Param[F53_ADCRANGE] == 0.0F)
      Param[F53_ADCRANGE] = 10.0F;
   if (Param[F53_DACRANGE] == 0.0F)
      Param[F53_DACRANGE] = 10.0F;
   if (Param[F53_ADCRESOLUTION] == 0.0F)
      Param[F53_ADCRESOLUTION] = 12.0F;
   if (Param[F53_DACRESOLUTION] == 0.0F)
      Param[F53_DACRESOLUTION] = 12.0F;

   Param[F53_INVALIDLASTDATA] = 0.0F;
   return TRUE;
}

//===============================================================================================
// FUNCTION: DataResolution
// PURPOSE:  Return the data resolution given the number of bits used.
//   
static long DataResolution( float fNumberOfBits )
{
   switch (short(fNumberOfBits))
   {
      case 16: return(32768L);
      case 14: return(8192L);
      default: return(2048L);
   }
}

//===============================================================================================
// FUNCTION: FetchexToABF1_x
// PURPOSE:  Converts FETCHEX parameters to ABF format.
//   
static void FetchexToABF1_x( float *Param, char *ADCLabel, char *DACLabel,
                             char *ChannelName, char *Comment, ABFFileHeader *pFH )
{
   short i, n;
   long OldSynchArraySize;

   // Initialize structure to all NULL's
   ABFH_Initialize(pFH);

   pFH->lFileSignature           = ABF_OLDPCLAMP;
   pFH->lActualEpisodes          = 1;    // Gap-free, EventLen files fixed later
   pFH->lActualAcqLength         = 
      (long)(Param[F53_SAMPLESPEREPISODE]) *
      (long)(Param[F53_ACTUALEPISODESPERFILE]) -
      (long)(Param[F53_NUMPOINTSIGNORED]) -
      (long)(Param[F53_INVALIDLASTDATA]);
   pFH->lFileStartTime           = (long)(Param[F53_FILESTARTTIME]);
   pFH->lStopwatchTime           =
      (long)(Param[F53_FILESTARTTIME] - Param[F53_FILEELAPSEDTIME]);
   pFH->fFileVersionNumber       = Param[F53_FILEVERSIONNUMBER];
   pFH->fHeaderVersionNumber     = ABF_CURRENTVERSION;
   pFH->lFileStartDate           = (long)(Param[F53_FILESTARTDATE]);

   pFH->nExperimentType          = ABF_VOLTAGECLAMP;  // Voltage Clamp
   pFH->fADCSecondSampleInterval = 0.0F;              // CLAMPEX only
   pFH->fADCSampleInterval       = Param[F53_ADCSAMPLEINTERVAL];
   pFH->lPreTriggerSamples       = 0;                 // default to 0
   pFH->nManualInfoStrategy      = ABF_ENV_WRITEEACHTRIAL; // default to write-each-trial

   if (Param[F53_OPERATIONMODE] == 0.0F)
   {
      pFH->nOperationMode = ABF_VARLENEVENTS;
      pFH->fSecondsPerRun = 0.0F;
      pFH->lPreTriggerSamples =
         (long)((1.0F-Param[F53_POSTTRIGGERPORTION]) * Param[F53_SAMPLESPEREPISODE]);
   }
   else
   {
      pFH->nOperationMode = ABF_GAPFREEFILE;
      pFH->fSecondsPerRun = Param[F53_SAMPLESPEREPISODE] / 1E6F *
                            Param[F53_EPISODESPERFILE] *
                            Param[F53_REQUESTEDSAMPLEINTERVAL];
   }

   pFH->nNumPointsIgnored        = short(Param[F53_NUMPOINTSIGNORED]);
   pFH->lTagSectionPtr           = long(Param[F53_TAGSECTIONPTR]);
   pFH->lNumTagEntries           = long(Param[F53_NUMTAGENTRIES]);
   if (Param[F53_SEGMENTSPEREPISODE] > 4) // Fixup for AXOTAPE bug.
      pFH->lNumSamplesPerEpisode = 512L;
   else
      pFH->lNumSamplesPerEpisode = (long)(Param[F53_SEGMENTSPEREPISODE]) * 512L;

   pFH->lClockChange             = 0;     // CLAMPEX only
   pFH->_lDACFilePtr             = 0;     // CLAMPEX only
   pFH->_lDACFileNumEpisodes     = 0;     // CLAMPEX only
   pFH->lStartDisplayNum         = 0;     // CLAMPEX only
   pFH->lFinishDisplayNum        = 0;     // CLAMPEX only
   pFH->nMultiColor              = 1;     // CLAMPEX only

   switch (short(Param[F53_AUTOSAMPLEINSTRUMENT]))
   {
      case 0:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEMANUAL;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 1:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 2:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1B;
         break;
      case 3:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEDISABLED;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 4:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH201;
         break;
      default:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEDISABLED;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
   }

   pFH->fCellID1                 = Param[F53_CELLID1];
   pFH->fCellID2                 = Param[F53_CELLID2];
   pFH->_fAutosampleMembraneCap   = 0.0F;
   pFH->fCellID3                 = Param[F53_THRESHOLDCURRENT];
   if (Param[F53_ADDITINSTGAIN] != 0.0F)
      pFH->_fAutosampleAdditGain = Param[F53_ADDITINSTGAIN];
   else 
      pFH->_fAutosampleAdditGain = 1.0F;
   pFH->_fAutosampleFilter        = Param[F53_INSTRUMENTFILTER];
   pFH->_nAutosampleADCNum        = short(Param[F53_ADCFIRSTLOGICALCHANNEL]);
   pFH->lNumberOfTrials          = 1;      // CLAMPEX only
   pFH->lRunsPerTrial            = 1;      // CLAMPEX only
   pFH->lEpisodesPerRun          = 1;      // CLAMPEX only
   pFH->nADCNumChannels          = short(Param[F53_ADCNUMCHANNELS]);
   pFH->nFirstEpisodeInRun       = 0;      // CLAMPEX only
   pFH->fTrialStartToStart       = 0.0F;   // CLAMPEX only
   pFH->fEpisodeStartToStart     = 0.0F;   // CLAMPEX only
   pFH->fScopeOutputInterval     = 0.0F;   // CLAMPEX only
   pFH->fADCRange                = Param[F53_ADCRANGE];
   pFH->fDACRange                = Param[F53_DACRANGE];
   pFH->lADCResolution           = DataResolution(Param[F53_ADCRESOLUTION]);
   pFH->lDACResolution           = DataResolution(Param[F53_DACRESOLUTION]);
   pFH->lDeltaArrayPtr           = 0;
   pFH->lNumDeltas               = 0;
   pFH->nDrawingStrategy         = 1;      // Display data in real time
   pFH->nTiledDisplay            = 1;      // CLAMPEX only
   pFH->nEraseStrategy           = 1;      // Erase after each trace
   pFH->nChannelStatsStrategy    = 1;      // Show channel stats in gap free
   pFH->lDisplayAverageUpdate    = 0;      // CLAMPEX only
   pFH->nDataDisplayMode         = short(Param[F53_DATADISPLAYMODE]);
   pFH->fTriggerThreshold        = 0.0F;
   pFH->nTriggerSource           = short(Param[F53_ADCFIRSTLOGICALCHANNEL]);
   pFH->nTriggerPolarity         = 0;
   pFH->nAveragingMode           = ABF_NOAVERAGING;      // AXOTAPE only
   pFH->fSynchTimeUnit           = 0.0F;   // use samples for all old files
   pFH->lSynchArraySize          = 0;      // These parameters will be reset later on
   pFH->lSynchArrayPtr           = 0;      //  for files that contain a Synch array
   pFH->lCalculationPeriod       = 16384;  // Calculate stats every 16k samples
   pFH->lSamplesPerTrace         = 16384;  // 16k samples per trace during acqn.
   pFH->nTriggerAction           = ABF_TRIGGER_STARTEPISODE;      // CLAMPEX only
   pFH->nUndoRunCount            = 0;      // CLAMPEX only
   pFH->lAverageCount            = 0;
   pFH->fStatisticsPeriod        = pFH->lCalculationPeriod * pFH->fADCSampleInterval / 1E3F;
   pFH->_nPNPolarity              = ABF_PN_SAME_POLARITY; // CLAMPEX only
   pFH->nStatsSmoothing = 1;

   strncpy(pFH->_sFileComment, Comment, ABF_OLDFILECOMMENTLEN);

   // Set up the channel mapping

   if (Param[F53_ADCNUMBERINGSTRATEGY] == 0.0F)
   {
      // Channel numbers are ascending
      // (AXOLAB-1, AXOLAB 1100, TL-1-125, TL-1-40, TL-3)
      // NOTE: AXOLAB 1100 only supports 8 channels (0-7)

      for (i=0; i<ABF_ADCCOUNT; i++)
         pFH->nADCPtoLChannelMap[i] = i;

      n = short(Param[F53_ADCFIRSTLOGICALCHANNEL]);
      for (i=0; i<pFH->nADCNumChannels; i++, n++)
         pFH->nADCSamplingSeq[i] = n;
   }
   else
   {
      // Channel numbers are decending
      // (TL-2 only at this stage)

      for (i=0; i<8; i++)
         pFH->nADCPtoLChannelMap[i] = short(7 - i);

      for (i=8; i<ABF_ADCCOUNT; i++)
         pFH->nADCPtoLChannelMap[i] = -1;

      n = short(8 - pFH->nADCNumChannels);
      for (i=0; i<pFH->nADCNumChannels; i++, n++)
         pFH->nADCSamplingSeq[i] = n;
   }

   // Set rest of sampling sequence to -1

   for (i=pFH->nADCNumChannels; i<ABF_ADCCOUNT; i++)
      pFH->nADCSamplingSeq[i] = -1;

   // Set up the ADC channel info

   for (i=0; i < ABF_ADCCOUNT; i++)
   {
      pFH->fInstrumentOffset[i]      = 0.0F;
      pFH->fInstrumentScaleFactor[i] = 1.0F;
      pFH->fADCDisplayAmplification[i] = 1.0F;
      pFH->fADCDisplayOffset[i]      = 0.0F;
      pFH->fSignalLowpassFilter[i]   = ABF_FILTERDISABLED;
      pFH->fSignalHighpassFilter[i]  = 0.0F;
      pFH->fADCProgrammableGain[i]   = 1.0F;
      pFH->fSignalGain[i]            = 1.0F;
      pFH->fSignalOffset[i]          = 0.0F;
   }

   for (i=0; i < ABF_ADCCOUNT; i++)
   {
      n = pFH->nADCPtoLChannelMap[i];
      if (n == -1)
         break;
      strncpy(pFH->sADCChannelName[n], ChannelName+ABF_ADCNAMELEN*i, ABF_ADCNAMELEN);
      strncpy(pFH->sADCUnits[n], ADCLabel+ABF_ADCUNITLEN*i, ABF_ADCUNITLEN);
      pFH->fInstrumentOffset[n] = Param[F53_INSTOFFSET + i];

      // protect against divide by zero errors from bad gains.

      if (Param[F53_INSTSCALEFACTOR + i] != 0.0F)
         pFH->fInstrumentScaleFactor[n] = Param[F53_INSTSCALEFACTOR + i];

      if (Param[F53_ADCDISPLAYGAIN + i] != 0.0F)
         pFH->fADCDisplayAmplification[n] = Param[F53_ADCDISPLAYGAIN + i];

      // convert offset from percentage of fullscreen to user units

      pFH->fADCDisplayOffset[n] = Param[F53_ADCDISPLAYOFFSET + i] *
                                  pFH->fADCRange /
                                  pFH->fInstrumentScaleFactor[i] /
                                  pFH->fADCDisplayAmplification[i];
      if ((pFH->_nAutosampleEnable != 0) &&
         (n == pFH->_nAutosampleADCNum))
         pFH->fADCDisplayOffset[n] /= pFH->_fAutosampleAdditGain;
   }

   strncpy(pFH->sDACChannelUnits[0], DACLabel, ABF_DACCOUNT * ABF_DACUNITLEN);
   for (i=0; i < ABF_DACCOUNT; i++)
   {
      pFH->fDACScaleFactor[i] = Param[F53_GAINDACTOCELL];
      pFH->fDACHoldingLevel[i] = Param[F53_DAC0HOLDINGLEVEL];
   }

   // Set up the data and Csynch array pointers

   if (pFH->nOperationMode != ABF_GAPFREEFILE)
   {
      pFH->lActualEpisodes = (long)(Param[F53_ACTUALEPISODESPERFILE]);
      pFH->lSynchArraySize = pFH->lActualEpisodes;

      if (Param[F53_FILEVERSIONNUMBER] >= 5.0F)
      {
         // In version 5.0 and later data files the Csynch array is at the
         // end of the file.

         pFH->lDataSectionPtr = 2L;
         pFH->lSynchArrayPtr  = 2L + (pFH->lActualAcqLength*2L + 511L) / 512L;
      }
      else
      {
         // For older data files just adjust the data pointer to allow
         // for the presence of the Csynch array before the data

         pFH->lSynchArrayPtr  = 2L;
         pFH->lDataSectionPtr = 2L + (pFH->lSynchArraySize*4L + 511L) / 512L;
      }
   }
   else
   {
      pFH->lSynchArrayPtr  = 0;
      pFH->lSynchArraySize = 0;

      if (Param[F53_FILEVERSIONNUMBER] >= 5.0F)
      {
         pFH->lDataSectionPtr = 2;
      }
      else 
      {
         // Adjust for older throughput data files that have a Csynch array
         // before the data anyway

         OldSynchArraySize = pFH->lActualAcqLength / (UINT)pFH->lNumSamplesPerEpisode;
         pFH->lDataSectionPtr = 2L + (OldSynchArraySize*4L + 511L) / 512L;
      }
   }
}

//===============================================================================================
// FUNCTION: FetchexConvert
// PURPOSE:  Convert an old FETCHEX file into the current ABF format.
//   
static BOOL FetchexConvert( FILEHANDLE hFile, ABFFileHeader *pFH, float *Param,
                            char *Comment, char *Label, int *pnError )
{
   char ADCLabel[ABF_ADCCOUNT*ABF_ADCUNITLEN];
   char ChannelName[ABF_ADCCOUNT*ABF_ADCNAMELEN];
   char DACLabel[ABF_DACCOUNT*ABF_DACUNITLEN];

   // Handle the case of old (pre-version 5.1) files.
   ABF_BLANK_FILL(ADCLabel);
   ABF_BLANK_FILL(DACLabel);
   ABF_BLANK_FILL(ChannelName);

   GetOldDACUnits(Label, DACLabel);

   if (Param[F53_FILEVERSIONNUMBER] < 5.0999F)     // MAC doen't like 5.1F
   {
      if (!FetchexToV5_3(Param))
         ERRORRETURN(pnError, ABFH_EINVALIDFILE);

      GetOldADCUnits(Label, ADCLabel);
   }
   else
   {
      // Handle the case of version 5.1 (and later) files.

      if (!ReadADCNames(hFile, ChannelName))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);

      if (!ReadADCInfo(hFile, Param, ADCLabel))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);

      // Get number of Tags if present

      if (Param[F53_TAGSECTIONPTR] != 0.0F)
      {
         int nNumTags = 0;

         // Some old versions of fetchex set this value to 2
         if (Param[F53_TAGSECTIONPTR] >= 3.0F)
         {
            // Seek to start of tag entries block
            long lNumBytes = (long)(Param[F53_TAGSECTIONPTR]) * 512L;
			c_SetFilePointer(hFile, lNumBytes, NULL, FILE_BEGIN);
            if (!ABFU_ReadFile(hFile, &nNumTags, 2))
               ERRORRETURN(pnError, ABFH_EHEADERREAD);
         }
         Param[F53_NUMTAGENTRIES] = (float)nNumTags;
      }
   }
   FetchexToABF1_x(Param, ADCLabel, DACLabel, ChannelName, Comment, pFH);
   return TRUE;
}

//===============================================================================================
// Code for translation of old CLAMPEX parameter and data file headers.
// It will handle two different file formats:
//
// 1. Data files (program versions before 5):
//   320  80 single-precision (4-byte) values in Microsoft BASIC binary format
//    77  77-byte ASCII character string for file comment
//    80  5 16-byte ASCII character strings for units-of-measure:
//          time-scale "us"
//          DAC channel #0
//          (unused)
//          ADC channel #0
//          ADC channel #1
//    35  35 bytes of environment data
//    64  64-byte ASCII character string for the presweep (conditioning) pulse parameters
//   448  bytes of unused header
//  ----  ---------------------------
//  1024  bytes total for file header
//
// 2. Parameter and data files (program versions 5 and later):
//
//   320  80 single-precision (4-byte) values in IEEE floating-point format
//    77  77-byte ASCII character string for file comment
//    80  5 16-byte ASCII character strings (80 bytes) for units-of-measure:
//          time-scale "us"
//          DAC channel #0
//          (unused)
//          (unused)
//          (unused)
//    35  35 bytes of environment data
//    64  64-byte ASCII character string for the presweep (conditioning) pulse parameters
//   320  80 single-precision (4-byte) values in IEEE floating-point format
//   128  16 8-byte ASCII character strings for units-of-measure:
//          ADC channel #0
//               .
//               .
//               .
//          ADC channel #15
//  ----  ---------------------------
//  1024  bytes total for file header
//
// See also the extensive documentation for the format of the Version 5
// parameter array.
//
// Support for pre- version 5 parameter has been dropped.
//
// NOTE: the parameter version number (Param[C52_FILEVERSIONNUMBER]) is NOT
//       updated here, since the trial routines need it to decide whether the
//       data is in averaged form (version 5+) or accumulated form (Version < 5).

//===============================================================================================
// FUNCTION: ReadCondit
// PURPOSE:  Read presweep (conditioning) train parameters from an old file.
//   
static BOOL ReadCondit(FILEHANDLE hFile, char *Condit)
{
   // Seek to the presweep (conditioning) string
	c_SetFilePointer(hFile, 512L, NULL, FILE_BEGIN);

   // Read the presweep (conditioning) pulse string.
   return ABFU_ReadFile(hFile, Condit, ABF_OLDCONDITLEN);
}

//===============================================================================================
// FUNCTION: ClampexToV5_2
// PURPOSE:  Convert CLAMPEX files prior to file version 5.2 to V5.2 format.
//   
#define STEPPED 1                    // stepped waveform
#define RAMPED  2                    // ramp waveform

static void ClampexToV5_2(float *Param)
{
   int i;

   // Set the default for the number of runs per trial.

   if (Param[C52_RUNSPERFILE] < 1.0F)
      Param[C52_RUNSPERFILE] = 1.0F;

   // Set the default for the number of episodes per run.

   if (Param[C52_EPISODESPERRUN] < 1.0F)
      Param[C52_EPISODESPERRUN] = 1.0F;

   // Set the default for the number of trials.

   if (Param[C52_NUMTRIALS] == 0.0F)
      Param[C52_NUMTRIALS] = 1.0F;

   // Set the default for the starting episode number.

   if (Param[C52_STARTEPISODENUM] == 0.0F)
      Param[C52_STARTEPISODENUM] = 1.0F;

   // Set the default for the number of data samples per episode.
   // (Very old versions of the program didn// t set Param[C52_SEGMENTSPEREPISODE],
   // since all episodes were 512 samples long)

   if (Param[C52_SEGMENTSPEREPISODE] < 1.0F)
      Param[C52_SEGMENTSPEREPISODE] = 1.0F;

   // Convert the DAC channel 0 gain from milli-volts to volts.

   Param[C52_GAINDACTOCELL] = Param[C52_GAINDACTOCELL] * 1000.0F;

   // Change the ADC multiplexer code (48) to the number of channels to sample.

   Param[C52_ADCNUMCHANNELS] = Param[C52_OLDMULTIPLEXCODE] + 1.0F;

   // Check that the number of channels is within range.

   if (Param[C52_ADCNUMCHANNELS] < 1)
      Param[C52_ADCNUMCHANNELS] = 1.0F;

   // Set the default for the stimulus duration to 10 samples.

   if (Param[C52_CH1PULSE] == 0.0F)
      Param[C52_CH1PULSE] = 10.0F;

   if (Param[C52_CH2PULSE] == 0.0F)
      Param[C52_CH2PULSE] = 10.0F;

   // Initialize the ADC scaling factors to 1.0

   for (i=0; i<ABF_ADCCOUNT; i++)
   {
      Param[C52_INSTSCALEFACTOR+i]  = 1.0F;
      Param[C52_ADCDISPLAYGAIN+i]   = 1.0F;
   }

   // Move the ADC gain parameters to their correct positions in the parameter
   // array, converting from milli-volts to volts along the way.

   if (Param[C52_OLDCHANNEL0GAIN] != 0.0F)
   {
      Param[C52_INSTSCALEFACTOR] = Param[C52_OLDCHANNEL0GAIN] / 1000.0F;
      Param[C52_INSTSCALEFACTOR+15] = Param[C52_INSTSCALEFACTOR];
   }
   if (Param[C52_OLDCHANNEL1GAIN] != 0.0F)
   {
      Param[C52_INSTSCALEFACTOR+1] = Param[C52_OLDCHANNEL1GAIN] / 1000.0F;
      Param[C52_INSTSCALEFACTOR+14] = Param[C52_INSTSCALEFACTOR+1];
   }

   if (Param[C52_CH0DISPLAYAMPLIFICATION] != 0.0F)
   {
      Param[C52_ADCDISPLAYGAIN] = Param[C52_CH0DISPLAYAMPLIFICATION];
      Param[C52_ADCDISPLAYGAIN+15] = Param[C52_CH0DISPLAYAMPLIFICATION];
   }
   Param[C52_ADCDISPLAYOFFSET] = Param[C52_CH0DISPLAYOFFSET];
   Param[C52_ADCDISPLAYOFFSET+15] = Param[C52_CH0DISPLAYOFFSET];

   if (Param[C52_CH1DISPLAYAMPLIFICATION] != 0.0F)
   {
      Param[C52_ADCDISPLAYGAIN+1] = Param[C52_CH1DISPLAYAMPLIFICATION];
      Param[C52_ADCDISPLAYGAIN+14] = Param[C52_CH1DISPLAYAMPLIFICATION];
   }
   Param[C52_ADCDISPLAYOFFSET+1] = Param[C52_CH1DISPLAYOFFSET];
   Param[C52_ADCDISPLAYOFFSET+14] = Param[C52_CH1DISPLAYOFFSET];

   // Check that ADC range and resolution were set properly

   if (Param[C52_ADCRANGE] == 0.0F)
      Param[C52_ADCRANGE] = 10.0F;
   if (Param[C52_ADCRESOLUTION] == 0.0F)
      Param[C52_ADCRESOLUTION] = 12.0F;
   if (Param[C52_DACRANGE] == 0.0F)
      Param[C52_DACRANGE] = 10.0F;
   if (Param[C52_DACRESOLUTION] == 0.0F)
      Param[C52_DACRESOLUTION] = 12.0F;

   // Readjust the command mode: handle the case of simple steps or step.

   if ((Param[C52_OLDPROTOCOLTYPE] == 4.0F) ||
      (Param[C52_OLDPROTOCOLTYPE] == 2.0F))
   {
      if (Param[C52_OLDPROTOCOLTYPE] == 2.0F)   // single step (version 3)
      {
         Param[C52_EPOCHCINITDURATION] = 0.0F; // C epoch duration
         Param[C52_EPOCHCDURATIONINC]  = 0.0F; // C epoch duration increment
      }

      // The B duration (57) is the difference between the A duration (30)
      // and the C duration (46).

      Param[C52_EPOCHBINITDURATION] = 500.0F * Param[C52_SEGMENTSPEREPISODE] -
            ( Param[C52_EPOCHAINITDURATION] + Param[C52_EPOCHCINITDURATION] );

      Param[C52_EPOCHATYPE] = (float)STEPPED;    // A epoch type = step
      Param[C52_EPOCHBTYPE] = (float)STEPPED;    // B epoch type = step
      Param[C52_EPOCHCTYPE] = (float)STEPPED;    // C epoch type = step
      Param[C52_EPOCHDTYPE] = (float)STEPPED;    // D epoch type = step

      Param[C52_EPOCHDINITDURATION] = 0.0F;  // D epoch duration
      Param[C52_EPOCHDDURATIONINC]  = 0.0F;  // D epoch duraion increment
   }

   // Handle the case of a triangular waveform.
   else if (Param[C52_OLDPROTOCOLTYPE] == 5.0F)
   {
      Param[C52_EPOCHATYPE] = (float)RAMPED;     // A epoch type = ramp
      Param[C52_EPOCHBTYPE] = (float)RAMPED;     // B epoch type = ramp
      Param[C52_EPOCHCTYPE] = (float)RAMPED;     // C epoch type = ramp
      Param[C52_EPOCHDTYPE] = (float)RAMPED;     // D epoch type = ramp
   }

   // Handle the case of a sawtooth waveform.
   else if (Param[C52_OLDPROTOCOLTYPE] == 6.0F)
   {
      // Load a sawtooth into the A epoch.

      // A initial level = sawtooth peak

      Param[C52_EPOCHALEVELINIT] = Param[C52_OLDEPOCHALEVEL];
      Param[C52_EPOCHAINCREMENT] = 0.0F;         // A level increment

      // A epoch duration = episode length

      Param[C52_EPOCHAINITDURATION] = 500.0F * Param[C52_SEGMENTSPEREPISODE];
      Param[C52_EPOCHADURATIONINC]  = 0.0F;  // A duration increment
      Param[C52_EPOCHATYPE] = (float)RAMPED;        // A epoch type = ramp

      // Clear the remaining epochs.

      Param[C52_EPOCHBINITDURATION] = 0.0F;     // B epoch duration
      Param[C52_EPOCHCINITDURATION] = 0.0F;     // C epoch duration
      Param[C52_EPOCHDINITDURATION] = 0.0F;     // D epoch duration
   }

   // Anything else will cause an error during the parameter check.
   else
   {
      // A epoch duration = episode length

      Param[C52_EPOCHAINITDURATION] = 500.0F * Param[C52_SEGMENTSPEREPISODE];
      Param[C52_EPOCHATYPE] = 0.0F;           // A epoch type = invalid
   }

   // Set the starting channel for the acquisition.

   Param[C52_ADCFIRSTLOGICALCHANNEL] = 0.0F;
   Param[C52_GAINMULTIPLIER]         = 1.0F;
   Param[C52_FILTERCUTOFF]           = ABF_FILTERDISABLED;
   Param[C52_AUTOSAMPLEINSTRUMENT]   = 3.0F;
   Param[C52_INTEREPISODEAMP]        = 0.0F;
   Param[C52_INTEREPISODEWRITE]      = 0.0F;

   // Convert the peak detection parameters, if present. This is done here
   // since Param[C52_AUTOPEAKCHANNEL], which is changed, contains display
   // information.

   if (Param[C52_AUTOPEAKSEARCHMODE] != 0.0F)
   {
      // Adjust the search mode to account for the C and D epochs.

      if (Param[C52_AUTOPEAKSEARCHMODE] >= 2.0F)
         Param[C52_AUTOPEAKSEARCHMODE] += 2.0F;

      // Set the number of the search channel.

      if (Param[C52_ADCNUMCHANNELS] == 2.0F)
         Param[C52_AUTOPEAKCHANNEL] = 1.0F;
      else
         Param[C52_AUTOPEAKCHANNEL] = 0.0F;
   }
   Param[C52_PNNUMPULSES]     = 0.0F;
   Param[C52_PNADCNUM]        = 0.0F;
   Param[C52_PNHOLDINGLEVEL]  = 0.0F;
   Param[C52_PNSETTLINGTIME]  = 0.0F;
   Param[C52_PNINTERPULSE]    = 0.0F;
}

//===============================================================================================
// FUNCTION: ClampexToABF1_x
// PURPOSE:  Convert old CLAMPEX parameters to ABF format.
//   
static void ClampexToABF1_x( float *Param, char *ADCLabel, char *DACLabel,
                             char *Condit, char *Comment, ABFFileHeader *pFH )
{
   short i, n;
   float EpiTime;

   // Initialize structure to all NULL's
   ABFH_Initialize(pFH);

   pFH->lFileSignature           = ABF_OLDPCLAMP;
   pFH->lActualEpisodes          = (long)(Param[C52_EPISODESPERFILE]);            // 4
   pFH->lActualAcqLength         = (long)(Param[C52_SAMPLESPEREPISODE]) *
                                   (long)(Param[C52_EPISODESPERFILE]);
   pFH->lFileStartTime           = (long)(Param[C52_FILESTARTTIME]);
   pFH->lStopwatchTime           = 
      (long)(Param[F53_FILESTARTTIME] - Param[F53_FILEELAPSEDTIME]);
   pFH->fFileVersionNumber       = Param[C52_FILEVERSIONNUMBER];
   pFH->fHeaderVersionNumber     = ABF_CURRENTVERSION;
   pFH->lFileStartDate           = (long)(Param[C52_FILESTARTDATE]);

   pFH->nExperimentType          = ABF_VOLTAGECLAMP;              // Voltage Clamp
   pFH->fADCSecondSampleInterval = Param[C52_SECONDCLOCKPERIOD];  
   pFH->fADCSampleInterval       = Param[C52_FIRSTCLOCKPERIOD];   
   pFH->nOperationMode           = ABF_WAVEFORMFILE;        // CLAMPEX file - waveform mode
   pFH->lPreTriggerSamples       = 0;                       // Fetchex only
   pFH->nManualInfoStrategy      = ABF_ENV_WRITEEACHTRIAL;   // default to write-each-trial
   pFH->nNumPointsIgnored        = 0;                       // Fetchex only
   pFH->lTagSectionPtr           = 0;
   pFH->lNumTagEntries           = 0;
   pFH->lNumSamplesPerEpisode    = (long)(Param[C52_SEGMENTSPEREPISODE]) * 512L;
   pFH->lClockChange             = 0;
   pFH->fSecondsPerRun           = 0.0F;
   pFH->_lDACFilePtr             = 0;
   pFH->_lDACFileNumEpisodes     = 0;
   pFH->lStartDisplayNum         = 1;
   pFH->lFinishDisplayNum        = 0;
   pFH->nMultiColor              = 1;
   if (Param[C52_DISPLAYSEGMENTNUM] != 0.0F)
   {
      pFH->lFinishDisplayNum = (long)(Param[C52_DISPLAYSEGMENTNUM]) * 512L;
      pFH->lStartDisplayNum = pFH->lFinishDisplayNum - 511L;
   }

   switch (short(Param[C52_AUTOSAMPLEINSTRUMENT]))
   {
      case 0:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEMANUAL;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 1:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 2:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1B;
         break;
      case 3:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEDISABLED;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
      case 4:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEAUTOMATIC;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH201;
         break;
      default:
         pFH->_nAutosampleEnable = ABF_AUTOSAMPLEDISABLED;
         pFH->_nAutosampleInstrument = ABF_INST_AXOPATCH1;
         break;
   }

   pFH->fCellID1                 = 0.0F;
   pFH->fCellID2                 = 0.0F;
   pFH->_fAutosampleMembraneCap   = 0.0F;
   pFH->fCellID3                 = 0.0F;

   if (Param[C52_GAINMULTIPLIER] == 0.0F)
      pFH->_fAutosampleAdditGain = 1.0F;
   else 
      pFH->_fAutosampleAdditGain = Param[C52_GAINMULTIPLIER];

   pFH->_fAutosampleFilter        = Param[C52_FILTERCUTOFF];               // 41  
   pFH->_nAutosampleADCNum        = short(Param[C52_AUTOSAMPLEADCNUM]);           // 96
   pFH->lNumberOfTrials          = long(Param[C52_NUMTRIALS]);           // 16  
   pFH->lRunsPerTrial            = long(Param[C52_RUNSPERFILE]);         // 12  
   pFH->lEpisodesPerRun          = long(Param[C52_EPISODESPERRUN]);      // 13  
   pFH->nADCNumChannels          = short(Param[C52_ADCNUMCHANNELS]);
   pFH->nFirstEpisodeInRun       = short(Param[C52_STARTEPISODENUM]);     // 17  
   pFH->fTrialStartToStart       = 0.0F;
   pFH->fEpisodeStartToStart     = Param[C52_INTEREPISODETIME];           // 11  
   pFH->fScopeOutputInterval     = Param[C52_STARTDELAY];                 // 15  
   pFH->fADCRange                = Param[C52_ADCRANGE];
   pFH->fDACRange                = Param[C52_DACRANGE];
   pFH->lADCResolution           = DataResolution(Param[C52_ADCRESOLUTION]);
   pFH->lDACResolution           = DataResolution(Param[C52_DACRESOLUTION]);
   pFH->lDeltaArrayPtr           = 0;
   pFH->lNumDeltas               = 0;
   pFH->nDrawingStrategy         = 1;      // Display data in real time
   pFH->nTiledDisplay            = 0;
   pFH->nEraseStrategy           = short(Param[C52_AUTOERASE]);                  // 69
   pFH->nChannelStatsStrategy    = 1;      // FETCHEX only
   pFH->lDisplayAverageUpdate    = short(Param[C52_AVERAGEDDATADISPLAY]);        // 65
   pFH->nDataDisplayMode         = short(Param[C52_DATADISPLAYMODE]);
   pFH->fTriggerThreshold        = 0.0F;      // FETCHEX only
   if (Param[C52_TRIGGERMODE] == 0.0F)
      pFH->nTriggerSource = short(Param[C52_ADCFIRSTLOGICALCHANNEL]);
   else
      pFH->nTriggerSource = short(-Param[C52_TRIGGERMODE]);
   pFH->nTriggerPolarity         = 0;      // FETCHEX only
   pFH->nAveragingMode           = ABF_NOAVERAGING;      // AXOTAPE only
   pFH->fSynchTimeUnit           = 0.0F;
   pFH->lSynchArraySize          = 0;      // FETCHEX only
   pFH->lSynchArrayPtr           = 0;      // FETCHEX only
   pFH->lCalculationPeriod       = 16384;  // FETCHEX only
   pFH->lSamplesPerTrace         = 16384;  // FETCHEX only
   pFH->nTriggerAction           = ABF_TRIGGER_STARTEPISODE;      // Start one episode
   pFH->nUndoRunCount            = 0;      // Disabled
   pFH->lDataSectionPtr          = 2;
   pFH->lAverageCount            = (long)(Param[C52_RUNSPERFILE]);
   pFH->fStatisticsPeriod        = pFH->lCalculationPeriod * pFH->fADCSampleInterval / 1E3F;

   strncpy(pFH->_sFileComment, Comment, ABF_OLDFILECOMMENTLEN);

   // Set up the channel mapping
   // Channel numbers are always ascending for old CLAMPEX files
   // (AXOLAB-1, AXOLAB 1100, TL-1-125, TL-1-40, TL-3)
   // NOTE: AXOLAB 1100 only supports 8 channels (0-7)

   for (i=0; i<ABF_ADCCOUNT; i++)
      pFH->nADCPtoLChannelMap[i] = i;

   n = short(Param[C52_ADCFIRSTLOGICALCHANNEL]);
   for (i=0; i<pFH->nADCNumChannels; i++, n++)
      pFH->nADCSamplingSeq[i] = n;

   // Set rest of sampling sequence to -1

   for (i=pFH->nADCNumChannels; i<ABF_ADCCOUNT; i++)
      pFH->nADCSamplingSeq[i] = -1;

   // Set up the ADC channel info

   for (i=0; i<ABF_ADCCOUNT; i++)
   {
      pFH->fInstrumentOffset[i]        = 0.0F;
      pFH->fInstrumentScaleFactor[i]   = 1.0F;
      pFH->fADCDisplayAmplification[i] = 1.0F;
      pFH->fADCDisplayOffset[i]        = 0.0F;
      pFH->fSignalLowpassFilter[i]     = ABF_FILTERDISABLED;
      pFH->fSignalHighpassFilter[i]    = 0.0F;
      pFH->fADCProgrammableGain[i]     = 1.0F;
      pFH->fSignalGain[i]              = 1.0F;
      pFH->fSignalOffset[i]            = 0.0F;
   }

   for (i=0; i < ABF_ADCCOUNT; i++)
   {
      // no physical to logical mapping necessary as CLAMPEX never supported
      // the TL2

      strncpy(pFH->sADCUnits[i], ADCLabel+ABF_ADCUNITLEN*i, ABF_ADCUNITLEN);

      // protect against divide by zero errors from bad gains.

      if (Param[C52_INSTSCALEFACTOR + i] != 0.0F)
         pFH->fInstrumentScaleFactor[i] = Param[C52_INSTSCALEFACTOR + i];

      if (Param[C52_ADCDISPLAYGAIN + i] != 0.0F)
         pFH->fADCDisplayAmplification[i] = Param[C52_ADCDISPLAYGAIN + i];

      pFH->fInstrumentOffset[i] = Param[C52_INSTOFFSET + i];
      pFH->fADCDisplayOffset[i] = Param[C52_ADCDISPLAYOFFSET + i] *
                                  pFH->fADCRange /
                                  pFH->fInstrumentScaleFactor[i] /
                                  pFH->fADCDisplayAmplification[i];
      if ((pFH->_nAutosampleEnable != 0) &&
         (i == pFH->_nAutosampleADCNum))
         pFH->fADCDisplayOffset[i] /= pFH->_fAutosampleAdditGain;
   }

   strncpy(pFH->sDACChannelUnits[0], DACLabel, ABF_DACCOUNT * ABF_DACUNITLEN);
   for (i=0; i<ABF_DACCOUNT; i++)
   {
      pFH->fDACScaleFactor[i] = Param[C52_GAINDACTOCELL];
      pFH->fDACHoldingLevel[i] = Param[C52_DAC0HOLDINGLEVEL];
   }

   // CLAMPEX specific parameter sections follow ...

   // GROUP #6 (14 bytes) - Synchronous timer outputs.
   if (((Param[C52_CH1PULSE] != 0.0F) || (Param[C52_CH2PULSE] != 0.0F)) &&
      (Param[C52_LASTTRIGGEREPISODE] != 0.0F) )
      pFH->nOUTEnable = 1;
   else
      pFH->nOUTEnable = 0;

   pFH->nSampleNumberOUT1        = short(Param[C52_PULSESAMPLECH1]);
   pFH->nSampleNumberOUT2        = short(Param[C52_PULSESAMPLECH2]);
   pFH->nFirstEpisodeOUT         = short(Param[C52_FIRSTTRIGGEREPISODE]);
   pFH->nLastEpisodeOUT          = short(Param[C52_LASTTRIGGEREPISODE]);     
   pFH->nPulseSamplesOUT1        = short(Param[C52_CH1PULSE]);     
   pFH->nPulseSamplesOUT2        = short(Param[C52_CH2PULSE]);     

   // GROUP #7 (172 bytes) - Epoch Output Waveform and Pulses
   pFH->nDigitalEnable           = 0;
   pFH->_nWaveformSource         = 1;   // Waveform from epochs.
   pFH->nActiveDACChannel        = 0;
   pFH->_nInterEpisodeLevel      = short(Param[C52_INTEREPISODEAMP]);

   pFH->_nEpochType[0]           = short(Param[C52_EPOCHATYPE]);
   pFH->_fEpochInitLevel[0]       = Param[C52_EPOCHALEVELINIT];
   pFH->_fEpochLevelInc[0]        = Param[C52_EPOCHAINCREMENT];
   pFH->_nEpochInitDuration[0]    = short(int(Param[C52_EPOCHAINITDURATION]) / pFH->nADCNumChannels);
   pFH->_nEpochDurationInc[0]     = short(int(Param[C52_EPOCHADURATIONINC]) / pFH->nADCNumChannels);

   pFH->_nEpochType[1]            = short(Param[C52_EPOCHBTYPE]);
   pFH->_fEpochInitLevel[1]       = Param[C52_EPOCHBLEVELINIT];
   pFH->_fEpochLevelInc[1]        = Param[C52_EPOCHBINCREMENT];
   pFH->_nEpochInitDuration[1]    = short(int(Param[C52_EPOCHBINITDURATION]) / pFH->nADCNumChannels);
   pFH->_nEpochDurationInc[1]     = short(int(Param[C52_EPOCHBDURATIONINC]) / pFH->nADCNumChannels);

   pFH->_nEpochType[2]            = short(Param[C52_EPOCHCTYPE]);
   pFH->_fEpochInitLevel[2]       = Param[C52_EPOCHCLEVELINIT];
   pFH->_fEpochLevelInc[2]        = Param[C52_EPOCHCINCREMENT];
   pFH->_nEpochInitDuration[2]    = short(int(Param[C52_EPOCHCINITDURATION]) / pFH->nADCNumChannels);
   pFH->_nEpochDurationInc[2]     = short(int(Param[C52_EPOCHCDURATIONINC]) / pFH->nADCNumChannels);

   pFH->_nEpochType[3]            = short(Param[C52_EPOCHDTYPE]);
   pFH->_fEpochInitLevel[3]       = Param[C52_EPOCHDLEVELINIT];
   pFH->_fEpochLevelInc[3]        = Param[C52_EPOCHDINCREMENT];
   pFH->_nEpochInitDuration[3]    = short(int(Param[C52_EPOCHDINITDURATION]) / pFH->nADCNumChannels);
   pFH->_nEpochDurationInc[3]     = short(int(Param[C52_EPOCHDDURATIONINC]) / pFH->nADCNumChannels);

   for (i=0; i<4; i++)
   {
      if ((pFH->_nEpochInitDuration[i] == 0) && 
         (pFH->_nEpochDurationInc[i] == 0))
         pFH->_nEpochType[i] = 0;
   }

   for (i=4; i<ABF_EPOCHCOUNT; i++)
   {
      pFH->_nEpochType[i]        = 0;
      pFH->_fEpochInitLevel[i]   = 0.0F;
      pFH->_fEpochLevelInc[i]    = 0.0F;
      pFH->_nEpochInitDuration[i]= 0;   
      pFH->_nEpochDurationInc[i] = 0;   
   }

   pFH->nDigitalHolding         = 0;
   pFH->nDigitalInterEpisode    = 0;
   pFH->nDigitalTrainActiveLogic= 1;
   for (i=0; i<ABF_EPOCHCOUNT; i++)
   {
      pFH->nDigitalValue[i]              = 0;
      pFH->nDigitalTrainValue[i]         = 0;
   }

   // GROUP #8 (80 bytes) - Analog Output File Waveform
   pFH->_fDACFileScale           = 1.0F;
   pFH->_fDACFileOffset          = 0.0F;
   pFH->_nDACFileEpisodeNum      = 1;
   pFH->_nDACFileADCNum          = 1;

   // GROUP #9 (32 bytes) - Presweep (conditioning) pulse train.
   if (Param[C52_PULSESINTRAIN] != 0.0F)
      pFH->_nConditEnable = 1;
   else
      pFH->_nConditEnable = 0;
   pFH->_nConditChannel           = 0;
   pFH->_lConditNumPulses         = (long)(Param[C52_PULSESINTRAIN]);
   pFH->_fBaselineDuration        = Param[C52_PRECONDURATION];     
   pFH->_fBaselineLevel           = Param[C52_PRECONLEVEL];     
   pFH->_fStepDuration            = Param[C52_CONDURATION];     
   pFH->_fStepLevel               = Param[C52_CONLEVEL];     
   pFH->_fPostTrainPeriod         = Param[C52_POSTCONDURATION];     
   pFH->_fPostTrainLevel          = Param[C52_POSTCONLEVEL];     

   // GROUP #10 ( 82 bytes) - Variable parameter list.

   // remap old parameter number to start at 1 for C52_PULSESINTRAIN. See const
   // definitions in ABFFILES.INC

   if (Param[C52_CONDITVARIABLE] != 0)
   {
      pFH->_nListEnable  = 1;
      pFH->_nParamToVary =
            short(Param[C52_CONDITVARIABLE] - C52_PULSESINTRAIN + 0.5F);
   }
   else
   {
      pFH->_nListEnable  = 0;
      pFH->_nParamToVary = 0;
   }

   strncpy(pFH->_sParamValueList, Condit, ABF_OLDCONDITLEN);

   // GROUP #11 (14 bytes) - Autopeak measurement.
   
   if (Param[C52_AUTOPEAKSEARCHMODE] == 0)
      pFH->nStatsEnable = 0;
   else
      pFH->nStatsEnable = 1;

   UINT uBitMask = 0x01 << short(Param[C52_AUTOPEAKCHANNEL]);
   pFH->nStatsActiveChannels = (short)uBitMask;

   n = short(Param[C52_AUTOPEAKSEARCHMODE]);
   for (UINT uRegion = 0; uRegion < ABF_STATS_REGIONS; uRegion++)
   {
      switch (n)
      {
         case 5:
            pFH->nStatsSearchMode[ uRegion ] = ABF_PEAK_SEARCH_ALL;
            break;
         case 6:
            pFH->nStatsSearchMode[ uRegion ] = 0;
            break;
         default:
            pFH->nStatsSearchMode[ uRegion ] = short(Param[C52_AUTOPEAKSEARCHMODE]);
            break;
      }
      pFH->lStatsStart[ uRegion ] = 0;
      pFH->lStatsEnd[ uRegion ]   = 0;
   }

   for (UINT uChannel = 0; uChannel < ABF_ADCCOUNT; uChannel++)
   {
      if (Param[C52_AUTOPEAKCENTER] < 0)
         pFH->nStatsChannelPolarity[ uChannel ] = ABF_PEAK_NEGATIVE;
      else
         pFH->nStatsChannelPolarity[ uChannel ] = ABF_PEAK_POSITIVE;
   }

   pFH->nStatsBaseline  = short(Param[C52_BASELINECALCULATION]);
   pFH->nStatsSmoothing = short(Param[C52_AUTOPEAKAVPOINTS]);

   if (pFH->nStatsSmoothing < 1)
      pFH->nStatsSmoothing = 1;

   // GROUP #12 (40 bytes) - Channel Arithmetic
   pFH->nArithmeticEnable        = 0;
   pFH->fArithmeticUpperLimit    = 0.0F;
   pFH->fArithmeticLowerLimit    = 0.0F;
   pFH->nArithmeticADCNumA       = 0;
   pFH->nArithmeticADCNumB       = 0;
   pFH->fArithmeticK1            = 0.0F;
   pFH->fArithmeticK2            = 0.0F;
   pFH->fArithmeticK3            = 0.0F;
   pFH->fArithmeticK4            = 0.0F;
   pFH->fArithmeticK5            = 0.0F;
   pFH->fArithmeticK6            = 0.0F;
   strncpy(pFH->sArithmeticOperator, "+ ", ABF_ARITHMETICOPLEN);
   pFH->nArithmeticExpression    = ABF_SIMPLE_EXPRESSION;

   // GROUP #13 (20 bytes) - On-line subtraction.
   if (Param[C52_PNNUMPULSES] != 0)
   {
      pFH->_nPNEnable    = 1;
      pFH->nPNNumPulses = short(fabs(Param[C52_PNNUMPULSES]));
   }
   else
   {
      pFH->_nPNEnable    = 0;
      pFH->nPNNumPulses = 1;
   }

   if (Param[C52_PNNUMPULSES] < 0)
      pFH->_nPNPolarity = ABF_PN_SAME_POLARITY;
   else 
      pFH->_nPNPolarity = ABF_PN_OPPOSITE_POLARITY;
   pFH->nPNPosition      = 0;
   pFH->_nPNADCNum       = short(Param[C52_PNADCNUM]);
   pFH->_fPNHoldingLevel = Param[C52_PNHOLDINGLEVEL];     
   pFH->fPNSettlingTime = Param[C52_PNSETTLINGTIME];     
   pFH->fPNInterpulse   = Param[C52_PNINTERPULSE];
   if (pFH->fPNInterpulse > 0.0F)
   {
      if (pFH->fADCSecondSampleInterval == 0.0F)
         EpiTime = pFH->fADCSampleInterval * pFH->lNumSamplesPerEpisode;
      else 
         EpiTime = (pFH->fADCSampleInterval + pFH->fADCSecondSampleInterval) *
                   pFH->lNumSamplesPerEpisode/2;
      pFH->fPNInterpulse += EpiTime / 1E3F;
   }
}

//===============================================================================================
// FUNCTION: ClampexConvert
// PURPOSE:  Convert an old CLAMPEX file to the current ABF format.
//   
static int ClampexConvert( FILEHANDLE hFile, ABFFileHeader *pFH, float *Param,
                           char *Comment, char *Label, int *pnError )
{
   char ADCLabel[ABF_ADCCOUNT*ABF_ADCUNITLEN];
   char DACLabel[ABF_DACCOUNT*ABF_DACUNITLEN];
   char ConditStr[ABF_OLDCONDITLEN];

   ABF_BLANK_FILL(ADCLabel);
   ABF_BLANK_FILL(DACLabel);
   ABF_BLANK_FILL(ConditStr);
   GetOldDACUnits(Label, DACLabel);

   // Handle the case of old (pre-version 5) files.

   if (Param[C52_FILEVERSIONNUMBER] < 5.0F)
   {
      ClampexToV5_2(Param);
      GetOldADCUnits(Label, ADCLabel);

      // Read in the conditioning string

      if (!ReadCondit(hFile, ConditStr))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);
   }
   else    // Handle the case of version 5 (and later) files.
   {
      if (Param[C52_FILEVERSIONNUMBER] < 5.1999F)    // MAC doen't like 5.2F
      {
         Param[C52_GAINMULTIPLIER]       = 1.0F;
         Param[C52_FILTERCUTOFF]         = ABF_FILTERDISABLED;
         Param[C52_AUTOSAMPLEINSTRUMENT] = 3.0F;
         Param[C52_INTEREPISODEAMP]      = 0.0F;
         Param[C52_INTEREPISODEWRITE]    = 0.0F;
      }
      else if (Param[C52_AUTOSAMPLEINSTRUMENT] == 3.0F)
      {
         Param[C52_GAINMULTIPLIER] = 1.0F;
         Param[C52_FILTERCUTOFF]   = ABF_FILTERDISABLED;
      }

      // Read in the conditioning string
      if (!ReadCondit(hFile, ConditStr))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);

      // Read the extended parameters (81-96). (16 x 4-byte floats)
      if (!ABFU_ReadFile(hFile, Param+80, 16 * sizeof(float)))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);

      if (!ReadADCInfo(hFile, Param, ADCLabel))
         ERRORRETURN(pnError, ABFH_EHEADERREAD);
   }

   ClampexToABF1_x(Param, ADCLabel, DACLabel, ConditStr, Comment, pFH);
   return TRUE;
}

//===============================================================================================
// FUNCTION: ReadOldHeader
// PURPOSE:  This is the main routine to read old FETCHEX and CLAMPEX data headers.
//
BOOL OLDH_ReadOldHeader( FILEHANDLE hFile, UINT uFileType, int bMSBinFormat,
                         ABFFileHeader *pFH, long lFileLength, int *pnError)
{
//   ABFH_WASSERT(pFH);
   int     i;
   BOOL    bRval;
   float   Param[ABF_OLDPARAMSIZE];     // Allocate old arrays for reading the data
   char    Comment[ABF_OLDCOMMENTLEN];
   char    Label[5 * ABF_OLDUNITLEN];

   // Read parameters 1-80    (80 x 4-byte floats)
   if (!ABFU_ReadFile(hFile, Param, 80 * sizeof(float)))
      ERRORRETURN(pnError, ABFH_EHEADERREAD);

   // convert the parameters to IEEE floating-point format if necessary.
   if (bMSBinFormat)
   {
      for (i=0; i < 80; i++)
         fMSBintoIeee(Param+i, Param+i);
      for(i=80; i < ABF_OLDPARAMSIZE; i++)
         Param[i] = 0.0F;
   }

   // Read the file comment.
   if (!ABFU_ReadFile(hFile, Comment, ABF_OLDCOMMENTLEN))
      ERRORRETURN(pnError, ABFH_EHEADERREAD);

   // Read the units-of-measure strings into a single string.

   if (!ABFU_ReadFile(hFile, Label, sizeof(Label)))
      ERRORRETURN(pnError, ABFH_EHEADERREAD);

   if (uFileType == ABF_FETCHEX)
      bRval = FetchexConvert(hFile, pFH, Param, Comment, Label, pnError);
   else
      bRval = ClampexConvert(hFile, pFH, Param, Comment, Label, pnError);

   pFH->nFileType    = short(uFileType);
   pFH->nMSBinFormat = short(bMSBinFormat);
   
   //
   // Check data length against the length of the file. Some old versions of AXOTAPE
   // incorrectly set the number of 512 byte blocks per episode to 512.
   //
   long lMaxLength = (lFileLength - pFH->lDataSectionPtr * 512L - pFH->lSynchArraySize * 8L) / 2L;
   if (pFH->lActualAcqLength > lMaxLength)
   {
      if ((pFH->nOperationMode != ABF_VARLENEVENTS) && (pFH->nOperationMode != ABF_GAPFREEFILE))
      {
         long lNumEpisodes = lMaxLength / pFH->lNumSamplesPerEpisode;
         pFH->lActualEpisodes = lNumEpisodes;

         if (pFH->lSynchArraySize != 0L)
            pFH->lSynchArraySize = lNumEpisodes;

         lMaxLength = lNumEpisodes * pFH->lNumSamplesPerEpisode;
      }
      pFH->lActualAcqLength = lMaxLength;
   }

   // Bring the header up to the current version.
   OLDH_ABFtoABF15(pFH);
   OLDH_ABFtoCurrentVersion(pFH);

   return bRval;
}

