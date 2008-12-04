/************************************************************************************************
**
**    Copyright (c) 1994-1997 Axon Instruments.
**    All rights reserved.
**
*************************************************************************************************
** HEADER:  ADCDAC.H
** PURPOSE: Contains #defines for working with ADC and DAC values.
** AUTHOR:  BHI  Jul 1994
*/

#ifndef INC_ADCDAC_H
#define INC_ADCDAC_H

//
// ADC values are 16 bit integers collected from Analog-to-Digital converters
//
#define ADC_MAX 32767
#define ADC_MIN -32768
typedef short ADC_VALUE;

//
// DAC values are 16 bit integers for output by Digital-to-Analog converters
//
#define DAC_MAX 32767
#define DAC_MIN -32768
typedef short DAC_VALUE;

//
// Define a linked list structure for holding acquisition buffers.
//
struct DATABUFFER
{
   UINT        uNumSamples;      // Number of samples in this buffer.
   UINT        uFlags;           // Flags discribing the data buffer.
   ADC_VALUE  *pnData;           // The buffer containing the data.
   BYTE       *psDataFlags;      // Flags split out from the data buffer.
   DATABUFFER *pNextBuffer;      // Next buffer in the list.
   DATABUFFER *pPrevBuffer;      // Previous buffer in the list.
};
typedef DATABUFFER *PDATABUFFER;


//
// Define a linked list structure for holding floating point acquisition buffers.
//
struct FLOATBUFFER
{
   UINT         uNumSamples;  // Number of samples in this buffer.
   UINT         uFlags;       // Flags discribing the data buffer.
   float       *pfData;       // The buffer containing the data.
   FLOATBUFFER *pNextBuffer;  // Next buffer in the list.
   FLOATBUFFER *pPrevBuffer;  // Previous buffer in the list.
};
typedef FLOATBUFFER *PFLOATBUFFER;


#endif   // INC_ADCDAC_H
