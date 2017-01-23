//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// MODULE:  MSBINCVT.CPP
// PURPOSE: Provides functions to convert 4-byte floating point numbers
//          from Microsoft Binary format to IEEE format and vice-versa.
// REFERENCES:
//     MASM 5.1 programmer's guide (does not mention IEEE bias)
//
/*****************************************************************************
**
**   The Microsoft 4-byte real format looks like this:
**
**   Byte 3   Byte 2   Byte 1   Byte 0
**   76543210 76543210 76543210 76543210
**   EEEEEEEE SMMMMMMM MMMMMMMM MMMMMMMM
**
**   The exponent occupies the 8 bits of byte 3. This is a biased exponent,
**   the bias is 0x81.
**   The sign bit is bit 7 of byte 2. 0 denotes positive, 1 negative.
**   The mantissa is in the remaining 23 bits.
**
**   It seems (from single steppong through the _fmsbintoieee() function in
**   the Microsoft C++ runtime libraby) that a number is deemed to be zero
**   if its exponent is less than (unsigned)2.
**
**
**   The IEEE 4-byte real format looks like this:
**
**   Byte 3   Byte 2   Byte 1   Byte 0
**   76543210 76543210 76543210 76543210
**   SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
**
**   The sign bit is bit 7 of byte 3. 0 denotes positive, 1 negative.
**   The exponent occupies the the lower 7 bits of byte 3 and bit 7 of byte 2.
**   This is a biased exponent, the bias is 0x7f.
**   The mantissa is in the remaining 23 bits.
**
*****************************************************************************/
#include "../Common/wincpp.hpp"
#include "msbincvt.h"

struct IEEEBITS
{
   unsigned mantissaLo:16;
   unsigned mantissaHi:7;
   unsigned exponent:8;
   unsigned sign:1;
};

typedef union
{
   float           fVal;
   struct IEEEBITS bits;
} IEEEFLOAT;

struct MSBBITS
{
   unsigned mantissaLo:16;
   unsigned mantissaHi:7;
   unsigned sign:1;
   unsigned exponent:8;
};

typedef union
{
   float          fVal;
   struct MSBBITS bits;
} MSBFLOAT;


void fMSBintoIeee(float *pfIn, float *pfOut)
{
   MSBFLOAT  msb;
   msb.fVal = *pfIn;
   if (msb.bits.exponent < 2)
   {
      *pfOut = 0.0F;
      return;
   }

   IEEEFLOAT ieee;
   ieee.bits.sign       = msb.bits.sign;
   ieee.bits.exponent   = msb.bits.exponent - 0x81 + 0x7F;
   ieee.bits.mantissaLo = msb.bits.mantissaLo;
   ieee.bits.mantissaHi = msb.bits.mantissaHi;
   *pfOut = ieee.fVal;
}

void fIeeetoMSBin(float *pfIn, float *pfOut)
{
   IEEEFLOAT ieee;
   ieee.fVal = *pfIn;

   if (ieee.fVal == 0.0F)     /* Zero is a special case - same in both formats. */
   {
      *pfOut = 0.0F;
      return;
   }

   MSBFLOAT  msb;
   msb.bits.sign       = ieee.bits.sign;
   msb.bits.exponent   = ieee.bits.exponent - 0x7F + 0x81;
   msb.bits.mantissaLo = ieee.bits.mantissaLo;
   msb.bits.mantissaHi = ieee.bits.mantissaHi;
   *pfOut = msb.fVal;
}
