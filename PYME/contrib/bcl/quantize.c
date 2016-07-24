//
//  quantize.c
//
//  Created by David Baddeley on 6/11/16.
//  Copyright © 2016 David Baddeley. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <x86intrin.h>
#include "quantize.h"

//#include "systimer.h"

#define TESTSIZE (2000*2000)
#define NUMITERS 10
#define NUMITERS_1 1000

/* square root quantize data, with a given offset and scale*/
void quantize_u16(uint16_t *data, uint8_t * out, int size, float offset, float scale)
{
    float qs = 1.0/scale;
    int i = 0;

    for (i = 0; i < size; i++)
    {
        out[i] = qs*sqrtf(data[i] - offset);
    }
}

/* square root quantize data, with a given offset and scale
uses avx command set to process 16 values in parallel
*/
void quantize_u16_avx( uint16_t * data, uint8_t * out, int size, float offset, float scale)
{
    float qs = 1.0/scale;
    int i = 0;

    __m256 x, x1, xs;
    __m128i t2, xlo, xhi, xp1, xp2, xpp;
    __m256i combined, xi, xp;

    __m256 offs;
    __m256 sc;
    offs = _mm256_set1_ps(offset);
    sc = _mm256_set1_ps(scale);

    /*process 16 values at a time*/

    for (i = 0; i < size; i+=16)
    {
        /* process first 8 values */
        t2 = _mm_load_si128((__m128i *) &(data[i]));
        xlo = _mm_unpacklo_epi16(t2, _mm_set1_epi16(0));
        xhi = _mm_unpackhi_epi16(t2, _mm_set1_epi16(0));
        //xhi = _mm_unpackhi_epi16( _mm_set1_epi16(0), t2);
        combined = (__m256i)_mm256_loadu2_m128i (&xhi, &xlo);
        x = (__m256)_mm256_cvtepi32_ps(combined);
        x1 = (__m256)_mm256_sub_ps(x, offs);
        xs = (__m256)_mm256_mul_ps(_mm256_mul_ps(_mm256_rsqrt_ps(x1),x1), sc);
        //xs = _mm256_mul_ps(_mm256_sqrt_ps(x1), sc);
        xi = (__m256i)_mm256_cvttps_epi32 (xs);
        xp = xi;
        //xp = _mm256_packs_epi32 (xi, _mm256_set1_epi32 (0));
        //xp = _mm256_packs_epi16 (xp, _mm256_set1_epi16 (0));
        /*xp =  _mm256_shuffle_epi8 (xi, _mm256_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                                                       0x80, 0x80, 0x80, 0x80, 28, 24, 20, 16,
                                                       0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                                                       0x80, 0x80, 0x80, 0x80, 12,   8,  4,  0));
        */
         xp1 = _mm_packus_epi16 (_mm256_extractf128_si256 (xp, 0), _mm256_extractf128_si256 (xp, 1));

        /* process next 8 values */
        t2 = _mm_load_si128((__m128i *) (&(data[i+8])));
        xlo = _mm_unpacklo_epi16(t2, _mm_set1_epi16(0));
        xhi = _mm_unpackhi_epi16(t2, _mm_set1_epi16(0));
        combined = (__m256i)_mm256_loadu2_m128i (&xhi, &xlo);
        x = (__m256)_mm256_cvtepi32_ps(combined);
        x1 = (__m256)_mm256_sub_ps(x, offs);
        xs = (__m256)_mm256_mul_ps(_mm256_mul_ps(_mm256_rsqrt_ps(x1),x1), sc);
        //xs = _mm256_mul_ps(_mm256_sqrt_ps(x1), sc);
        xi = (__m256i)_mm256_cvttps_epi32 (xs);
        xp = xi;
        //xp = _mm256_packs_epi32 (xi, _mm256_set1_epi32 (0));
        //xp2 = _mm256_extractf128_si256 (xp, 0);
        xp2 = _mm_packus_epi16 (_mm256_extractf128_si256 (xp, 0), _mm256_extractf128_si256 (xp, 1));

        xpp = _mm_packus_epi16 (xp1, xp2);
        _mm_store_si128((__m128i *)&(out[i]), xpp);

        //out += 16

        //out[i] = qs*sqrtf(data[i] - offset);
    }
}
