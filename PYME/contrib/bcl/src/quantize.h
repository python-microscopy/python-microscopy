#ifndef _quantize_h_
#define _quantize_h_

#ifdef __cplusplus
extern "C" {
#endif

//#include <x86intrin.h>
#include <stdint.h>

void quantize_u16(uint16_t *data, uint8_t * out, int size, float offset, float scale);
void quantize_u16_avx( uint16_t * data, uint8_t * out, int size, float offset, float scale);

#ifdef __cplusplus
}
#endif

#endif /* _quantize_h_ */
