#pragma once
#include "BaseRenderer.h"

class CLUT_RGBRenderer :
	public CBaseRenderer
{
public:
	CLUT_RGBRenderer(void);
	~CLUT_RGBRenderer(void);

	void Render(byte * bmp, CDataStack * ds, int zpos=-1);
protected:
	double rGainOld, gGainOld, bGainOld;
	int rOffOld, gOffOld, bOffOld;
	
	//LUTS - only suitable for 12 bit data!!!
	unsigned char redLUT[4096];
	unsigned char greenLUT[4096];
	unsigned char blueLUT[4096];

	void GenerateLUT(unsigned char *LUT, float Gain, float Off);
};
