//#ifdef __MSVC__
//#include "stdafx.h"
//#endif

#include "displayopts.h"

CDisplayOpts::CDisplayOpts(void)
{
	disp1chan = -1;
	disp2chan = -1;
	disp3chan = -1;

	disp1gain = 1;
	disp2gain = 1;
	disp3gain = 1;

	disp1off = 0;
	disp2off = 0;
	disp3off = 0;
	
	flip[0] = 0;
	flip[1] = 0;
	flip[2] = 0;

	orientation = UPRIGHT;
	slice = SLICE_XY;
}

CDisplayOpts::~CDisplayOpts(void)
{
}

void CDisplayOpts::Optimise(CDataStack *ds)
{
	float min1 = 5000, min2 = 5000, min3 = 5000;
	float max1 = 0, max2 = 0, max3 = 0;

	unsigned short *c1, *c2, *c3; //pointers to accept the data for each channel
	unsigned short cz = 0; //set up some dummy data for null channels

	//if we're imaging a channel set the relavent pointers, otherwise make them point to the dummy data.
	if (getDisp1Chan() >= 0) {c1 = ds->getCurrentChannelSlice(getDisp1Chan());} else c1 = &cz;
	if (getDisp2Chan() >= 0) {c2 = ds->getCurrentChannelSlice(getDisp2Chan());} else c2 = &cz;
	if (getDisp3Chan() >= 0) {c3 = ds->getCurrentChannelSlice(getDisp3Chan());} else c3 = &cz;

	//loop through each pixel in the slice
	for(int i=0; i < ds->getWidth()*ds->getHeight(); i++)
	{
		min1 = min(min1, (float)*c1);
		min2 = min(min2, (float)*c2);
		min3 = min(min3, (float)*c3);

		max1 = max(max1, (float)*c1);
		max2 = max(max2, (float)*c2);
		max3 = max(max3, (float)*c3);

		if (c1 != &cz) c1++; //only increment the pointer if its not the dummy data
		if (c2 != &cz) c2++;
		if (c3 != &cz) c3++;
	}
	
	
	//Set offset to be the lowest value in each channel
	setDisp1Off(min1);
	setDisp2Off(min2);
	setDisp3Off(min3);

	//Set the gain such that we use the full 256 value depth
	//of each colour channel
	setDisp1Gain(256/(max1 - min1));
	setDisp2Gain(256/(max2 - min2));
	setDisp3Gain(256/(max3 - min3));

}
