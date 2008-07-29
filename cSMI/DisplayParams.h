/*Virtual base class ('interface') for the display parameters
Currently these are implemented within the code for the DisplayOptions
dialog, but it would be bad if a Renderer, whose only job is to convert
a slice of a DataStack into a bitmap had to know about a dialog.*/

#pragma once
#ifndef DisplayParamsH
#define DisplayParamsH

class CDisplayParams
{
public:
    /*const static*/ enum Slice {SLICE_XY,SLICE_XZ,SLICE_YZ};
	/*const static*/ enum Orientation {UPRIGHT, ROT90};

    CDisplayParams(void);
    ~CDisplayParams(void);
    virtual int getDisp1Chan()=0;
    virtual int getDisp2Chan()=0;
    virtual int getDisp3Chan()=0;
    virtual double getDisp1Gain()=0;
    virtual double getDisp2Gain()=0;
    virtual double getDisp3Gain()=0;
    virtual int getDisp1Off()=0;
    virtual int getDisp2Off()=0;
    virtual int getDisp3Off()=0;
	virtual int getOrientation()=0;
	
    virtual Slice getSliceAxis(){return SLICE_XY;}
};

#endif
