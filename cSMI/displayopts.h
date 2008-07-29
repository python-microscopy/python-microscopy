#pragma once
#ifndef DisplayOptsH
#define DisplayOptsH

#include "DisplayParams.h"
#include "DataStack.h"

class CDisplayOpts : public CDisplayParams
{
private:
	int disp1chan,disp2chan,disp3chan;
	double disp1gain,disp2gain,disp3gain;
	int disp1off,disp2off,disp3off;
	int orientation;
	Slice slice;

public:
    CDisplayOpts(void);
    ~CDisplayOpts(void);
	int getDisp1Chan(){return disp1chan;};
    int getDisp2Chan(){return disp2chan;};
    int getDisp3Chan(){return disp3chan;};
    double getDisp1Gain(){return disp1gain;};
    double getDisp2Gain(){return disp2gain;};
    double getDisp3Gain(){return disp3gain;};
    int getDisp1Off(){return disp1off;};
    int getDisp2Off(){return disp2off;};
    int getDisp3Off(){return disp3off;};
	int getOrientation(){return orientation;};

	void setDisp1Chan(int v){disp1chan = v;};
    void setDisp2Chan(int v){disp2chan = v;};
    void setDisp3Chan(int v){disp3chan = v;};
    void setDisp1Gain(double v){disp1gain = v;};
    void setDisp2Gain(double v){disp2gain = v;};
    void setDisp3Gain(double v){disp3gain = v;};
    void setDisp1Off(int v){disp1off = v;};
    void setDisp2Off(int v){disp2off = v;};
    void setDisp3Off(int v){disp3off = v;};
	void setOrientation(int v){orientation = v;};
	
    Slice getSliceAxis(){return slice;}
	void setSliceAxis(Slice v){slice = v;}

	void Optimise(CDataStack *ds);
};

#endif
