/* Base class from which all ***Renderers are derived.
Defines the interface through which the renderer is called
from the view to turn the data into a bitmap - allows multiple
different renderers (eg for pseudocolour) to be easily swapped.

The most important function is 'Render(...)'*/
#define byte unsigned char

#include "DataStack.h"
#include "DisplayParams.h"

#pragma once

class CBaseRenderer
{
protected:
	CDisplayParams * dopts;
public:
	CBaseRenderer(void);
	~CBaseRenderer(void);
	virtual void Render(byte * bmp, CDataStack * ds, int zpos=-1)=0;
	void setDispOpts(CDisplayParams * dopts);
protected:
	void verifyDS(CDataStack *ds);
};
