/* Base class from which all ***Renderers are derived.
Defines the interface through which the renderer is called
from the view to turn the data into a bitmap - allows multiple
different renderers (eg for pseudocolour) to be easily swapped.*/

//#include "stdafx.h"
#include "BaseRenderer.h"

CBaseRenderer::CBaseRenderer(void)
{
}

CBaseRenderer::~CBaseRenderer(void)
{
}

void CBaseRenderer::setDispOpts(CDisplayParams * dopts)
{
	this->dopts = dopts;
}

void CBaseRenderer::verifyDS(CDataStack *ds)
{
	if (ds ==0) throw "No Data Stack";
	/*for (int i = 0; i < numChans; i++)
	{
		if (chans[i] >= ds->getNumChannels())
			throw "Not enough channels in Data Stack";
	}*/
}
