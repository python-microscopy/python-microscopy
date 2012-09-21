#include "LineProfile.h"
#include "excepts.h"

LineProfile::LineProfile(CDataStack *ds, int chan, direction dir, int roi, int back)
{
	this->ds = ds;
	this->chan = chan;
	this->dir = dir;
	this->roi = roi;
	this->sback = back;
}

LineProfile::~LineProfile(void)
{
}

int LineProfile::length()
{
	switch(dir)
	{
	case X:
		return ds->getWidth();
		break;
	case Y:
		return ds->getHeight();
		break;
	case Z:
		return ds->getDepth();
		break;
	default:
		return 0;
	}
}

const double LineProfile::operator[](int ind)
{
	if ((ind < 0) || (ind >= length())) throw IndexOutOfBounds(""); 
	
	unsigned short * data = ds->getChannel(chan);

	//int length;
    int d1pos;
    int d1mult;
    int d2pos;
    int d2mult;
    int d2max;
    int d3pos;
    int d3mult;
    int d3max;


    double sig;
    double csig;
    double back;
    double cback;
	
	switch (dir)
    {
    case X:
        //length = ds->getWidth();
        d1pos = ds->getXPos();
        d1mult = 1;

        d2pos = ds->getYPos();
        d2mult = ds->getWidth();
        d2max = ds->getHeight();

        d3pos = ds->getZPos();
        d3mult = ds->getWidth()*ds->getHeight();
        d3max = ds->getDepth();
        break;

    case Y:
        d2max = ds->getWidth();
        d2pos = ds->getXPos();
        d2mult = 1;

        d1pos = ds->getYPos();
        d1mult = ds->getWidth();
        //length = ds->getHeight();

        d3pos = ds->getZPos();
        d3mult = ds->getWidth()*ds->getHeight();
        d3max = ds->getDepth();
        break;

    case Z:
        d2max = ds->getWidth();
        d2pos = ds->getXPos();
        d2mult = 1;

        d3pos = ds->getYPos();
        d3mult = ds->getWidth();
        d3max = ds->getHeight();

        d1pos = ds->getZPos();
        d1mult = ds->getWidth()*ds->getHeight();
        //length = ds->getDepth();
        break;
    }
	
	sig=0;
    csig=0;
    back=0;
    cback=0;

    for (int j = -roi; j <= roi; j++)
            for (int k = -roi; k <= roi; k++)
            {
                    if (((d2pos+j) < d2max) && ((d2pos+j) >=0) && ((k+d3pos) < d3max) && ((k+d3pos) >=0))
                    {
                            sig += *(data + d1mult*ind + d2mult*(d2pos+j) + d3mult*(k+d3pos));
                            csig ++;
                    }
            };


    sig /= csig;

    if (sback > 0)
    {
            for (int j = -(sback + roi); j <= roi; j++)
                    for (int k = -(sback + roi); k <= (sback + roi); k++)
                    {
                            if (((d2pos+j) < d2max) && ((d2pos+j) >=0) && ((k+d3pos) < d3max) && ((k+d3pos) >=0))
                            {
                                    back += *(data + d1mult*ind + d2mult*(d2pos+j) + d3mult*(k+d3pos));
                                    cback ++;
                            }
                    };

            back /= cback;
            sig -= back;
    }

	return sig;
}
