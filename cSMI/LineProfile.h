#pragma once
#include "DataStack.h"

class LineProfile
{
public:
	enum direction{X,Y,Z};
private:
	CDataStack * ds;
	direction dir;
	int chan, roi, sback;
public:
	LineProfile(CDataStack *ds, int chan, direction dir = Z, int roi = 0, int back = 0);
	~LineProfile(void);

	int length();
	const double operator[](int ind);
};
