/* ShutterControl.c
Dec 2003
David Baddeley
david.baddeley@kip.uni-heidelberg.de

See ShutterControl.h for description
*/

#include "stdafx.h"
#include "ioDL.h"
#include "ShutterControl.h"

short int CShutterControl::port = CShutterControl::DEF_PORT;

void CShutterControl::setShutterStates(unsigned char states)
{
	PortOut(port, states);
}

char CShutterControl::getShutterStates()
{
	return PortIn(port);
}

bool CShutterControl::getShutterState(unsigned char shutter)
{
	return (bool)(PortIn(port) & shutter);
}

void CShutterControl::openShutters(unsigned char shutters)
{
	unsigned char t;
	t = PortIn(port);
	PortOut(port, t|shutters);
}

void CShutterControl::closeShutters(unsigned char shutters)
{
	unsigned char t;
	t = PortIn(port);
	PortOut(port, t& (!shutters));
}

void CShutterControl::setPort(short int p)
{
	port = p;
}

void CShutterControl::init()
{
	if(LoadIODLL() != 0)
	{
		MessageBox(0,"Could not load IODLL", "ERROR", MB_OK);
		throw "Could not load IODLL";
	}
}