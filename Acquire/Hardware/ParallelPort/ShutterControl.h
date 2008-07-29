/* ShutterControl.h
Dec 2003
David Baddeley
david.baddeley@kip.uni-heidelberg.de

Class to control shutters attached to PC parallel port - 
Consider it a 'wrapper' class - ie should never need to be 
instantiated - all functions & variables are static.

Uses io.dll to do low level h/w io.

Requires: ShutterControl.cpp, ioDL.cpp, ioDL.h, & io.dll

Will use port 0x378 unless told to do otherwise by the 'setPort'
command - note that for robust usage, the value should be read 
from settings in the registry or otherwise attained and set rather
than relying on the default - but that is not the job of this class.

Usage as per following examples:

// close shutter on ch1
CShutterControl::closeShutters(CShutterControl::CH1);
//open chs 1 & 3
CShutterControl::openShutters(CShutterControl::CH1 | CShutterControl::CH3); 
 //close all shutters except CH2 - leave CH2 as it is
CShutterControl::closeShutters(CShutterControl::ALL ^ CShutterControl::CH2);

etc ...  
*/
#pragma once

class CShutterControl
{
public:
	//Some constants 
	static const short int DEF_PORT = 0x378; //default port
	/*Aliases for the bits of the respective channels
	once we know which laser goes with which shutter, make this 
	a bit more robust by specifiying for ex ... LASER_xxxNM instead of
	CH1 etc ... Or better yet put it in a config file*/
	static const unsigned char CH1 = 1;
	static const unsigned char CH2 = 2;
	static const unsigned char CH3 = 4;
	static const unsigned char CH4 = 8;
	static const unsigned char ALL = 0xFF;

	static void closeShutters(unsigned char shutters);
	static void openShutters(unsigned char shutters);

	static char getShutterStates();
	static void setShutterStates(unsigned char states);
	static bool getShutterState(unsigned char shutter);
	static void setPort(short int port);

	static void init();
private:
	static short int port;

};