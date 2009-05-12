/* CDataStack - Internal representation of a stack of images 
with arbitrarily many named colour channels.

Representation is closely analogous to that in a KDF data file, 
to facilitate easy loading & saving.
*/

#pragma once

#ifndef datastackH
#define datastackH
//#include <stdio.h>
//#include <fstream.h>
#include <string>
#include <stdlibq.h>
//#include <vcl/dstring.h>


#ifdef __PYWIN__
	#define ASSERT //
	#include <windows.h>
#endif

#define LPCSTR char*

#ifdef __MSVC__
	
#endif

using namespace std; 

class CDataStack
{
protected:
	int width, height, depth, numChannels;
	int zPos,xPos,yPos; //Currently active slice in the stack
    // Will point to an array of single channel data stacks
 	// IE each channel is stored in a contiguous block of size
 	// width*height*depth*sizeof(unsigned short)
    	unsigned short **data;
 	//CString *channelNames;
 	string *channelNames;

public:
 	//Create a new data stack
   	CDataStack(int width, int height, int depth, int numChannels);
	CDataStack(LPCSTR filename);
	CDataStack(CDataStack &ds); // Copy consructor
	CDataStack(CDataStack &ds,int x1,int y1,int z1,int x2,int y2,int z2, int nchs, int *chans); //create a new ds by cropping an existing one
   	~CDataStack(void);
 	//get the relavent dimensions etc ...
   	int getWidth(void);
   	int getHeight(void);
   	int getDepth(void);
   	int getNumChannels(void);
   	int getZPos(void);
   	void setZPos(int pos);
	//Increment the position in the stack - return false & do nothing 
	//if already at the end
	int getXPos(void){return xPos;}
 	void setXPos(int pos);
	int getYPos(void){return yPos;}
	void setYPos(int pos);
   	bool nextZ(void);

	unsigned short getValue(int x, int y, int z, int ch);

	//Return a pointer to the area of memory occupied by the requested 
	//slice from the requested channel
   	unsigned short * getChannelSlice(int chan, int zpos);
	//Same as above, but for the currently active slice
	unsigned short * getCurrentChannelSlice(int chan);
 	//Get a pointer to begining of the memory block corresponding
 	//to the specified channel
   	unsigned short * getChannel(int chan);
 	//Static member to use before allocating a new stack to check how
	//many channels we can allocate and still fit in the available memory.
	static int numberOfAvailableSlices(int iWidth, int iHeight, int iChannels);
	
	//Fairly self explanatory - should probably change the CString (MSFC) to an STL string
	//to improve portability
	//CString getChannelName(int chan);
	string getChannelName(int chan);
   	void setChannelName(int chan, LPCSTR name);
    
    static CDataStack* OpenFromFile(LPCSTR filename);
    bool SaveToFile(LPCSTR filename);
    };

#endif
