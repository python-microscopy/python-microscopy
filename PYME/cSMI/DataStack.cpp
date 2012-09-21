 /*Implementaion of a data stack in memory - see datastack.h for
 more info*/
//#include "stdafx.h"

#ifdef __BORLAND__
	#include <vcl.h>
	#include <assert.h>
	#define ASSERT assert
	#include <strstream.h>
#endif

#include "DataStack.h" 
#include "khoros2.h"
#include "excepts.h"

#ifdef __MSVC__
	//#include <new>
	
	#include <strstream>
	
#endif

#define __GCC__ 

#ifdef __GCC__
	#include <assert.h>
	#define ASSERT assert
	#include <strstream>
#endif

using namespace std;

/*The next line is involved in getting a suitable exception
thrown if we can't allocate our memory (not unlikely with large
3D data sets) rather than having the pointers quitely being set to
zero. At the moment no attempt is made to handle the 
exception, so a failure to allocate will crash the program, but
at least this way it easier to write it cleanly when the time comes.*/

//bad_alloc ba;

CDataStack::CDataStack(int width, int height, int depth, int numChannels)
{
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->numChannels = numChannels;

    zPos = 0;
	xPos = 0;
	yPos = 0;

    data = new unsigned short*[numChannels];
    if (data == NULL) throw MemoryAllocError("Cannot allocate DataStack");

    for (int i = 0; i < numChannels; i++)
    {
    	data[i]=new unsigned short[depth*width*height];
    	if (data[i] == NULL) 
		{
			//Tidy up ....
			for (int j = 0; j < i; j++) delete [] data[j];
			delete [] data;
			throw MemoryAllocError("Cannot allocate DataStack");
		}
    };

	channelNames = new string[numChannels];


    for (int i = 0; i < numChannels; i++)
    {
		channelNames[i]=("Channel_");
	    channelNames[i]+=i;
    }
}

CDataStack::CDataStack(LPCSTR filename)
{
	char strHeader[256], strType[256], temp2[256];
	string temp = "";
	FILE *fin;

	fin = fopen(filename, "rb");
	//if(!fin) return (CDataStack*) NULL;
	if(!fin) throw(FileIOError("Could not open file"));

	ReadKhorosHeader(fin, strHeader, strType,
		width, height, depth, numChannels);


	if(width == -1){  // Kein KDF-Format
		//MessageBox(0,"Not KDF Format", "Error",MB_OK);
		fclose(fin);
		throw(FileIOError("Not KDF Format"));
		//return 0;
	}

	temp = strType;

	if(temp != "Unsigned Short"){  // nur Datentyp = "Unsigned Short" zulassen
		//MessageBox(0,"Unrecognised Data Type","Error",MB_OK);
		fclose(fin);
		throw(FileIOError("Unrecognised Data Type - Should be Unsigned Short"));
		//return 0;
	}

    zPos = 0;
	xPos = 0;
	yPos = 0;

    data = new unsigned short*[numChannels];
    if (data == NULL) 
	{
		fclose(fin);
		throw MemoryAllocError("Cannot allocate DataStack");
	}

    for (int i = 0; i < numChannels; i++)
    {
    	data[i]=new unsigned short[depth*width*height];
    	if (data[i] == NULL) 
		{
			//Tidy up ....
			for (int j = 0; j < i; j++) delete [] data[j];
			delete [] data;
			fclose(fin);
			throw MemoryAllocError("Cannot allocate DataStack");
		}
    };

	channelNames = new string[numChannels];


	istrstream tok(strHeader);
	tok >> temp2; // get the "Channels:

	for (int i=0;i < numChannels;i++)
	{
 		int t = fread(data[i], sizeof(unsigned short), (width*height*depth),fin);
		if (t != (width*height*depth))
		{
			//Tidy up ....
			for (int j = 0; j < numChannels; j++) delete [] data[j];
			delete [] data;
			fclose(fin);
			throw(FileIOError("Premature EOF"));
		}

		tok >> temp2;
 		channelNames[i] = temp2;
	}

	fclose(fin);
}

CDataStack::CDataStack(CDataStack &ds)
{
	this->width = ds.getWidth();
	this->height = ds.getHeight();
	this->depth = ds.getDepth();
	this->numChannels = ds.getNumChannels();

    zPos = 0;
	xPos = 0;
	yPos = 0;

    data = new unsigned short*[numChannels];
    if (data == NULL) throw MemoryAllocError("Cannot allocate DataStack");

    for (int i = 0; i < numChannels; i++)
    {
    	data[i]=new unsigned short[depth*width*height];
    	if (data[i] == NULL) 
		{
			//Tidy up ....
			for (int j = 0; j < i; j++) delete [] data[j];
			delete [] data;
			throw MemoryAllocError("Cannot allocate DataStack");
		}
		memcpy(data[i],ds.data[i],depth*width*height*sizeof(unsigned short));
    };

	channelNames = new string[numChannels];


    for (int i = 0; i < numChannels; i++)
    {
		channelNames[i]=ds.getChannelName(i);
    }
}

CDataStack::CDataStack(CDataStack &ds,int x1,int y1,int z1,int x2,int y2,int z2, int nchs, int *chans)
{
	//Create a new data stack by cropping an existing one
	
	if (x1 < 0 || x1 >= ds.getWidth()) throw IndexOutOfBounds("x1");
	if (x2 < 0 || x2 >= ds.getWidth()) throw IndexOutOfBounds("x2"); 
	if (y1 < 0 || y1 >= ds.getHeight()) throw IndexOutOfBounds("y1");
	if (y2 < 0 || y2 >= ds.getHeight()) throw IndexOutOfBounds("y2"); 
	if (z1 < 0 || z1 >= ds.getDepth()) throw IndexOutOfBounds("z1");
	if (z2 < 0 || z2 >= ds.getDepth()) throw IndexOutOfBounds("z2");

	if (nchs < 1) throw IndexOutOfBounds("nchs");

	for (int i = 0; i < nchs; i++) if (chans[i] >= ds.getNumChannels()) throw IndexOutOfBounds("chans");

	this->width = abs(x1 - x2);
	this->height = abs(y1 - y2);
	this->depth = abs(z1 - z2);
	this->numChannels = nchs;

    zPos = 0;
	xPos = 0;
	yPos = 0;

    data = new unsigned short*[numChannels];
    if (data == NULL) throw MemoryAllocError("Cannot allocate DataStack");

    for (int i = 0; i < numChannels; i++)
    {
    	data[i]=new unsigned short[depth*width*height];
    	if (data[i] == NULL) 
		{
			//Tidy up ....
			for (int j = 0; j < i; j++) delete [] data[j];
			delete [] data;
			throw MemoryAllocError("Cannot allocate DataStack");
		}
		
		int j = 0;

		for (int z = min(z1,z2);z < max(z1,z2); z++)
		{
			for (int y = min(y1,y2);y < max(y1,y2); y++)
			{
				for (int x = min(x1,x2);x < max(x1,x2); x++)
				{
					data[i][j] = ds.data[chans[i]][z*ds.width*ds.height + y*ds.width + x];
					j++;
				}
			}
		}
    };

	channelNames = new string[numChannels];


    for (int i = 0; i < numChannels; i++)
    {
		channelNames[i]=ds.getChannelName(chans[i]);
    }
}

    
CDataStack::~CDataStack(void)
{
    for(int i = 0; i < numChannels; i++)
    	delete [] data[i];
    delete [] data;

    delete [] channelNames;
}

int CDataStack::getWidth(void)
{
    return width;
}

int CDataStack::getHeight(void)
{
    return height;
}

int CDataStack::getDepth(void)
{
    return depth;
}

int CDataStack::getNumChannels(void)
{
    return numChannels;
}

unsigned short * CDataStack::getChannelSlice(int chan, int zpos)
{
    if ((chan < numChannels) && (zpos < depth) && (chan >= 0) && (zpos >= 0))
    	return data[chan]+(width*height*zpos);
    else
    	throw IndexOutOfBounds("");
    
}

unsigned short * CDataStack::getChannel(int chan)
{
    if ((chan < numChannels) && (chan >=0))
    	return data[chan];
    else
    	throw IndexOutOfBounds("");
}

int CDataStack::getZPos(void)
{
    return zPos;
}
    
    void CDataStack::setZPos(int pos)
    {
    	if (pos < 0) pos = 0;
    	if (pos >= depth) pos = depth -1;
	 	ASSERT((pos >=0) && (pos < depth));
    	zPos = pos;
    }
    
void CDataStack::setXPos(int pos)
{
 	if (pos < 0) pos = 0;
 	if (pos >= width) pos = width -1;
 	ASSERT((pos >=0) && (pos < width));
 	xPos = pos;
}

void CDataStack::setYPos(int pos)
{
 	if (pos < 0) pos = 0;
 	if (pos >= height) pos = height -1;
 	ASSERT((pos >=0) && (pos < height));
  	yPos = pos;
}
  
bool CDataStack::nextZ(void)
{
    if (zPos < (depth-1))
    {
    	zPos ++;
    	return true;
    }
    else return false;
}

unsigned short * CDataStack::getCurrentChannelSlice(int chan)
{
    return getChannelSlice(chan,zPos);
}

int CDataStack::numberOfAvailableSlices(int iWidth, int iHeight, int iChannels)
{
    
    return 10000;
}
    
//CString CDataStack::getChannelName(int chan)
string CDataStack::getChannelName(int chan)
{
 	ASSERT(chan >=0 && chan < numChannels);
    return channelNames[chan];
}
    
void CDataStack::setChannelName(int chan, LPCSTR name)
{
 	ASSERT(chan >=0 && chan < numChannels);
    channelNames[chan]=name;
}
    
CDataStack*  CDataStack::OpenFromFile(LPCSTR filename)
{
	int     iPicWidth, iPicHeight, iNumber;
	int     iChannels;
	int     iColors = 0;
	int	  curTokPos = 0;
	char strHeader[256], strType[256], temp2[256];
	string temp = "";
	FILE *fin;
	CDataStack *ds;
	//ifstream fin(filename,os_base::binary);

	//bResult = OpenDataFile(ModifyFileName(m_strFileName, 0, 0, 0), FALSE);
	fin = fopen(filename, "rb");
	if(!fin) return (CDataStack*) NULL;

	ReadKhorosHeader(fin, strHeader, strType,
		iPicWidth, iPicHeight, iNumber, iChannels);


	if(iPicWidth == -1){  // Kein KDF-Format
		//AfxMessageBox(strKDFError);
		//MessageBox(0,"Not KDF Format", "Error",MB_OK);
		fclose(fin);
		return 0;
	}//endif

	temp = strType;

	if(temp != "Unsigned Short"){  // nur Datentyp = "Unsigned Short" zulassen
		//MessageBox(0,"Unrecognised Data Type","Error",MB_OK);
		fclose(fin);
		return 0;
	}


	ds = new CDataStack(iPicWidth, iPicHeight, iNumber, iChannels);

	istrstream tok(strHeader);
	tok >> temp2; // get the "Channels:


	//try{
	for (int i=0;i < iChannels;i++)
	{
 		int t = fread(ds->getChannel(i), sizeof(unsigned short), (iPicWidth*iPicHeight*iNumber),fin);
					tok >> temp2;
 		ds->setChannelName(i, temp2);
	}

		fclose(fin);
	//}catch()/*(CFileException* pFileError)*/{  // Dateifehler aufgetreten
		/*AfxMessageBox(m_strMessage[8]);
		pFileError->Delete();*/
		//fclose(fin);
		//delete ds;
		//return 0;
	//}//endcatch

	return ds;
}

bool CDataStack::SaveToFile(LPCSTR filename)
{
  int     iColors = 0;
  string strHeader;
  bool    bResult;

  FILE *fout;

  //bResult = OpenDataFile(ModifyFileName(m_strFileName, 0, 0, 0), TRUE);
  //if(bResult == FALSE) return FALSE;

  fout = fopen(filename, "wb");
  if(!fout) return false;

  strHeader = "Channels:";

  for (int i = 0; i < getNumChannels(); i++)
	  strHeader += " " + getChannelName(i);



  WriteKhorosHeader(fout, strHeader.c_str(), "Unsigned Short",
        getWidth(), getHeight(),getDepth(), getNumChannels());

  
	for (int i=0;i < getNumChannels();i++)
	{
		int t = fwrite(getChannel(i),sizeof(unsigned short),getWidth()*getHeight()*getDepth(),fout);
		if (t != (getWidth()*getHeight()*getDepth())) bResult = false;
	}


  fclose(fout);

  return bResult;
}

unsigned short CDataStack::getValue(int x, int y, int z, int ch)
{
	if (x < 0 || x >= width) throw IndexOutOfBounds("x");
	if (y < 0 || y >= height) throw IndexOutOfBounds("y");
	if (z < 0 || z >= depth) throw IndexOutOfBounds("z");
	if (ch < 0 || ch >= numChannels) throw IndexOutOfBounds("ch");
	
	return data[ch][z*width*height + y*width + x];
}
 
