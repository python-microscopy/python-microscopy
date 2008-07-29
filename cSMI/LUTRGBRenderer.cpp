/* Renders the desired 3 channels as an RGB composite image. 
Uses a lookup table for speed, which is recalculated iff
the gains or offsets have changed since the last call to the
Render method*/ 

//#include "stdafx.h"
#include "LUTRGBRenderer.h"
#include <math.h>
    
CLUT_RGBRenderer::CLUT_RGBRenderer(void)
{
    rGainOld = -1;
    rOffOld = -1;
    gGainOld = -1;
    gOffOld = -1;
    bGainOld = -1;
    bOffOld = -1;
}

CLUT_RGBRenderer::~CLUT_RGBRenderer(void)
{
}
    
void CLUT_RGBRenderer::Render(byte * bmp, CDataStack * ds, int zpos)
{
    //int nc = numChans;
    
    //verifyDS(ds);
    
	//Check that there is a data stack to read from
	//If it's not there, save us the embarassment of accessing a null pointer
    if (ds !=0)
    {
    	//if (ds->getNumChannels() < nc) nc = ds->getNumChannels();
    
		//pointers for the data ascociated with the various channels
    	unsigned short *rdat=0;
    	unsigned short *gdat=0;
    	unsigned short *bdat=0;
    
		//Declaration of gains & offsets
    	double rGain=-1, gGain=-1, bGain=-1;
    	int rOff=-1, gOff=-1, bOff=-1;
    
		switch (dopts->getSliceAxis())
        {
			//Normal XY display    
			case (CDisplayParams::SLICE_XY):
            {
				if (zpos == -1)//we haven't been told where abouts in the stack we should plot from - so use current position
				{
    			    //Check to see if there is data for the Red display channel, if present, fill
					//the relevant fields.
					if ((dopts->getDisp1Chan() != -1) && (dopts->getDisp1Chan() < ds->getNumChannels()))
    			    {
    				    rdat = ds->getCurrentChannelSlice(dopts->getDisp1Chan());
    				    rGain = dopts->getDisp1Gain();
    				    rOff = dopts->getDisp1Off();
					}else rdat = 0; //if no data, set the pointer to 0;
					//likewise for Green
    			    if ((dopts->getDisp2Chan() != -1) && (dopts->getDisp2Chan() < ds->getNumChannels()))
    			    {
    				    gdat = ds->getCurrentChannelSlice(dopts->getDisp2Chan());
    				    gGain = dopts->getDisp2Gain();
    				    gOff = dopts->getDisp2Off();
    			    }else gdat = 0;
					//and for Blue
    			    if ((dopts->getDisp3Chan() != -1) && (dopts->getDisp3Chan() < ds->getNumChannels()))
    			    {
    				    bdat = ds->getCurrentChannelSlice(dopts->getDisp3Chan());
    				    bGain = dopts->getDisp3Gain();
    				    bOff = dopts->getDisp3Off();
    			    }else bdat = 0;

    		    }else //We should plot the slice we've been told to
				{
    			    if (dopts->getDisp1Chan() != -1)
    			    {
    				    rdat = ds->getChannelSlice(dopts->getDisp1Chan(), zpos);
    				    rGain = dopts->getDisp1Gain();
    				    rOff = dopts->getDisp1Off();
    			    } else rdat = 0;
    			    if (dopts->getDisp2Chan() != -1)
    			    {
    				    gdat = ds->getChannelSlice(dopts->getDisp2Chan(), zpos);
    				    gGain = dopts->getDisp2Gain();
    				    gOff = dopts->getDisp2Off();
    			    }else gdat = 0;
    			    if (dopts->getDisp3Chan() != -1)
    			    {
    				    bdat = ds->getChannelSlice(dopts->getDisp3Chan(), zpos);
    				    bGain = dopts->getDisp3Gain();
    				    bOff = dopts->getDisp3Off();
    			    }else bdat = 0;
    		    }
    
    		    //If the gains etc have changed -> regenerate the LUTs

    		    if ((rGain != rGainOld) || (rOff != rOffOld)) GenerateLUT(redLUT, rGain, rOff);
    		    if ((gGain != gGainOld) || (gOff != gOffOld)) GenerateLUT(greenLUT, gGain, gOff);
    		    if ((bGain != bGainOld) || (bOff != bOffOld)) GenerateLUT(blueLUT, bGain, bOff);
	 		
				//Support for rotating the display
				int istep,imax,jstep,jmax;

				switch(dopts->getOrientation())
				{
				case CDisplayParams::UPRIGHT:
					istep = ds->getWidth();
					imax = ds->getWidth()*ds->getHeight();
					jstep = 1;
					jmax = ds->getWidth();
					break;
				case CDisplayParams::ROT90:
					jstep = ds->getWidth();
					jmax = ds->getWidth()*ds->getHeight();
					istep = 1;
					imax = ds->getWidth();
					break;
				}

				//loop to translate data from the data stack into the bitmap
 				//along with relevant scaling & offsets (by indexing into the relevant LUT)
				for(int i = 0; i < imax;i+=istep)
				{
					for(int j = 0; j < jmax;j+=jstep)
					{
							//Blue
    					if (bdat != 0)
    					{
 							*bmp =  blueLUT[bdat[i+j] & 4095];
            			} else *bmp = 0;
    					bmp++;

    					//Green
    					if (gdat != 0)
            			{
							*bmp =  greenLUT[gdat[i+j]& 4095];
    					} else *bmp = 0;
    					bmp++;

            			//Red
            			if (rdat != 0)
    					{
							*bmp =  redLUT[rdat[i+j]& 4095];
						} else *bmp = 0;
    					bmp++;
    				}

						
				}
				break;
			}

		
			//XZ - Display, At present only supports the current slice, and doesn't support rotation
			case (CDisplayParams::SLICE_XZ):
			{
				if (dopts->getDisp1Chan() != -1)
				{
  		       		rdat = ds->getChannel(dopts->getDisp1Chan());
  		      		rGain = dopts->getDisp1Gain();
  		     		rOff = dopts->getDisp1Off();
					rdat += ds->getYPos()*ds->getWidth();
  				} else rdat = 0;

  	       		if (dopts->getDisp2Chan() != -1)
  	      		{
  	     			gdat = ds->getChannel(dopts->getDisp2Chan());
  	    			gGain = dopts->getDisp2Gain();
  	   				gOff = dopts->getDisp2Off();
					gdat += ds->getYPos()*ds->getWidth();
  	  			}else gdat = 0;

  	 			if (dopts->getDisp3Chan() != -1)
  				{
         			bdat = ds->getChannel(dopts->getDisp3Chan());
        			bGain = dopts->getDisp3Gain();
       				bOff = dopts->getDisp3Off();
					bdat += ds->getYPos()*ds->getWidth();
      			}else bdat = 0;
	  
	            
				//If the gains etc have changed -> regenerate the LUTs
				if ((rGain != rGainOld) || (rOff != rOffOld)) GenerateLUT(redLUT, rGain, rOff);
  				if ((gGain != gGainOld) || (gOff != gOffOld)) GenerateLUT(greenLUT, gGain, gOff);
  				if ((bGain != bGainOld) || (bOff != bOffOld)) GenerateLUT(blueLUT, bGain, bOff);
  
                for(int i = 0; i < ds->getDepth();i++)
		        {
                    for (int j = 0; j < ds->getWidth();j++)
                    {
    			        //Blue
    			        if (bdat != 0)
    			        {
		        		    *bmp =  blueLUT[*bdat & 4095];
	        			    bdat++;
            			} else *bmp = 0;
    			        bmp++;

    		        	//Green
    	        		if (gdat != 0)
            			{
				            *bmp =  greenLUT[*gdat& 4095];
        		        	gdat++;
    	                } else *bmp = 0;
    	        	    bmp++;

            			//Red
            			if (rdat != 0)
    	        	    {
 		                	*bmp =  redLUT[*rdat& 4095];
 	        		        rdat++;
                    	} else *bmp = 0;
    	        		bmp++;
    		        }
		
                    if (bdat !=0) bdat += (ds->getHeight() - 1)*ds->getWidth();
                    if (gdat !=0) gdat += (ds->getHeight() - 1)*ds->getWidth();
                    if (rdat !=0) rdat += (ds->getHeight() - 1)*ds->getWidth();
                }
				break;
            }

			//YZ Display - limitations as for XZ
            case (CDisplayParams::SLICE_YZ):
            {
                if (dopts->getDisp1Chan() != -1)
                {
		       		rdat = ds->getChannel(dopts->getDisp1Chan());
		      		rGain = dopts->getDisp1Gain();
		     		rOff = dopts->getDisp1Off();
                    rdat += ds->getXPos();
		        } else rdat = 0;

 	       	    if (dopts->getDisp2Chan() != -1)
 	      	    {
	     		    gdat = ds->getChannel(dopts->getDisp2Chan());
	    		    gGain = dopts->getDisp2Gain();
	   				gOff = dopts->getDisp2Off();
                    gdat += ds->getXPos();
 	  	        }else gdat = 0;

	 	        if (dopts->getDisp3Chan() != -1)
		        {
       			    bdat = ds->getChannel(dopts->getDisp3Chan());
       			    bGain = dopts->getDisp3Gain();
     			    bOff = dopts->getDisp3Off();
                    bdat += ds->getXPos();
      		    }else bdat = 0;
  
                if ((rGain != rGainOld) || (rOff != rOffOld)) GenerateLUT(redLUT, rGain, rOff);
  		        if ((gGain != gGainOld) || (gOff != gOffOld)) GenerateLUT(greenLUT, gGain, gOff);
  		        if ((bGain != bGainOld) || (bOff != bOffOld)) GenerateLUT(blueLUT, bGain, bOff);
  
                for(int i = 0; i < ds->getHeight();i++)
  		        {
                    for (int j = 0; j < ds->getDepth();j++)
                    {
  			            //Blue
  			            if (bdat != 0)
  			            {
  		        		    *bmp =  blueLUT[*bdat & 4095];
  	        			    bdat+=ds->getWidth();
          			    } else *bmp = 0;
  			            bmp++;

  		        	    //Green
  	        		    if (gdat != 0)
          			    {
  				            *bmp =  greenLUT[*gdat& 4095];
          		        	gdat+=ds->getWidth();
  	                	} else *bmp = 0;
  	        	        bmp++;

          			    //Red
          			    if (rdat != 0)
  	        	        {
  		                	*bmp =  redLUT[*rdat& 4095];
  	        		        rdat+=ds->getWidth();
                  		} else *bmp = 0;
  	        		    bmp++;
  		            }                                  
                }
                break;
            }
        };
  
 		//save a copy of the gains so we can compare them next time
		// to see if we need a new LUT

    	rGainOld = rGain;
    	rOffOld = rOff;

    	gGainOld = gGain;
    	gOffOld = gOff;

    	bGainOld = bGain;
    	bOffOld = bOff;
    }
}
    
//Make a lookup table, put it in LUT
void CLUT_RGBRenderer::GenerateLUT(unsigned char *LUT, float Gain, float Off)
{
    for (int i = 0; i < 4096;i++)
    {
    	LUT[i] =  (unsigned char)max(min(((double)(i - Off))*(Gain), (double)0xFF),(double)0);
    }
}
