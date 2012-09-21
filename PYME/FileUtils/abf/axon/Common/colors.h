//****************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//
//****************************************************************************************
// HEADER:  COLORS.H
// PURPOSE: Contains RGB definitions for VGA solid colors.

#ifndef __COLORS_H__
#define __COLORS_H__

#ifndef RGB
#define RGB(r,g,b)          ((DWORD)(((BYTE)(r)|((WORD)(g)<<8))|(((DWORD)(BYTE)(b))<<16)))
#endif

//                                        // MS name
#ifndef RGB_BLACK                         

#define RGB_WHITE      RGB(255,255,255)   // White
#define RGB_VLTGRAY    RGB(224,224,224)
#define RGB_LTGRAY     RGB(192,192,192)   // Silver
#define RGB_DKGRAY     RGB(128,128,128)   // Gray
#define RGB_VDKGRAY    RGB(96,96,96)      
#define RGB_BLACK      RGB(0,0,0)         // Black

#define RGB_VLTYELLOW  RGB(255,255,192)   //
#define RGB_LTYELLOW   RGB(255,255,128)   //
#define RGB_YELLOW     RGB(255,255,0)     // Yellow
#define RGB_ORANGE     RGB(255,128,0)     // ????
#define RGB_DKYELLOW   RGB(128,128,0)     // Olive?

#define RGB_BLUE       RGB(0,0,255)       // Blue
#define RGB_LTBLUE     RGB(0,255,255)     // Cyan
#define RGB_DKBLUE     RGB(0,0,128)       // Navy
#define RGB_BLUEGRAY   RGB(0,128,128)     // Teal

#define RGB_AQUAMARINE RGB(64,128,128)    // 
#define RGB_PURPLE     RGB(64,0,128)      // 

#define RGB_DKRED      RGB(128,0,0)       // Maroon
#define RGB_MAUVE      RGB(128,0,128)     // Purple
#define RGB_RED        RGB(255,0,0)       // Red
#define RGB_PINK       RGB(255,0,128)     // ????
#define RGB_LTPINK     RGB(255,0,255)     // Magenta or Fuschia

#define RGB_GREEN      RGB(0,255,0)       // Lime
#define RGB_DKGREEN    RGB(0,128,0)       // MS-Green

#endif // RGB_BLACK

#endif  /* __COLORS_H__ */
