typedef unsigned char  byte;     /* 8-bit  */
typedef unsigned short word;     /* 16-bit */
typedef unsigned long  dword;    /* 32-bit */

#ifdef __cplusplus
extern "C" {            //  Assume C declarations for C++
#endif  //C++

//camop functions S95_CAM.dll
int WINAPI SET_BOARD(int boardnr);
//set the current board to work with
//after selecting a board the first time, call SET_INIT(1..2) to initialize it
//boardnr: 0..9 selects PCI-Board set

int WINAPI GET_BOARD(int *boardnr);
//get the current board to work with


int WINAPI SET_INIT(int mode);
//resets or initialize the PCI-Board, the camera and all global values of the dll
//mode=0 : reset
//         frees memory
//         close dialog boxes
//         SET_INIT(0) should be called before you close your program

//mode=1 : init default
//mode=2 : init with values of registry (HKEY_CURRENT_USER\\software\\PCO\\Camera Settings

int WINAPI GET_STATUS(int *camtype, int *eletemp, int *ccdtemp);
//reads out the connected camera LONG or FAST
//and the electronic and CCD temperature in °C

int WINAPI GET_IMAGE_SIZE(int *width, int *height);
//reads out the actual size of the picture, which is send from the camera

int WINAPI GET_IMAGE_STATUS(int *status);
//reads out the status of the PCI-Buffer process
// *status Bit0 = 0 : no read image process is running
//                1 : read image process is running
//         Bit1 = 0 : 1 or 2 pictures are in PCI-Buffer
//                1 : no picture is in PCI-BUFFER
//         Bit2 = 0 : camera is idle, no exposure is running
//                1 : camera is busy, exposure is running or a picture
//                    is send from the camera to the PCI-Buffer

int WINAPI SET_COC( int mode,    int trig,
                    int roixmin, int roixmax,
                    int roiymin, int roiymax,
                    int hbin,    int vbin,
                    char *timevalues);
// build an COC and load it into the camera
// mode = (typ&0xFFFF)+subtyp<<16

// typ=0    : set LONG EXPOSURE camera (if camtyp=FAST return error WRONGVAL)
// subtyp=0 : sequential mode
// subtyp=1 : simultaneous mode
// subtyp=2 : simultaneous mode
// subtyp=3 : simultaneous mode

// typ=1    : set FAST SHUTTER camera  (if camtyp=LONG return error WRONGVAL)
// subtyp=0 : standard fast

// subtyp=1 : double fast, only for doubleshutter cameras, else returns WRONGVAL
// subtyp=2 : double long, only for doubleshutter cameras, else returns WRONGVAL

// mode=2 : set DOUBLE SHUTTER camera (if camtyp=LONG return error WRONGVAL)
// mode=3 : set DOUBLE SHUTTER LONG (if camtyp=LONG return error WRONGVAL)

// trig=0 : continuos software triggering
// trig=1 : external trigger raising edge
// trig=2 : external trigger falling edge

// roi... : values for area of interest, 32x32pixel quadrants
//    x   : range 1 to 40 (20)
//    y   : range 1 to 32 (15)

// hbin   : horizontal binning (1,2,4,8)
// vbin   : vertical binning (1,2,4,8,16,32)

// timevalues : Null terminated ASCII string
//              delay0,exposure0,delay0,exposur1 ... -1,-1
//              The pair -1,-1 must be last of the table
//              The LONG camera expects only one pair of timevalues
//              DOUBLE and DOUBLE LONG have no timevalues
//              FAST can have up to 100 pairs of timevalues
// changing the values of roi..., hbin and vbin changes also the size
// of the picture, which is send from the camera

int WINAPI GET_SETTINGS(int *mode,int *trig,
                        int *roixmin, int *roixmax,
                        int *roiymin, int *roiymax,
                        int *hbin, int *vbin,
                        char *(*tab));
// get values, which are in use now

int WINAPI LOAD_USER_COC(word *codetable);
// load or own COC = codetable into the camera (see LAYER2 commands description)
// none of the SET_COC parameters is in use,
// GET_IMAGE_SIZE reads out wrong values

int WINAPI RUN_COC(int mode);
// starts camera grabing pictures
// mode=0 continuos runninng
//        don't use other values

int WINAPI STOP_COC(int mode);
// stops camera grabing pictures
// mode=0 stops and make PCI-BUFFER empty
//        don't use other values

int WINAPI LOAD_OUTPUT_LUT(byte *lut);
// loads the convert LUT for black&white pictures
// converting is actual done only by software
// in the new PCI-BUFFER Board also by hardware
// lut : table with 4096 8-Bit values

int WINAPI CONVERT_BUFFER_12TO8(int mode, int width,int height,
                                word *b12, byte *b8);
// converts an 12Bit image to an 8Bit image using LUT
// mode=0 : normal converting TOP 12Bit = TOP 8Bit
// mode=1 : flips image  TOP 12Bit = BOTTOM 8Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// b12    : pointer to PC memory area, from which to read the 12BIT image
// b8     : pointer to PC memory area, to write the 8Bit image in

int WINAPI READ_IMAGE_8BIT(int mode,int width,int height, byte *b8);
// transfers and converts one picture from PCI-Buffer to PC memory area
// mode=0 : normal transfer and converting TOP camera = TOP 8Bit
// mode=1 : flips image  TOP camera = BOTTOM 8Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// b8     : pointer to PC memory area, to write the 8Bit image in

int WINAPI READ_IMAGE_12BIT(int mode,int width,int height, word *b12);
// transfers and converts one picture from PCI-Buffer to PC memory area
// mode=0 : normal transferand converting TOP camera = TOP 12Bit
// mode=1 : flips image  TOP camera = BOTTOM 12Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// b12    : pointer to PC memory area, to write the 12Bit image in

int WINAPI READ_IMAGE_12BIT_EXT(int mode,int width,int height,int x, word *b12);
// transfers and converts one picture from PCI-Buffer to PC memory area
// mode=0 : normal transferand converting TOP camera = TOP 12Bit
// mode=1 : flips image  TOP camera = BOTTOM 12Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// x      : image number within the PC memory area, start with 0!
// b12    : pointer to begin of PC memory area, to write the 12Bit image in




// Color camera functions
int WINAPI LOAD_COLOR_LUT(byte *redlut,byte *greenlut,byte *bluelut);
// loads the convert COLOR LUT's for color pictures
// converting is always done by software
// redlut : table with 4096 8-Bit values for red color
// greenlut : table with 4096 8-Bit values for green color
// bluelut : table with 4096 8-Bit values for blue color

int WINAPI LOAD_PSEUDO_COLOR_LUT(byte *redlut,byte *greenlut,byte *bluelut);
// loads the convert PSEUDO-COLOR LUT's for bw pictures
// converting is always done by software
// redlut : table with 256 8-Bit values for red color
// greenlut : table with 256 8-Bit values for green color
// bluelut : table with 256 8-Bit values for blue color

int WINAPI CONVERT_BUFFER_12TOCOL(int mode, int width, int height,
                               word *b12, byte *gb8);
// converts an 12Bit color image to an 32Bit (24Bit) image using COLOR LUT's
// mode=0 : normal converting TOP 12Bit = TOP 24Bit
// mode=1 : flips image  TOP 12Bit = BOTTOM 24Bit  (BMP-Format)
// mode=2 : normal converting TOP 12Bit = TOP 32Bit
// mode=3 : flips image  TOP 12Bit = BOTTOM 32Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// b12    : pointer to PC memory area, from which to read the 12BIT image
// b8     : pointer to PC memory area, to write the 8Bit image in

int WINAPI READ_IMAGE_COL(int mode,int width,int height, byte *b8);
// transfers and converts one picture from PCI-Buffer to PC memory area
// mode=0 : normal converting TOP 12Bit = TOP 24Bit
// mode=1 : flips image  TOP 12Bit = BOTTOM 24Bit  (BMP-Format)
// mode=2 : normal converting TOP 12Bit = TOP 32Bit
// mode=3 : flips image  TOP 12Bit = BOTTOM 32Bit  (BMP-Format)
//          don't use other values
// width  : number of pixel in each line in the image
// height : number of lines in the image
// b8     : pointer to PC memory area, to write the 8Bit image in

// Dialog functions:
// every Dialog store his exit values in the registry and load's it when starting

int WINAPI OPEN_DIALOG_CAM(HWND hWnd, int mode, char *title);
// create an thread, which controls an dialog box, where you can input
// all values for SET_COC. If one value is changed SET_COC is send.
// hWnd   : your main window handle
// mode=0 : dialogbox, no message is send to hWnd
// mode=1 : dialogbox, message is send to hWnd
//long exposure Sensicam
// mode=2 : short dialogbox, only time values, no message 
// mode=3 : short dialogbox, only time values, message 
// mode=4 : great dialogbox, does not return before 'ok', no message 
// mode=5 : great dialogbox, does not return before 'ok', message 
//          message=(PostMessage(hwnd,WM_COMMAND,updmsg,0);
//          (updmsg is IDC_UPDATE or read out from the registry)
// title  : Null terminated ASCII string
//          title of the window
//          if NULL default title is used

int WINAPI LOCK_DIALOG_CAM(int mode);
// inhibits user input to dialogbox
// mode=0 : user can change values
// mode=1 : user cannot change values

int WINAPI CLOSE_DIALOG_CAM(void);
// closes the dialog box

int WINAPI SET_DIALOG_CAM(int mode, int trig,
                        int roixmin, int roixmax,
                        int roiymin, int roiymax,
                        int hbin, int vbin,
                        char *timevalues);
// set values in the dialogbox

int WINAPI GET_DIALOG_CAM(int *mode,int *trig,
                          int *roixmin, int *roixmax,
                          int *roiymin, int *roiymax,
                          int *hbin, int *vbin,
                          char *(*tab));
// get values from the dialogbox

int WINAPI STATUS_DIALOG_CAM(int *hwnd,int *status);
// hwnd   : window handle of Dialogbox
//          NULL if Dialogbox is closed
// status : 0 no user input, since last call
//          1 any user input, since last call

int WINAPI OPEN_DIALOG_BW(HWND hwnd, int mode, char *title);
// create an thread, which controls an dialog box, where you can input
// the black&white LUT. If one value is changed LOAD_OUTPUT_LUT is send.
// hWnd   : your main window handle
// mode=0 : no message is send to hWnd
// mode=1 : message is send to hWnd
//          (PostMessage(hwnd,WM_COMMAND,updmsg,0);
//          (updmsg is IDC_UPDATEBW or read out from the registry)
// title  : Null terminated ASCII string
//          title of the window
//          if NULL default title is used

int WINAPI LOCK_DIALOG_BW(int mode);
// inhibits user input to dialogbox
// mode=0 : user can change values
// mode=1 : user cannot change values

int WINAPI SET_DIALOG_BW(int bwmin, int bwmax, int bwlinlog);
// create an new lut with the values
// bwmin : 0..4094 lower offset all below will cxovert to black
// bwmax : 1..4095 upper offset all above will convert to white
// bwlinlo=0 : linear curve between bwmin and bwmax
//        =1 : logarithmic curve between bwmin and bwmax

int WINAPI STATUS_DIALOG_BW(int *hwnd,int *status);
// hwnd   : window handle of Dialogbox
//          NULL if Dialogbox is closed
// status : 0 no user input, since last call
//          1 any user input, since last call

int WINAPI GET_DIALOG_BW(int *lutmin,int *lutmax,int *lutlinlog);

int WINAPI CLOSE_DIALOG_BW(void);
// closes the dialog box

// the same as BW-Dialog only for COLOR-LUT
int WINAPI OPEN_DIALOG_COL(HWND hwnd, int mode, char *title);
int WINAPI LOCK_DIALOG_COL(int mode);
int WINAPI SET_DIALOG_COL(int redmin, int redmax,
                          int greenmin, int greenmax,
                          int bluemin, int bluemax,
                          int linlog);
int WINAPI GET_DIALOG_COL(int *redmin, int *redmax,
                          int *greenmin, int *greenmax,
                          int *bluemin, int *bluemax,
                          int *linlog);
int WINAPI STATUS_DIALOG_COL(int *hwnd,int *status);
int WINAPI CLOSE_DIALOG_COL(void);

//*****************************************************************
// the following functions are for the MULTI-BUFFER Option
//

int WINAPI GET_CCDSIZE(int *ccdsize);
int WINAPI GET_CCD_SIZE(int *ccdsize);
// returns ccdsize of connected camera

int WINAPI ALLOC_RECORDER(int *bufanz, int ccdsize);
// allocates *bufanz buffer's with Pixelsize ccdsize
// if *bufanz=0 allocates as much as possible
// returns count of buffers in *bufanz

int WINAPI FREE_RECORDER(void);
// free allocated buffers

int WINAPI GET_RECORDER_ADDR(int buf, int *linadr);
// get address of allocated buffer
// buf: number of buffer
// *linadr 32bit address

int WINAPI SET_BUFFER_SIZE(int *numbers, int width,int height);
// creates an queue of buffer adresses for buffers with size width*height
// *numbers: count of buffers
//           returns possible buffers
// width:    width  of  pictures in Pixels
// height:   height of  pictures in Pixels

int WINAPI GET_BUFFER_ADDR(int picnr,int *linadr,int *picwidth,int *picheight);
// gets linadr and width,height of picture Nr picnr

int WINAPI GET_DMA_STATUS(int *count);
// returns buffer, in which the dma transfers now

int WINAPI RUN_DMA(int pics,int pice, int mode);
//starts dma
//pics:   number of buffer to start
//pice:   number of buffer to end
//mode:  0=writing from start to end and return from routine
//       1=wraping continuous
//       in both modes the windows message queue is called, while
//       waiting for pictures come in and waiting for DMA is done
//       call DMA_STOP() to return from function

int WINAPI STOP_DMA(void);
//stops dma at the next possible time, an running DMA is completed

int WINAPI DMA_START_SINGLE(int pics);
//starts one single dma, call DMA_DONE to see if done

int WINAPI DMA_DONE(int *pic);
//pic : 0 dma done
//      else buffernr called in Dma_Start_Single

int WINAPI RUN_DMA_AVG(int pics, int pice, int mode,int avgmode);
//start averanging
//pics:  number of buffer to start
//pice:  number of buffer to end
//mode:  0=writing from start to end and return from routine
//       1=wraping continuous
//avgmode: count of pictures to average
//       in both modes the windows message queue is called, while
//       waiting for pictures come in and waiting for DMA is done
//       call DMA_STOP() to return from function


void WINAPI GET_PALETTE(HPALETTE *ghpal);
//returns the Palette used for dialogs

int WINAPI  AUTO_MINMAX(int mode, int *min, int *max, int width,int height,word *frame);
//calculate min and max values in the picture in frame
//mode   : 0 return
//         1 set actual LUT and return
//*min	 : minimal value in picture
//*max	 : maximal value in picture
//width  : number of pixel in each line in the image
//height : number of lines in the image
//frame  : pointer to begin of PC memory area to calculate or
//         NULL, allocate a buffer,read in a single picture and calculate

int WINAPI  AUTO_MINMAX_OFF(int mode, int *min, int *max,int lowoff, int hioff, int width,int height,word *frame);
//calculate min and max values excluding all pixel
//below lowoff percent and above hioff percent
//mode   : 0 return
//         1 set actual LUT and return
//*min	 : minimal value in picture
//*max	 : maximal value in picture
//width  : number of pixel in each line in the image
//height : number of lines in the image
//frame  : pointer to begin of PC memory area to calculate or
//         NULL, allocate a buffer,read in a single picture and calculate

int WINAPI AUTO_COLOR(int width,int height,word *frame);
//set the actual BW-LUT so, that it is between lowoff percent
//and hioff percent of values in the picture in frame
//if frame==NULL, allocate a buffer and read in a single picture

int WINAPI AUTO_RANGE(int mode, int *time, int off,int width,int height,word *frame);


int WINAPI READ_REGISTRY_VALUES(void);

float WINAPI GET_COCTIME(void);
float WINAPI GET_BELTIME(void);
float WINAPI GET_EXPTIME(void);
float WINAPI GET_DELTIME(void);


#ifdef __cplusplus
}            // Assume C declarations for C++
#endif  //C++

