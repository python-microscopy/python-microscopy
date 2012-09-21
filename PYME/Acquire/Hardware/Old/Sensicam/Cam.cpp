// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * Cam.cpp implementation of CCamera, based on ... 
// *
// *CamOp.cpp:                           *
// * (C) 1999-2001 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am  1. Juni      1999                                      *
// * 95. Änderung am 13. November  2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * *
// * ACHTUNG! HINWEIS!!  *
// * * * * * * * * * * * *
//
// Diese Version von CCamera unterscheidet sich von der ursprünglichen (in Pro-
// gramm-Version < 3.4) dadurch, dass ZWEI unterschiedliche CCD-Kameras ange-
// steuert werden können.
//
// Diese sind:
// 1. PCO "SensiCam", Rot/Grün/Blau, 1280 x 1024 Pixel
// 2. LaVision "Imager3", Schwarz/Weiß, 1280 x 1024 Pixel.
//
// Die Kameras können jedoch nicht gleichzeitig, sondern nur einzeln angesteu-
// ert werden.
// 
// Um zu Programmstart eine Default-Kamera zu wählen, bitte 
// den entsprechenden Eintrag in die Registry vornehmen:
// 
// Software\\Uni-Heidelberg\\SMIProject\\Settings\\Camera1

//#include "stdafx.h"
#include <stdio.h>
#include <math.h>
#include "Cam.h"
#include "Sencam.h"


CCamera::CCamera(){
  // Initialisierung der internen Parameter bzw. der Parameter, die zur Lauf-
  // zeit vom Benutzer nicht verändert werden können:

  //int I;
  // -------------------------------- Strings für (Fehler-)Meldungen:
  m_strMessageKey  = "CameraErrors";
  m_strMessage[0]  = "Initialisierung fehlgeschlagen.\nKeine Kamara registriert.";
  m_strMessage[1]  = "Überschreitung des Zeitlimits\naller Funktionen.";
  m_strMessage[2]  = "Funktionsaufruf mit falschen Parametern.";
  m_strMessage[3]  = "PCI-Karte oder Treiber nicht gefunden.";
  m_strMessage[4]  = "DMA-Puffer nicht ansprechbar.";
  m_strMessage[5]  = "";
  m_strMessage[6]  = "DMA-Zeitlimit-Überschreitung.";
  m_strMessage[7]  = "Ungültiger Kamera-Modus.";
  m_strMessage[8]  = "Kein Treiber installiert.";
  m_strMessage[9]  = "Kein PCI-BIOS gefunden.";
  m_strMessage[10] = "Gerät wird von einem anderen Prozeß beansprucht.";
  m_strMessage[11] = "Fehler beim Lesen oder Schreiben\nvon Daten auf Board.";
  m_strMessage[12] = "";
  m_strMessage[13] = "";
  m_strMessage[14] = "";
  m_strMessage[15] = "";
  m_strMessage[16] = "";
  m_strMessage[17] = "";
  m_strMessage[18] = "";
  m_strMessage[19] = "LOAD_COC-Fehler\nKamera führt Programmspeicher aus).";
  m_strMessage[20] = "Zu viele Werte in COC.";
  m_strMessage[21] = "CCD- oder Elektronik-Temperatur zu hoch.";
  m_strMessage[22] = "Puffer-Ansprech-Fehler.";
  m_strMessage[23] = "READ_IMAGE-Fehler.";
  m_strMessage[24] = "Set/Reset-Puffer-Flag fehlgeschlagen.";
  m_strMessage[25] = "Puffer belegt.";
  m_strMessage[26] = "Aufruf einer Windows-Funktion\nfehlgeschlagen.";
  m_strMessage[27] = "DMA-Fehler.";
  m_strMessage[28] = "Datei kann nicht geöffnet werden.";
  m_strMessage[29] = "Registry-Fehler";
  m_strMessage[30] = "Fehler beim Öffnen eines Dialogfelds.";
  m_strMessage[31] = "Benötige neuere VXD oder DLL.";
  m_strMessage[32] = "Zu wenig freier RAM-Speicher vorhanden.";
  m_strMessage[33] = "Keine Bilddaten vorhanden.";
  m_strMessage[34] = "Falscher Kameratyp.";
  m_strMessage[35] = "Kamera-Fehler.\nBitte zuerst initialisieren.";
  m_strMessage[36] = " (Fehler Nr. ";
  // ----------------------------------------------------------------
  m_iCamType      = 0;
  m_iDataType     = 1;
  m_iCCDTypeIndex = 0;
  m_iCamVerIndex  = 0;
  m_iCamIDIndex   = 0;
  m_iDigitization = 12;
  m_iCCDTemp      = 0;
  boardNum = 0;

  m_bCamOK = FALSE;   // default: deaktiviert
  //Init();  
}

CCamera::~CCamera()
{
	SET_INIT(0);
	delete []currentFrame;
}

// ================================================================= Konstanten
// --------------------------------------------------------- mögliche CCD-Typen
const char* CCamera::m_strCCDType[6] = {
  "640 x 480, B/W",
  "640 x 480, B/W",
  "1280 x 1024, B/W",
  "640 x 480, RGB",
  "640 x 480, RGB",
  "1280 x 1024, RGB",
};

// -------------------------------------------------------- mögliche Kamera-IDs
const char* CCamera::m_strCamID[4] = {
  "LongExposure",
  "FastShutter/DoubleShutter",
  "Special Version",
  "DiCAM-PRO"
};

// -------------------------------------------------- mögliche Kamera-Versionen
const char* CCamera::m_strCamVer[6] = {
  "1.0", "1.5", "2.0", "2.5", "3.0", "3.5"
};

// ----------------------------------- mögliche Binningeinstellungen horizontal
const int CCamera::m_iHorizBin[4] = {
  1, 2, 4, 8
};

// ------------------------------------- mögliche Binningeinstellungen vertikal
const int CCamera::m_iVertBin[6] = {
  1, 2, 4, 8, 16, 32
};



// ------------------------------------------------------------ Initialisierung
int CCamera::Init()
{
	m_bCamOK = TRUE; 

	int iErr  = 0;
	int iSize = 0;

	iErr = SET_BOARD(boardNum);
	DisplayError(iErr);
	if (iErr < 0) return iErr;
	
//	printf("boardnum set: %d, %d\n", iErr, m_bCamOK);

	iErr = SET_INIT(1);
	DisplayError(iErr);
	if (iErr < 0) return iErr; 
	
//	printf("boardnum set: %d, %d\n", iErr, m_bCamOK);
	 
	iErr = GET_CCD_SIZE(&iSize);
	DisplayError(iErr);
	if (iErr < 0) return iErr;

//	printf("boardnum set: %d, %d\n", iErr, m_bCamOK);

	switch (iSize)
	{
	case 307200: //VGA
		CCDWidth  = 640;
		CCDHeight = 480;
		break;
	case 1310720: //SVGA
		CCDWidth  = 1280;
		CCDHeight = 1024;
		break;
	case 1431040: //SensicamQE
		CCDWidth  = 1376;
		CCDHeight = 1040;
		break;
	default: // Camera we don't know about
		iErr = -34;
		return iErr;
	};
	  
	InitParameters();  // Parameter auf voreingestellte Werte setzen

	//printf("boardnum set: %d, %d\n", iErr, m_bCamOK);
      
	iErr = SetCOC();       // Hardware mit Parametern programmieren
	DisplayError(iErr);
	if(iErr < 0) return iErr;

//	printf("boardnum set: %d, %d\n", iErr, m_bCamOK);

	iErr = GetStatus();
	if(iErr < 0) return iErr;

	currentFrame = new word[CCDWidth*CCDHeight];

	m_bCamOK = TRUE; 
	return iErr;
}

// -------------------------------- Parameter mit Default-Werten initialisieren
int CCamera::InitParameters(){
  m_iTrigMode      = 0;      // Trigger-Modus sequentiell
  m_iDelayTime     = 0;      // Verzögerungszeit in ms
  m_iIntegTime     = 100;    // Integrationszeit in ms
  m_iROIMode       = 0;      // ROI-Modus
  m_iROIPosX       = 0;
  m_iROIPosY       = 0;
  m_iROIWidth      = CCDWidth;   // ganze CCD-Breite
  m_iROIHeight     = CCDHeight;  // ganze CCD-Höhe
  m_iSelfROIWidth  = 640;
  m_iSelfROIHeight = 512;
  m_iHorizBinIndex = 0;
  m_iVertBinIndex  = 0;
  m_iBinningX      = 1;
  m_iBinningY      = 1;

  //iMode = 768;  // 0 = sequentiell, 65536 = parallel
  iMode = 66304;
  contMode = TRUE;

  SetROI(0,0,CCDWidth,CCDHeight);

  return 0;
}

// ---------------------- Kameratyp, Elektronik- und CCD-Temperatur ermitteln -
// |               Diese Funktion ist vor Aufruf der Funktionen GetCCDTemp(), |
// |                               GetElectrTemp() und GetCamID() aufzurufen! |
// ----------------------------------------------------------------------------
int CCamera::GetStatus(){
  if(m_bCamOK == FALSE){
    DisplayError(-35);
    return -35;
  }//endif
  int iCamType;
  
  int iErr = 0;

  iErr = SET_BOARD(boardNum + 0x100);

  iErr = GET_STATUS(&iCamType, &m_iElectrTemp, &m_iCCDTemp);
  
  m_iCCDTypeIndex = 3*Bit(iCamType, 8) + 2*Bit(iCamType, 15) +
                    Bit(iCamType, 14);   // CCD-Typ ermitteln
  m_iCamIDIndex   = 2*Bit(iCamType, 13) + Bit(iCamType, 12);  // Kamera-
                    // Typ bzw. Camera-ID ermitteln
  m_iCamVerIndex  = 4*Bit(iCamType, 11) + 2*Bit(iCamType, 10) + 
                    Bit(iCamType, 9);   // Kamera-Version ermitteln

  m_iCamType = 0;
  if(m_iCCDTypeIndex == 5) m_iCamType = 1;  // Rot/Grün/Blau-Kamera
  if(m_iCCDTypeIndex == 2) m_iCamType = 2;  // Schwarz/Weiß-Kamera

  if((m_iCamType == 0) || (m_iCamIDIndex != 0)) iErr = -34;  // Nur CCD-Typ
    // 1280x1024, RGB oder S/W und Kameratyp LongExposure akzeptieren!
  DisplayError(iErr);

  if(m_iCamType == 1) // RGB-Kamera
	{  
		if(m_iBinningX == 1)
		{
			m_iChannels = 4;  // Bei Binning = 1 werden alle 4 Farbkanäle eingelesen
		}else
		{
			m_iChannels = 1;
		}
	}else // S/W-Kamera
	{                
		m_iChannels = 1;
	}

  return iErr;
}

// ------- COC (Camera Operation Code) generieren, d. h. Hardware programmieren
// Erst der Aufruf dieser Methode gleicht die Parameter der Kamera-Hardware mit
// -- den zuvor gesetzten Werten dieser Klasse wie z.B. Integrationszeit ab! --
int CCamera::SetCOC(){
  if(m_bCamOK == FALSE){
    DisplayError(-35);
    return -99;
  }//endif

  int  iErr = SET_BOARD(boardNum + 0x100);
//  printf("boardnum set: %d, %d\n", iErr, m_bCamOK);

  //int iMode = 0;  // 0 = sequentiell, 65536 = parallel
  int iTrig = 0;  // Auto Start, Auto Frame
  //int iROIX1 = (m_iROIPosX + 32)/32;  // iROIX1/2 = 1..40
  //int iROIX2 = iROIX1 - 1 + m_iROIWidth/32;
  //int iROIY1 = (m_iROIPosY + 32)/32;  // iROIY1/2 = 1..32
  //int iROIY2 = iROIY1 - 1 + m_iROIHeight/32;
  
  int iHBin  = m_iBinningX;
  int iVBin  = m_iBinningY;
  
  char pTable[50];
  sprintf(pTable,"0,%7d,-1,-1",m_iIntegTime);

  
  /*char rstring[255];
  sprintf(rstring,"%d  %d  %d  %d  %d  %d",ROIX1, ROIX2, ROIY1, ROIY2, CCDWidth, CCDHeight);
  MessageBox(NULL,rstring,"ROI",MB_OK);*/

//  printf("COC: %d, %d, %d, %d, %d, %d, %d, %d, %s, %d\n", iMode, iTrig, ROIX1/32 + 1, ROIX2/32, ROIY1/32 + 1, ROIY2/32, iHBin,
//    iVBin, pTable, GetIntegTime());

  iErr = SET_COC(iMode, iTrig, ROIX1/32 + 1, ROIX2/32, ROIY1/32 + 1, ROIY2/32, iHBin,
    iVBin, pTable);
//  printf("COC set: %d, %d\n", iErr, m_bCamOK);

  DisplayError(iErr);

  iErr = GET_IMAGE_SIZE(&picWidth, &picHeight);
  DisplayError(iErr);
//  printf("image size got: %d, %d\n", iErr, m_bCamOK);

  return iErr;
}





//// -------------------------- Neue ROI-Koordinaten setzen (SP kann sich ändern)
//void CCamera::SetNewROI(int iCenterX, int iCenterY, int iWidth, int iHeight,
//  bool bButton)  // Es werden Bild-, nicht ROI-Koordinaten übergeben!
//{
//  // Neue ROI-Position bestimmen (alle ROI-Koordinaten müssen Vielfache von
//  // 32 sein!):
//  int iNewROIPosX, iNewROIPosY;
//
//  iCenterX *= 2;  // Umrechung auf CCD-Koordinaten
//  iCenterY *= 2;
//  iWidth   *= 2;
//  iHeight  *= 2;
//  if(bButton == TRUE){  // Breite und Höhe werden nicht verändert
//    // (linke Maustaste)
//    iNewROIPosX = (int)(((iCenterX - iWidth/2) + 16)/32)*32;
//    iNewROIPosY = (int)(((iCenterY - iHeight/2) + 16)/32)*32;
//  }else{  // auch Breite und Höhe neu festlegen (iCenterX/Y = linke obere Ecke,
//    // rechte Maustaste)
//    iNewROIPosX = (int)((iCenterX + 16)/32)*32;
//    iNewROIPosY = (int)((iCenterY + 16)/32)*32;
//    iWidth      = (int)((iWidth + 16)/32)*32;
//    iHeight     = (int)((iHeight + 16)/32)*32;
//    if(iWidth == 0)  iWidth  = 32;  // minimale ROI-Größe = 32 x 32 Pixel
//    if(iHeight == 0) iHeight = 32;
//  }//endif
//  // ROI darf nicht breiter oder höher als CCD sein:
//  if(iWidth > CCDWidth)   iWidth = CCDWidth;
//  if(iHeight > CCDHeight) iHeight = CCDHeight;
//  // ROI darf nicht über CCD-Rand hinausreichen:
//  if(iNewROIPosX < 0) iNewROIPosX = 0;
//  if((iNewROIPosX + iWidth) >= CCDWidth) iNewROIPosX = CCDWidth - iWidth;
//  if(iNewROIPosY < 0) iNewROIPosY = 0;
//  if((iNewROIPosY + iHeight) >= CCDHeight)
//    iNewROIPosY = CCDHeight - iHeight;
//  m_iROIPosX   = iNewROIPosX;
//  m_iROIPosY   = iNewROIPosY;
//  m_iROIWidth  = iWidth;
//  m_iROIHeight = iHeight;
//  if(bButton == FALSE){
//    m_iSelfROIWidth  = iWidth;
//    m_iSelfROIHeight = iHeight;
//  }//endif
//}

// -------------------------- Neue ROI-Koordinaten setzen (SP kann sich ändern)
void CCamera::SetROI(int x1, int y1, int x2, int y2)
{
  // Neue ROI-Position bestimmen (alle ROI-Koordinaten müssen Vielfache von
  // 32 sein!):
  //int iNewROIPosX, iNewROIPosY;

  /*char rstring[255];
  sprintf(rstring,"%d  %d  %d  %d  %d  %d",x1, x2, y1, y2, CCDWidth, CCDHeight);
  MessageBox(NULL,rstring,"Set ROI",MB_OK);*/

  if (GetNumberChannels() == 4 || m_iBinningX > 1)
  {
	x1 *= 2;  // Umrechung auf CCD-Koordinaten ... most likely broken for binning other than 1 or 2
	y1 *= 2;
	x2 *= 2;
	y2 *= 2;
  }

  //check that roi corners are not reversed
  if (x2 < x1)
  {
	  int t;
	  t = x1;
	  x1 = x2;
	  x2 = t;
  }

  if (y2 < y1)
  {
	  int t;
	  t = y1;
	  y1 = y2;
	  y2 = t;
  }

  //take nearest multiple of 32
  x1 = (int)(32*floor(((double)x1)/32.0));
  y1 = (int)(32*floor(((double)y1)/32.0));
  x2 = (int)(32*ceil(((double)x2)/32.0));
  y2 = (int)(32*ceil(((double)y2)/32.0));

  //minimum size of ROI = 32x32
  if(x2 < (x1 + 32)) x2 = x1 + 32;
  if(y2 < (y1 + 32)) y2 = y1 + 32;

  // ROI darf nicht breiter oder höher als CCD sein:
  if (x1 < 0 || x1 > (CCDWidth - 32)) x1 = 0; 
  if (x2 < 32 || x2 > CCDWidth) x2 = CCDWidth;

  if (y1 < 0 || y1 > (CCDHeight - 32)) y1 = 0; 
  if (y2 < 32 || y2 > CCDHeight) y2 = CCDHeight;

  ROIX1   = x1;
  ROIX2   = x2;
  ROIY1   = y1;
  ROIY2   = y2;
  
}

void CCamera::SetCamModeEx(bool lowLight, bool continuous)
{
    iMode = 3*(int)lowLight*256 + (int)continuous*65536;
    contMode = continuous;
}

float CCamera::GetCycleTime()
{
    return GET_COCTIME()*1000;
}

// ---------------------------------------------------- Einzel-Aufnahme starten
int CCamera::StartExposure(){
  // iErr      = Fehlernummer
	int iErr = 0;
  if(m_bCamOK == FALSE){
    iErr = -35;
    DisplayError(iErr);
    return iErr;
  }//endif

  iErr = SET_BOARD(boardNum + 0x100);
  //iErr = STOP_COC(0);   // eventuell noch laufende Aufnahme stoppen, PCI-
    // Interface-Board-Puffer leeren.
  DisplayError(iErr);
  if(m_bCamOK == FALSE) return iErr;

  iErr = STOP_COC(0);
  DisplayError(iErr);

  if (contMode == FALSE)
  {
      iErr = RUN_COC(4);   // 4 => Einzelaufnahme starten
      DisplayError(iErr);
  } else
  {
      iErr = RUN_COC(0);   // 0 => Kontinuierliche Aufnahme starten
      DisplayError(iErr);
  }

  return iErr;
}

// ------------------------------------------------------ Life-Vorschau starten
int CCamera::StartLifePreview()
{
	int iErr = 0;
  // iErr      = Fehlernummer
  if(m_bCamOK == FALSE){
    iErr = -35;
    DisplayError(iErr);
    return iErr;
  }//endif

  iErr = SET_BOARD(boardNum + 0x100);

  iErr = STOP_COC(0);
  DisplayError(iErr);
  if(m_bCamOK == FALSE) return iErr;
  iErr = RUN_COC(0);   // 0 => Kontinuierliche Aufnahme starten
  DisplayError(iErr);
  //MessageBox(0,"Start Prev Done", "Debug", MB_OK);
  return iErr;
}



// ------------------------------------------------------ Life-Vorschau stoppen
int CCamera::StopLifePreview()
{
	int iErr = 0;
	
	iErr = SET_BOARD(boardNum + 0x100);
	
	
  if(m_bCamOK == TRUE){
    iErr = STOP_COC(0);
    DisplayError(iErr);
  }//endif
  return iErr;
}//endif

// -------------------------------------------- Nachschauen, ob Aufnahme fertig
bool CCamera::ExpReady(){  // liefert TRUE, falls Bild verfügbar
	int iErr = 0;
  
	iErr = SET_BOARD(boardNum + 0x100);
	
	if(m_bCamOK == FALSE){
    iErr = -35;
    DisplayError(iErr);
    return FALSE;
  }//endif
  int Status = 0;  // enthält Kamerastatus-Information
  iErr = GET_IMAGE_STATUS(&Status);
  DisplayError(iErr);
  if((m_bCamOK == FALSE) || (Bit(Status, 1) != 0)) return FALSE;

  //MessageBox(0,"GetPicFromCam","Debug",MB_OK);
  GetPicFromCam();

  return TRUE;
}

// ---------------------------------- Bild einlesen und in RAM-Speicher ablegen
int CCamera::GetPicFromCam(){
	int iErr = 0;
  // p12BitPic = Zeiger auf word-Element (2 Byte), an dem Bilddaten beginnen
  // iErr      = Fehlernummer
  
	iErr = SET_BOARD(boardNum + 0x100);
	
	if(m_bCamOK == FALSE){
    iErr = -35;
    DisplayError(iErr);
    return iErr;
  }//endif
  // Aufnahme war erfolgreich => Bilddaten in Speicher einlesen:
  iErr = GET_IMAGE_SIZE(&picWidth, &picHeight);
  DisplayError(iErr);
  if(m_bCamOK == FALSE) return iErr;

  //MessageBox(0,"Read Image 12 bit", "Debug",MB_OK);
  iErr = READ_IMAGE_12BIT(0, picWidth, picHeight, currentFrame);
    // Bild von Kameraspeicher in PC-Speicher übertragen
  //MessageBox(0,"Read Image 12 bit done", "Debug",MB_OK);
  DisplayError(iErr);
  if(m_bCamOK == FALSE) return iErr;

  //m_iDispBinning = m_iBinningX;  // Bilddaten müssen passend zu den jeweiligen
  m_iDataType    = m_iCamType;   // Einstellungen der Kamera bzw. passend zum
    // jeweiligen Kamera-Typ interpretiert werden!

  if(m_iCamType == 1){  // RGB-Kamera
    if(m_iBinningX == 1){
      m_iChannels = 4;  // Bei Binning = 1 werden alle 4 Farbkanäle eingelesen
    }else{
      m_iChannels = 1;
    }
  }else{                // S/W-Kamera
    m_iChannels = 1;
  }
  return iErr;
}

// ---------- Bildparameter (notwendig zur Berechnung der RGB-Bilddaten) setzen
/*void CCamera::SetPicParameters(int iBin, int iChannels, int iDataType,
  int iWidth, int iHeight)
{
  // Die hier zu übergebenden Parameter beziehen sich auf Bilddaten!
  // Die hier zu setzenden Parameter sind Attribute der Rohdaten. Diese Funk-
  // tion muss aufgerufen werden, wenn die Rohdaten nicht direkt von der Kamera
  // stammen (s. GetPicture()), sondern z. B. aus einer Datei.

  ASSERT((iBin > 0) && (iBin < 3));  // Nur Binning = 1 oder 2 zulassen
  ASSERT((iBin == 1) || (iBin == 2) && (iChannels == 1));
  ASSERT((iChannels > 0) && (iChannels < 5));  // Nur 1..4 Kanäle zulassen
  ASSERT((iDataType == 1) || (iDataType == 2));  // Nur Datentyp 1 oder 2
  ASSERT((iWidth > -1) && (iWidth < CCDWidth));  // Bildbreite muss zwischen
    // -1 und CCD-Breite liegen
  ASSERT((iHeight > -1) && (iHeight < CCDHeight));  // Bildhöhe muss zwi-
    // schen -1 und CCD-Höhe liegen

//  if(m_iDataType != iDataType) SetColorLUT();  // Falls sich Datentyp geändert
    // hat, muss LUT neu berechnet werden.

  //m_iDispBinning = iBin;
  m_iChannels    = iChannels;
  m_iDataType    = iDataType;

  /*if((iWidth > CCDWidth/2) || (iHeight > CCDHeight/2)){  // Bilddaten
    // können dann nur von S/W-Kamera stammen.
    m_iDataType = 2;
    m_iChannels = 1;
    //SetColorLUT();
  }//endif

  picWidth  = iWidth*CCDPixelChoice();   // Umrechung auf Breite und
  picHeight = iHeight*CCDPixelChoice();  // Höhe auf CCD-Chip
}*/

// ------------------------------------ Breite der 12-Bit-Bilddaten zurückgeben
int CCamera::GetPicWidth(){
  if (m_iChannels == 1) return picWidth;

	else return picWidth/2;
}

// -------------------------------------- Höhe der 12-Bit-Bilddaten zurückgeben
int CCamera::GetPicHeight(){
  
	if (m_iChannels == 1) return picHeight;

	else return picHeight/2;
}


// ----------------- Bilddaten für Rot, Grün, Blau aus CCD-Speicher extrahieren
void CCamera::ExtractColor(word* p12BitEx, int iMode){
  // p12BitPic: Zeiger auf Anfang der Bilddaten
  // p12BitEx:  Zeiger auf Anfang der extrahierten Daten
  // iMode = 0: Schwarz/Weiß extrahieren
  //       = 1: Rot  extrahieren
  //       = 2: Grün     "
  //       = 3: Blau     "

  ASSERT((iMode > -1) && (iMode < 5));
  //if(m_iDataType == 2) return;  // Aufruf dieser Methode macht bei Schwarz/
    // Weiß-Daten keinen Sinn.

  if (iMode == 0 || m_iChannels == 1) GetBWPicture(p12BitEx);
  else
  {
	word* p12 = Get12BitData(currentFrame, iMode);
	int L     = CCDPixelChoice();

	// Werte selektieren:
	for(int I = 0; I < picHeight; I += 2){  // jede L. CCD-Zeile durchgehen
		for(int J = 0; J < picWidth; J += 2){  // Innerhalb einer CCD-Zeile
		// jedes L. Pixel nehmen:
		*p12BitEx = *p12;
		p12      += 2;
		p12BitEx += 1;
		}//next J
		p12 += picWidth;  // falls Binning = 1, eine Zeile überspringen
	}//next I
  }
}



// ----------------- Übergabewerte auf Konsistenz mit Bildparametern überprüfen
bool CCamera::CheckCoordinates(int iPosX, int iPosY, int iWidth, int iHeight){
  //int iFactor     = 3 - m_iDispBinning;
  int iFullWidth  = picWidth;///iFactor;   // Breite des vollen Bildes
  int iFullHeight = picHeight;///iFactor;  // Höhe    "    "      "

  // Übergabewerte überprüfen:
  if((iFullWidth == 0) || (iFullHeight == 0)) return FALSE;
  if((iPosX < 0) || (iPosY < 0) || (iWidth <= 0) || (iHeight <= 0))
    return FALSE;
  if(((iPosX + iWidth) > iFullWidth) || ((iPosY + iHeight) > iFullHeight))
    return FALSE;
  return TRUE;
}

// ============================================================ Hilfsfunktionen
// -------------------------- Bit an Position BitPos einer int-Zahl zurückgeben
int CCamera::Bit(int iZahl, int iBitPos){
  int iZahl2 = (iZahl & (1 << iBitPos));
  if(iZahl2 == 0){
    return 0;
  }else{
    return 1;
  }//endif
}

// ------------------------------------------------------ Kamerafehler anzeigen
void CCamera::DisplayError(int iErr){
  ASSERT(iErr > -37);
  string strErr;
  char strErr1[10];

  if(iErr < 0){
    sprintf(strErr1,"%d",-iErr);
    strErr = (m_strMessage[-iErr - 1]) + m_strMessage[36] + strErr1 + ")";
    //MessageBox(NULL,strErr.c_str(),"Camera Error",MB_OK);
    m_bCamOK = FALSE;
  }//endif
}

// ---------- Zeiger auf 12-Bit-Daten in Abhängigkeít vom gewünschten Farbkanal
inline word* CCamera::Get12BitData(word* p12BitPic, int iMode){
  //if(m_iDataType == 2) iMode = 0;  // bei Schwarz/Weiß-Kamera immer nur Zeiger
    // auf p12BitPic zurückliefern!
  switch(iMode){
  case 0:  // Binning = 2 => Jedes CCD-Pixel nehmen
    return p12BitPic;
    break;
  case 1:  // Rot
    return p12BitPic;
    break;
  case 2:  // Grün 1
    return(p12BitPic + 1);
    break;
  case 3:  // Blau
    return(p12BitPic +picWidth + 1);
    break;
  case 4:  // Grün 2
    return(p12BitPic + picWidth);
    break;
  }//endswitch
  return NULL;
}


// ---------------------- Liefert 1 oder 2, je nachdem ob Binning = 2 oder 1...
inline int CCamera::CCDPixelChoice(){
  if(m_iDataType == 1){  // Rot/Grün/Blau-Kamera
    //return 3 - m_iDispBinning;  // Binning = 1 => Rückgabewert = 2, d.h. jedes
      // 2. Pixel nehmen; Binning = 2 => Rückgabewert = 1, d.h. jedes Pixel
      // nehmen.
	  return 2;
  }else{                // Schwarz/Weiß-Kamera
    return 1;  // jedes Pixel nehmen
  }//endif
}

void CCamera::GetBWPicture(word* p12BitEx)
{
	
	if(m_iChannels ==1)
	{
		memcpy(p12BitEx, currentFrame, picWidth*picHeight*2);
	}
	else
	{
		//average across the channels

		word *pR = currentFrame;
		word *pG1 = currentFrame + 1;
		word *pB = currentFrame + picWidth + 1;
		word *pG2 = currentFrame + picWidth;

		for(int I = 0; I < picHeight; I += 2)
		{ 
			for(int J = 0; J < picWidth; J += 2)
			{  				
				*p12BitEx = (*pR + *pG1 + *pG2 + *pB);
				pR+=2;
				pG1+=2;
				pG2+=2;
				pB+=2;
				p12BitEx ++;
			}

			pR += picWidth; 
			pG1 += picWidth;
			pG2 += picWidth;
			pB += picWidth;
		}
	}
}
