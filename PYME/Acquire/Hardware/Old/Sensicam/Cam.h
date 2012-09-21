// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * CamOp.h: Schnittstelle für die Klasse CCamOp.                           *
// * (C) 1999-2001 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am  1. Juni      1999                                      *
// * 65. Änderung am  6. August    2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Die Klasse CCamOp (Mitglied der Dokumentklasse) enthält
//  - Kommandos zur Kamera-Ansteuerung (Setzen von Integrationszeit, Binning,
//    ROI; Starten einer Aufnahme usw.)
//  - Berechnung der anzuzeigenden Bilddaten (8 Bit) aus den Kameradaten
//    (12 Bit)
//  - Bildattribute (Breite, Höhe, Binning)
//  - Histogrammfunktionen
//  - Funktionen zum Zugriff auf rote, grüne oder blaue Bilddaten (nur bei RGB-
//    Kamera)
//  - (Kanalweises, nur bei RGB) Aufsummieren der Intensitätswerte innerhalb
//    einer ROI

// * * * * * * * * * * * *
// * ACHTUNG! HINWEIS!!  *
// * * * * * * * * * * * *
//
// Diese Version von CCamOp unterscheidet sich von der ursprünglichen (in Pro-
// gramm-Version < 3.4) dadurch, dass ZWEI unterschiedliche CCD-Kameras ange-
// steuert werden können.
//
// Diese sind:
// 1. PCO "SensiCam", Rot/Grün/Blau, 1280 x 1024 Pixel
// 2. LaVision "Imager3", Schwarz/Weiß, 1280 x 1024 Pixel.
//
// Die Kameras können jedoch nicht gleichzeitig, sondern nur einzeln angesteu-
// ert werden.

#if !defined(AFX_CAM_H__6D9C46BA_17F0_11D3_9900_0000C0E169AB__INCLUDED_)
#define AFX_CAMOP_H__6D9C46BA_17F0_11D3_9900_0000C0E169AB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <windows.h>

//typedef bool BOOL 

#define ASSERT //
#define BOOL bool

//#define  LPCTSTR *char

//#include <cstringt.h>
#include <string>
using namespace std;

typedef unsigned char  byte;     /* 8-bit  */
typedef unsigned short word;     /* 16-bit */
typedef unsigned long  dword;    /* 32-bit */

class CCamera
{
public:
  // Konstruktor/Destruktor:
  CCamera();
  virtual ~CCamera();

  // Elementzugriffsfunktionen:
  /*void SetMessage(int iIndex, CString strMessage){
    m_strMessage[iIndex] = strMessage;
  }
  CString GetMessageKey(){return m_strMessageKey;}*/
  int GetCamType()    {return m_iCamType;}
  int GetDataType()   {return m_iDataType;}
  int GetADBits()     {return m_iDigitization;}
  int GetMaxDigit()   {return m_iMaxDigit;}
  int GetNumberCh()  {return NumberCh;}
  int GetBytesPerPoint(){return sizeof(word);}
  string GetCCDType(){return string(m_strCCDType[m_iCCDTypeIndex]);}
  string GetCamID()  {return string(m_strCamID[m_iCamIDIndex]);}
  string GetCamVer() {return string(m_strCamVer[m_iCamVerIndex]);}
  void SetTrigMode    (int iTrig){m_iTrigMode = iTrig;}
  int GetTrigMode()   {return m_iTrigMode;}
  void SetDelayTime   (int iDelay){m_iDelayTime = iDelay;}
  int GetDelayTime()  {return m_iDelayTime;}
  void SetIntegTime   (int iInteg){m_iIntegTime = iInteg;}
  int GetIntegTime()  {return m_iIntegTime;}
  void SetROIMode     (int iMode){m_iROIMode = iMode;}
  int GetROIMode()    {return m_iROIMode;}
  void SetCamMode     (int im){iMode=im;}
  int GetCamMode()	  {return iMode;}
  void SetCamModeEx   (bool lowLight, bool continuous);

  void SetBoardNum    (int boardN){boardNum = boardN;}
  int GetBoardNum()  {return boardNum;}

  // Die Funktionen für Zugriff auf CCD-bzw. ROI-Koordinaten geben bzw. er-
  // warten Bildkoordinaten (und nicht CCD-Koordinaten)!
  // Für Binning = 1 sind zwei Fälle zu unterscheiden:
  // 1. RGB-Kamera => 1 Bildpixel = 2x2 CCD-Pixel = 1x1 CCD-Pixel PRO KANAL
  // 2. S/W-Kamera => 1 Bildpixel = 2x2 CCD-Pixel
  // Für Binning = 2 gilt für beide Kameratypen:
  // 1 Bildpixel = 2x2 CCD-Pixel
  int GetCCDWidth()   {return CCDWidth;}
  int GetCCDHeight()  {return CCDHeight;}
  //void SetROIPosX     (int iPosX){m_iROIPosX = 2*iPosX;}
  //int GetROIPosX()    {return m_iROIPosX;}
  //void SetROIPosY     (int iPosY){m_iROIPosY = 2*iPosY;}
  //int GetROIPosY()    {return m_iROIPosY/2;}
  //void SetROIWidth    (int iWidth){m_iROIWidth = 2*iWidth;}
  //int GetROIWidth()   {return m_iROIWidth/2;}
  //void SetROIHeight   (int iHeight){m_iROIHeight = 2*iHeight;}
  //int GetROIHeight()  {return m_iROIHeight/2;}
  //void SetSelfROIWidth(int iWidth){m_iSelfROIWidth = 2*iWidth;}
  //int GetSelfROIWidth(){return m_iSelfROIWidth/2;}
  //void SetSelfROIHeight(int iHeight){m_iSelfROIHeight = 2*iHeight;}
  //int GetSelfROIHeight(){return m_iSelfROIHeight/2;}

  //void SetLogMode     (int iMode){m_iLogMode = iMode;}
  //int GetLogMode()    {return m_iLogMode;}
  void SetHorizBin    (int iBin){m_iBinningX = 
                        m_iHorizBin[m_iHorizBinIndex = iBin];}
  int GetHorizBin()   {return m_iHorizBinIndex;}  // liefert Index
  int GetHorzBinValue(){return m_iBinningX;}      // liefert Wert
  void SetVertBin     (int iBin){m_iBinningY = 
                        m_iVertBin[m_iVertBinIndex = iBin];}
  int GetVertBin()    {return m_iVertBinIndex;}
 
  int GetNumberChannels(){return m_iChannels;}
  
  int GetElectrTemp() {return m_iElectrTemp;}
  int GetCCDTemp()    {return m_iCCDTemp;}
  BOOL CamReady()     {return m_bCamOK;}

  float GetCycleTime();
  
 

  //void SetPicParameters(int iBin, int iChannels, int iDataType, int iWidth,
  //  int iHeight);                      // Bildparameter (notwendig zur Berech-
                                       // nung der RGB-Bilddaten) setzen
  int GetPicWidth();                   // Bildbreite (nicht Breite auf CCD-
                                       // Chip) zurückgeben
  int GetPicHeight();                  // Bildhöhe (nicht Höhe auf CCD-Chip)
                                       // zurückgeben
                                      
  // void SetNewROI(int iCenterX, int iCenterY, int iWidth, int iHeight,
  //  BOOL bButton);                     // neue ROI-Koordinaten setzen

  void SetROI(int x1, int y1, int x2, int y2);
  int GetROIX1(){return ROIX1;}
  int GetROIX2(){return ROIX2;}
  int GetROIY1(){return ROIY1;}
  int GetROIY2(){return ROIY2;}

  void DisplayError(int iErr);         // Kamera-Fehler anzeigen
  
  //void Serialize(CArchive &ar);        // Serialisierung

// Funktionen zur Kamera-Ansteuerung:
  int Init();                 // Initialisierung der Kamera
  int GetStatus();                     // Kamerastatus abfragen (insbesondere
                                       // Elektronik- und CCD-Temperatur)
  int SetCOC();                        // COC setzen
  int StartExposure();       // Einzel-Aufnahme starten
  int StartLifePreview();    // Life-Vorschau starten
  int StopLifePreview();     // Life-Vorschau stoppen

  BOOL ExpReady();             // liefert TRUE, falls Bild verfügbar
  void GetBWPicture(word* p12BitPic);  

  void ExtractColor(word* p12BitEx, int iMode);  // Bilddaten
                                       // für Rot, Grün, Blau aus CCD-Speicher
                                       // extrahieren
 
  BOOL CheckCoordinates(int iPosX, int iPosY, int iWidth, int iHeight);
                                       // Übergabewerte überprüfen

  bool contMode;
private:
  int GetPicFromCam();
  // Hilfsfunktionen:
  int InitParameters();        // Parameter mit Default-Werten initia-
                                       // lisieren
  int Bit(int iZahl, int iBitPos);     // Bit an Position BitPos einer int-Zahl
                                       // zurückgeben
  word* Get12BitData(word* p12BitPic, int iMode);  // liefert Zeiger auf 12-
                                       // Bit-Daten in Abhängigkeít vom gewün-
                                       // schten Farbkanal
  int CCDPixelChoice();                // Liefert 1 oder 2, je nachdem ob Bin-
                                       // ning = 2 oder 1...

  // Konstanten:
  string m_strMessageKey;             // Schlüsselwort für Messages-Datei
  string m_strMessage[100];           // Messages
  static const char* m_strCCDType[6];  // mögliche CCD-Typen
  int m_iCCDTypeIndex;                 // Index des aktuellen CCD-Typs
  static const char* m_strCamID[4];    // mögliche Kamera-IDs
  int m_iCamIDIndex;                   // Index der aktuellen Kamera-ID
  static const char* m_strCamVer[6];   // mögliche Kamera-Versionen
  int m_iCamVerIndex;                  // Index des aktuellen Kamera-Index
  static const int m_iHorizBin[4];     // mögliche Binningeinstellungen horiz.
  int m_iHorizBinIndex;                // Index des aktuellen horiz. Binnings
  static const int m_iVertBin[6];      // mögliche Binningeinstellungen vertik.

  // Variablen/Attribute:
  int m_iElectrTemp;   // Elektronik-Temperatur
  int m_iCamType;      // Kamera-Typ: = 1: RGB-Kamera; = 2: S/W-Kamera
  int m_iDataType;     // Gibt an, wie Bilddaten interpretiert werden müssen.
                       // = 1: wie bei RGB-Kamera; = 2: wie bei S/W-Kamera
  int m_iCCDTemp;      // CCD-Temperatur
  BOOL m_CamComp;      // = TRUE, falls Kamera-Typ mit dieser Anwendung ver-
                       // träglich.
  BOOL m_bCamOK;       // = TRUE, falls keine Fehler aufgetreten.
  int m_iChannels;     // Anzahl detektierter Farbkanäle
  int m_iDigitization; // Digitalisierungstiefe
  int m_iMaxDigit;     // Maximalwert Digitalisierung = 4095 bei 12 Bit
  int m_iVertBinIndex;  // Index des aktuellen vertikalen Binnings
  int m_iTrigMode;     // Trigger-Modus: 0 sequentiell
                       //                1 parallel
  int m_iDelayTime;    // Verzögerungszeit in ms
  int m_iIntegTime;    // Integrations- bzw. Belichtungszeit in ms
  int CCDWidth;     // Breite des CCD-Chips in Pixeln
  int CCDHeight;    // Höhe des CCD-Chips in Pixeln
  int m_iROIMode;      // ROI-Modus: 0 = volle Bildgröße,
                       //            1 = 1024 x 1024 Pixel auf CCD
                       //            2 =  512 x  512 Pixel  "   "
                       //            3 =  256 x  256 Pixel
                       //            4 =  128 x  128 Pixel
                       //            5 =   64 x   64 Pixel
                       //            6 =   32 x   32 Pixel
                       //            7 = selbstdefiniert
  int m_iROIPosX;      // X-Koordinate der linken oberen ROI-Ecke
  int m_iROIPosY;      // Y-Koordinate der linken oberen ROI-Ecke
  int m_iROIWidth;     // ROI-Breite auf CCD-Chip!
  int m_iROIHeight;    // ROI-Höhe   auf CCD-Chip!
                       // m_iROIPosX/Y, m_iROIWidth/Height müssen ganzzahlige
                       // Vielfache von 32 sein!
  int m_iSelfROIWidth;   // Breite der selbstdefinierten ROI
  int m_iSelfROIHeight;  // Höhe der selbstdefinierten ROI
  //int m_iLogMode;      // 0 = Lineare Bilddarstellung
                       // 1 = Logarithmische Darstellung
  int picWidth;     // Bildbreite auf CCD-Chip
  int picHeight;    // Bildhöhe   auf CCD-Chip
  int m_iBinningX;     // Horizontales Binning = 1, 2, 4 oder 8
  int m_iBinningY;     // Vertikales Binning   = 1, 2, 4, 8, 16 oder 32
  //int m_iLowThresh;    // unterer Schwellwert für Falschfarbendarstellung
  //int m_iHighThresh;   // oberer Schwellwert für Falschfarbendarstellung
  int NumberCh;     // Anzahl Farbkanäle

  int ROIX1,ROIX2,ROIY1,ROIY2;
  
  int iMode;  //Camera operation mode (for SET_COC)
  

  // ACHTUNG!: Die Variablen m_iCCDWidth/Height, m_iROIPosX/Y, m_iROIWidth/
  //           Height, m_iPicWidth/Height bezeichnen die entsprechenden Größen
  //           auf dem CCD-CHIP! Die Benutzeroberfläche der Anwendung zeigt je-
  //           doch die entsprechenden Größen PRO FARBKANAL an, d. h. zur
  //           Weiterverarbeitung in CCamDialog usw. muß entsprechend durch 2
  //           dividiert werden!

 word *currentFrame; //Buffer for raw data from camera.
 int boardNum;
};

#endif // !defined(AFX_CAM_H__6D9C46BA_17F0_11D3_9900_0000C0E169AB__INCLUDED_)
