// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * StepOp.cpp: Implementierung der Klasse CStepOp.                         *
// * (C) 1999-2001 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am 30. März      2000                                      *
// * 18. Änderung am 25. Juli      2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

//#include "stdafx.h"
//#include "stdafx.h"
// #include "SMIProjectDoc.h"
#include "StepOp.h"
#include "MsgStep.h"

#define bool bool

#include <strstream>
using namespace std;

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif


// =================================================== Konstruktion/Destruktion
CStepOp::CStepOp(){
  int I;
  // m_iPortNr    = 1;           // Default: Schrittmotorsteuerung an COM1
  // -------------------------------- Strings für Schrittmotorsteuerung:
  m_strDeviceName  = _MSG_STEP_DEVICE;
  // -------------------------------- Strings für (Fehler-)Meldungen:
  // Allgemeine Kommandos:
  m_strMessageKey  = _MSG_STEP_KEY;
  m_strMessage[0]  = _MSG_STEP_0;
  m_strMessage[1]  = _MSG_STEP_1;
  m_strMessage[2]  = _MSG_STEP_2;
  m_strMessage[3]  = _MSG_STEP_3;
  m_strMessage[4]  = _MSG_STEP_4;
  m_strMessage[5]  = _MSG_STEP_5;
  // Spezielle Kommandos für Schrittmotor-Steuerung:
  m_strMessage[10] = _MSG_STEP_10;
  m_strMessage[11] = _MSG_STEP_11;
  m_strMessage[12] = _MSG_STEP_12;
  m_strMessage[13] = _MSG_STEP_13;
  m_strMessage[14] = _MSG_STEP_14;
  m_strMessage[15] = _MSG_STEP_15;
  // ----------------------------------------------------------------
  // Parameter der Schnittstelle (= Mitglieder der Basisklasse CSerialOp)
  // setzen:
  m_bControlOK = FALSE;
  m_bConnected = FALSE;
  m_dwBaudRate = CBR_9600;
  m_btByteSize = 8;
  m_btParity   = NOPARITY;
  m_btStopBits = ONESTOPBIT;
  m_btFlowCtrl = FC_XONXOFF;

  // Parameter für Schrittmotoren:
  for(I = 0; I < 32; I++) m_strInitCom[I] = "";
  m_iComNumber          = 0;
  m_iExpect             = 0;
  m_iTryGetPos          = 0;
  m_iMaxTryGetPos       = 10;  // maximal 10mal versuchen, Position abzufragen
  m_strInputBuffer      = string("");
  m_bIsCalibrating      = FALSE;
  m_bIsSettingZInterval = FALSE;
  m_iCalibrationMode    = 0;
  m_bCalibrated         = FALSE;
  m_strVersion          = "";
  m_fPosX               = 0;
  m_fPosY               = 0;
  m_fPosZ               = 0;
  m_strPosX             = "";
  m_strPosY             = "";
  m_strPosZ             = "";
  m_fMaxX               = 0;  // Intervallgrenzen werden später gesetzt
  m_fMaxY               = 0;
  m_fMaxZ               = 0;
  m_fMinX               = 0;
  m_fMinY               = 0;
  m_fMinZ               = 0;
  m_strMinX             = "";
  m_strMaxX             = "";
  m_strMinY             = "";
  m_strMaxY             = "";
  m_strMinZ             = "";
  m_strMaxZ             = "";
  InitUserValues();
  m_fNullX              = 0;
  m_fNullY              = 0;
  m_fNullZ              = 0;
  m_fMinXS              = 0;
  m_fMaxXS              = 0;
  m_fMinYS              = 0;
  m_fMaxYS              = 0;
  m_fMinZS              = 0;
  m_fMaxZS              = 0;
  m_fPosXOSOut          = 40000;
  m_fPosYOSOut          = 25000;
  m_fPosXS[0]           = 0;
  m_fPosXS[1]           = 0;
  m_fPosYS[0]           = 0;
  m_fPosYS[1]           = 0;
  m_fPosZS[0]           = 0;
  m_fPosZS[1]           = 0;
  m_chInitComPath       = "StepInit.txt";
  m_chParPath           = "StepArea.dat";
}

CStepOp::~CStepOp(){
}



// =========================== Kommandos für Schrittmotor-Steuerung/Menübefehle
// ------------------------------------------------------------ Initialisierung
int CStepOp::Init(int iMode){
  // iMode = 1: Default-Werte verwenden
  //       = 2: andere/geladene Werte übernehmen, keine Kalibrierung durch-
  //            führen und Verfahrbereich nicht ermitteln
  // Alle an die Schrittmotor-Steuerung gesendeten Kommandos müssen mit einem
  // Leerzeichen (= ASCII 32) abgeschlossen werden.
  bool    bResult;
  string strCommand = "";

  string strLoadError = m_strMessage[10];
  string strVerError  = m_strMessage[11];
  string strQuestion  = m_strMessage[12];

  m_strInputBuffer = "";
  int I, iRes;

  if((iMode == 2) && (m_bControlOK == FALSE)) return -1;  // Liegt bereits beim
    // Laden eines Dokuments bzw. der Einstellungen ein Fehler vor, nicht ini-
    // tialisieren

  if(iMode < 2){  // Initialisierungs- und Kalibrierungsprozedur nur bei Pro-
    // grammstart oder ausdrücklicher Initialisierung (aus Hauptmenü) durch-
    // führen (also nicht bei Dokument-Laden):

    // Mauszeiger auf Warte-Symbol setzen:
    //HCURSOR lhCursor = AfxGetApp()->LoadStandardCursor(IDC_WAIT);
    //SetCursor(lhCursor);

    bResult = LoadInitCom();  // Initialisierungskommandos laden
    if(bResult == FALSE){
//		string openFileErrorMsg = strLoadError + "\nFile '" + m_chInitComPath + "'";
//	  ((CSMIProjectApp*)AfxGetApp())->MsgLog.ShowMsg(m_strDeviceName,openFileErrorMsg);
		DisplayError(10,m_chInitComPath);
//      AfxMessageBox(strLoadError);
      return -1;
    }//endif
    bResult = OpenConnection(m_iPortNr);  // Verbindung zu Port mit Schritt-
      // motor-Steuerung einrichten
    if(bResult == FALSE) return -1;
    Wait(500);  // besser kurz warten
    SendData(m_strInitCom[0].c_str());  // Steuerung in "Host-Mode" setzen

    // Auf evtl. ankommende Daten warten und diese abfangen:
    for(I = 0; I < 4; I++){  // 3 s warten
      Wait(1000);
      GetData();
      m_strInputBuffer = "";
    }//next I
    SendData(m_strInitCom[1].c_str());  // Befehlsstack leeren

    // Testen der Verbindung:
    SendData(m_strInitCom[2].c_str());  // "version "
    Wait(500);
    GetData();
    if(m_strInputBuffer.substr(0,4) != m_strVersion){
      strVerError += m_strInputBuffer.substr(0,4);
      // AfxMessageBox(strVerError);
      DisplayError(11);
      m_bControlOK = FALSE;
      CloseConnection();
      return -1;
    }//endif
    m_strInputBuffer = "";

    // Initialisierungskommandos senden:
    for(I = 3; I < m_iComNumber; I++) SendData(m_strInitCom[I].c_str());

    InitUserValues();  // Falls iMode = 0 oder 1, werden voreingestellte Werte
      // verwendet. Falls iMode = 2, wurde Initialisierung der Parameter schon
      // bei der Serialisierung mit geladenen Werten vorgenommen.

    // Mauszeiger wieder auf Standard-Pfeil zurücksetzen:
    //lhCursor = AfxGetApp()->LoadStandardCursor(IDC_ARROW);
    //SetCursor(lhCursor);
  }//endif

  // -------------------------------------- Vom Benutzer einstellbare Parameter
  SetMoveAccel(m_iMoveAccel);
  SetMoveSpeed(m_iMoveSpeed);
  SetJoystickSpeed(m_iJoystickSpeed);
  if(m_bJoystickStatus == TRUE){
    SendData("1 joystick ");  // Joystick einschalten
  }else{
    SendData("0 joystick ");
  }//endif

  // Kalibrierung durchführen und Verfahrbereich festlegen:
  if(iMode < 2){  // nur, falls initialisiert werden soll ohne, dass Dokument
    // geladen wurde
	  //iRes = MessageBox(NULL, strQuestion.c_str(), "Calibration ...", MB_YESNO);
	  iRes = IDYES;
  }else{
    iRes = IDNO;
  }//endif
  if(iRes == IDYES){
    iMode = 2;  // Voreinstellung = geladene Parameter übernehmen
    bResult = LoadAreaValues();
    if(bResult == FALSE) iMode = 1;  // falls Parameter nicht geladen werden
      // konnten, gesamtes physikalisches z-Intervall freigeben
    Calibrate(iMode);  // InitAreaValues() wird auch hier ausgeführt, aber in
      // ContIO()!
    return 1;
  }else{
    InitAreaValues();  // Bereichsvariablen initialisieren
    return 0;
  }//endif
}

// ------------------------- Vom Benutzer einstellbare Parameter initialisieren
void CStepOp::InitUserValues(){
  m_iMoveAccel      = 20;     // 0 < m_iMoveAccel < 100
  m_iMoveSpeed      = 5;      // 0 < m_iMoveSpeed < 11
  m_iJoystickSpeed  = 5;      // 0 < m_iJoystickSpeed < 11
  m_bJoystickStatus = FALSE;  
}

// -------------------------------- Verfahrbereich der Schrittmotoren ermitteln
int CStepOp::Calibrate(int iMode){
  // Es ist zu beachten, daß während der Kalibrierung keine Kommandos an die
  // Schrittmotor-Steuerung geschickt werden dürfen!
  m_fNullX = 0;  // absolute Position des aktuellen Nullpunkts ist nach Kali-
  m_fNullY = 0;  // brierung identisch mit absolutem Nullpunkt
  m_fNullZ = 0;
  SendData("0 joystick ");  // Während der Kalibrierung sollte der Joystick
    // ausgeschaltet sein!
  SendData("calibrate ");
  SendData("geterror ");
  m_bIsCalibrating = TRUE;
  m_iCalibrationMode = iMode;
  m_iExpect = 1;  // warte auf Rückmeldung des Befehls "geterror" nach 
    // "calibrate" => Wenn Rückmeldung eintrifft, ist Kommando "cal" beendet!
  return 0;
}

// ------------------------------------- Initialisierung der Bereichs-Variablen
void CStepOp::InitAreaValues(){
  int     iLength, iPos;
  bool    bFlag = FALSE;
  string strNumber;

  if(m_iExpect > 10){  // kontinuierliche Schrittmotor-Überwachung stört In-
    // Schrittmotor-Kommando "getlimit"
    StopContIO();
    bFlag = TRUE;
  }//endif
  SendData("getlimit ");
  Wait(500);  // Hier nicht WaitForData() verwenden!
  GetData();

  //HORRIBLE !!!!!!
  // [CR][LF] aus Pufferstring entfernen:
  /*iLength = m_strInputBuffer.GetLength();
  for(int I = 1; I <= 2; I++){
    iPos = m_strInputBuffer.Find("\n\0");
    if(iPos == -1){  // Keine Antwort oder falsche Version
      DisplayError(3);
      return;  // -1;
    }//endif
    m_strInputBuffer = m_strInputBuffer.Left(iPos - 1) + " " +
      m_strInputBuffer.Mid(iPos + 1, iLength - I);
  }//next I
  m_fMinX   = GetValue(m_strInputBuffer, 0);
  m_fMaxX   = GetValue(m_strInputBuffer, 1);
  m_fMinY   = GetValue(m_strInputBuffer, 2);
  m_fMaxY   = GetValue(m_strInputBuffer, 3);
  m_fMinZ   = GetValue(m_strInputBuffer, 4);
  m_fMaxZ   = GetValue(m_strInputBuffer, 5);
  m_strMinX = GetString(m_strInputBuffer, 0);
  m_strMaxX = GetString(m_strInputBuffer, 1);
  m_strMinY = GetString(m_strInputBuffer, 2);
  m_strMaxY = GetString(m_strInputBuffer, 3);
  m_strMinZ = GetString(m_strInputBuffer, 4);
  m_strMaxZ = GetString(m_strInputBuffer, 5);*/

  int sPos = 0;

  istrstream tok(m_strInputBuffer.c_str());
 
  /*m_strMinX = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMinX = atof(m_strMinX);
  m_strMaxX = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMaxX = atof(m_strMaxX);

  m_strMinY = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMinY = atof(m_strMinY);
  m_strMaxY = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMaxY = atof(m_strMaxY);

  m_strMinZ = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMinZ = atof(m_strMinZ);
  m_strMaxZ = m_strInputBuffer.Tokenize(" \n\0",sPos);
  m_fMaxZ = atof(m_strMaxZ);*/

  tok >> m_strMinX;
  m_fMinX = atof(m_strMinX.c_str());
  tok >> m_strMaxX;
  m_fMaxX = atof(m_strMaxX.c_str());
  tok >> m_strMinY;
  m_fMinY = atof(m_strMinY.c_str());
  tok >> m_strMaxY;
  m_fMaxY = atof(m_strMaxY.c_str());
  tok >> m_strMinZ;
  m_fMinZ = atof(m_strMinZ.c_str());
  tok >> m_strMaxZ;
  m_fMaxZ = atof(m_strMaxZ.c_str());

  m_strInputBuffer = "";
  if(bFlag == TRUE) StartContIO();
}

// ------------------------------ Gesamten physikalischen Verfahrbereich öffnen
int CStepOp::SetArea(bool bJoystick){
  int iResult;
  iResult = SetArea(-m_fNullX, m_fMaxPhysX - m_fNullX, -m_fNullY,
    m_fMaxPhysY - m_fNullY, -m_fNullZ, m_fMaxPhysZ - m_fNullZ, bJoystick);
  return iResult;
}

// ------------------------------------------------------ Verfahrbereich setzen
int CStepOp::SetArea(float fMinX, float fMaxX, float fMinY, float fMaxY,
  float fMinZ, float fMaxZ, bool bJoystick)
{
  char strCommand[255];
  string strError = m_strMessage[13];
  bool bFlag = FALSE;
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return -1;
  }//endif
  // if((m_fPosX <= fMinX) || (m_fPosX >= fMaxX) || (m_fPosY <= fMinY) ||
  //   (m_fPosY >= fMaxY) || 
  if((m_fPosZ <= fMinZ) || (m_fPosZ >= fMaxZ)){
	  DisplayError(13);
	  // AfxMessageBox(strError);
    return -1;
  }//endif
  if(m_iExpect > 10){  // kontinuierliche Schrittmotor-Überwachung stört
    // Schrittmotor-Kommando "setlimit"
    StopContIO();
    bFlag = TRUE;
  }//endif
  // m_fMinX = fMinX;  // Vorerst Verfahrbereich nur in z-Richtung einschränken
  // m_fMaxX = fMaxX;
  // m_fMinY = fMinY;
  // m_fMaxY = fMaxY;
  m_fMinZ = fMinZ;
  m_fMaxZ = fMaxZ;
  // m_strMinX.Format("%6.2f", fMinX);
  // m_strMaxX.Format("%6.2f", fMaxX);
  // m_strMinY.Format("%6.2f", fMinY);
  // m_strMaxY.Format("%6.2f", fMaxY);
  char buf[20];
  sprintf(buf,"%6.2f", fMinZ);
  m_strMinZ = string(buf);
  sprintf(buf,"%6.2f", fMaxZ);
  m_strMaxZ = string(buf);
  if(m_bCalibrated == TRUE){  // Intervallgrenzen nur dann für Seriali-
    // sierung speichern, falls Schrittmotoren kalibriert wurden.
    m_fMinXS = fMinX + m_fNullX;
    m_fMaxXS = fMaxX + m_fNullX;
    m_fMinYS = fMinY + m_fNullY;
    m_fMaxYS = fMaxY + m_fNullY;
    m_fMinZS = fMinZ + m_fNullZ;
    m_fMaxZS = fMaxZ + m_fNullZ;
    SaveAreaValues();
  }//endif
  sprintf(strCommand,"%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f setlimit ",
    m_fMinX, m_fMinY, fMinZ, m_fMaxX, m_fMaxY, fMaxZ);
  SendData(strCommand);
  if(bJoystick == TRUE){
    SendData("1 joystick ");
  }else{
    if(m_bJoystickStatus == FALSE) SendData("0 joystick ");
  }//endif
  if(bFlag == TRUE) StartContIO();
  return 0;
}

// ----------------------------------------------------------------------------
// ---------------------------------- Kontinuierliche Positionsabfrage beginnen
int CStepOp::StartContIO(){
  if(m_bControlOK == FALSE){
    // DisplayError(1);
    return -1;
  }//endif
  SendData("pos ");  // erstes Kommando für kontinuierliche Positionsbestimmung
    // senden
  m_iExpect = 11;
  m_iTryGetPos  = 0;
  return 0;
}

// ------------------ Kontinuierlichen Kommunikation mit Schrittmotor-Steuerung
void CStepOp::ContIO(){
  // Diese Routine wird ca. alle 50 ms vom Timer der Ansicht aufgerufen und
  // verwaltet die kontinuierliche Kommunikation mit der Schrittmotor-Steuerung.
  int iInputNumber, iFindPos;

  if(m_bControlOK == FALSE) return;
  if(m_iExpect == 11){
    m_iTryGetPos++;
    if(m_iTryGetPos > m_iMaxTryGetPos){  // maximale Anzahl Positionsabfragen
      // überschritten => Fehler:
      DisplayError(3);
      return;
    }//endif
  }//endif
  iInputNumber = GetData();
  if(iInputNumber == 0) return;  // Wenn keine Daten anliegen => Aussteigen

  iFindPos = m_strInputBuffer.find("\n\0");  // Nach Zeichenkette
    // [CR][LF] (= Abschlußzeichen der gesendeten Daten) suchen
  if(iFindPos == -1) return;  // nicht gefunden => Aussteigen
  
  m_strInputBuffer = m_strInputBuffer.substr(0,iFindPos);  // Die letzten
    // zwei Zeichen ("\n\0") abschneiden

  // Fehlermeldung ausgeben, falls Eingangspuffer überläuft?

  switch(m_iExpect){
  case 1:  // -------------------- Befehl "geterror" nach "calibrate"
    m_strInputBuffer = "";  // Rückmeldung verarbeitet => Pufferstring löschen
    SendData("rm ");
    SendData("geterror ");
    m_iExpect = 2;
    break;
  case 2:  // --- Objektträger aus Zwischenraum der Objektive heraus-
    // - bewegen und in Mitte des festzulegenden z-Intervalls bewegen
    m_strInputBuffer = "";
    InitAreaValues();
    m_fMaxPhysX = m_fMaxX;  // max. phys. Verfahrbereich merken
    m_fMaxPhysY = m_fMaxY;
    m_fMaxPhysZ = m_fMaxZ;
    if(m_iCalibrationMode == 1){  // gesamten phys. Verfahrbereich merken
      m_fMinXS = m_fMinX;
      m_fMaxXS = m_fMaxX;
      m_fMinYS = m_fMinY;
      m_fMaxYS = m_fMaxY;
      m_fMinZS = m_fMinZ;
      m_fMaxZS = m_fMaxZ;
      SaveAreaValues();
    }//endif
    MoveTo(m_fMaxX - 60000, (m_fMinY + m_fMaxY)/2, (m_fMinZS + m_fMaxZS)/2);
    // MoveTo((m_fMinX + m_fMaxX)/2, (m_fMinY + m_fMaxY)/2, (m_fMinZS + m_fMaxZS)/2);
    Wait(200);
    SendData("pos ");  // erstes Kommando für kontinuierliche Positionsbestim-
      // mung senden
    m_iExpect = 3;
    break;
  case 3:  // ----------- z-Intervall setzen und Kalibrierung beenden
    ExtractPos();
    if(m_fPosZ > ((m_fMinZS + m_fMaxZS)/2 + 10)){  // falls Endposition noch
      // nicht erreicht:
      SendData("pos ");  // Position weiter abfragen
      m_strInputBuffer = "";
    }else{  // Position erreicht, z-Intervall kann gesetzt werden:
      m_bIsCalibrating = FALSE;
      m_bCalibrated = TRUE;
      if(m_iCalibrationMode == 2)
        SetArea(m_fMinXS, m_fMaxXS, m_fMinYS, m_fMaxYS, m_fMinZS, m_fMaxZS,
          m_bJoystickStatus);  // Verfahrbereich mit serialisierten (ge-
          // speicherten) Grenzen setzen
      m_iExpect = 0;  // Es wird nichts mehr erwartet.
	  StartContIO();
    }//endif
    break;
  case 11:  // ----------------------------------------- Befehl "pos"
    ExtractPos();
    SendData("pos ");
    m_strInputBuffer = "";  
    m_iTryGetPos = 0;
    break;
  }//endswitch
}

// ----------- Kontinuierliche Kommunikation mit Schrittmotor-Steuerung beenden
void CStepOp::StopContIO(){
  Wait(500);
  GetData();  // Puffer leeren
  m_strInputBuffer = "";
  m_iExpect = 0;  // Es wird nichts mehr erwartet.
}

// -------------------------------------------------------------- Verfahrbefehl
void CStepOp::MoveTo(float fPosX, float fPosY, float fPosZ){
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  // SendData("0 joystick ");
  //Wait(100);
  Wait(10);
  char strCommand[255];
  sprintf(strCommand,"%6.2f %6.2f %6.2f move ", fPosX, fPosY, fPosZ);
  SendData(strCommand);
  //Wait(400);
  Wait(10);
}

void CStepOp::MoveRel(float fDeltaX, float fDeltaY, float fDeltaZ){
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  // SendData("0 joystick ");
  //Wait(100);
  Wait(10);
  char strCommand[255];
  sprintf(strCommand,"%6.2f %6.2f %6.2f rmove ", fDeltaX, fDeltaY, fDeltaZ);
  SendData(strCommand);
  //Wait(400);
  Wait(10);
}

// -------------------------------------------- Schrittmotor-Aktionen abbrechen
void CStepOp::Break(){
  SendData("\003 ");  // 003 = ASCII-Code für [Ctrl] + [C].
  Wait(500);
  SendData("clear ");  // Befehlsstack leeren
  Wait(100);  // Eingangs-Puffer leeren
  GetData();
  if(m_iExpect < 5){  // Falls gerade Kalibrierung läuft
    SendData("\003 ");  // Zur Sicherheit nochmal
    Wait(500);
    GetData();
    m_bIsCalibrating = FALSE;
    m_bControlOK     = FALSE;
    m_iExpect        = 0;  // Nach Abbruch der Kalibrierung wird nichts mehr
      // erwartet
  }//endif
  if(m_iExpect == 11) SendData("pos ");
  Wait(200);
  if(m_bJoystickStatus == TRUE) SendData("1 joystick ");
  m_strInputBuffer = "";
}

// ---------------- Aktuelle Position zum Nullpunkt/Koordinaten-Ursprung setzen
void CStepOp::SetNull(){
  if(m_bIsCalibrating == TRUE) return;
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  m_fNullX += m_fPosX;  // Ursprungskoordinaten bezüglich unterer End-
  m_fNullY += m_fPosY;  // schalter merken
  m_fNullZ += m_fPosZ;
  SendData("0 0 0 setpos ");
  InitAreaValues();
  SendData("pos ");  // fortlaufende Kommunikation muß weitergehen
}

// -------------------------------------------------- Position iIndex speichern
void CStepOp::SavePos(int iIndex){
  m_fPosXS[iIndex] = m_fNullX + m_fPosX;
  m_fPosYS[iIndex] = m_fNullY + m_fPosY;
  m_fPosZS[iIndex] = m_fNullZ + m_fPosZ;
}

// --------------------------------------------------- Position iIndex anfahren
void CStepOp::MoveToPos(int iIndex){
  MoveTo(-m_fNullX + m_fPosXS[iIndex], -m_fNullY + m_fPosYS[iIndex],
    -m_fNullZ + m_fPosZS[iIndex]);
}

// -------------------------------------------------- Joystick ein-/ausschalten
void CStepOp::SetJoystickOnOff(){
  if(m_bIsCalibrating == TRUE) return;
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  m_bJoystickStatus = !m_bJoystickStatus;
  if(m_bJoystickStatus == TRUE){  // Joystick einschalten
    SendData("1 joystick ");
  }else{  // Joystick ausschalten
    SendData("0 joystick ");
  }//endif
}

//void CStepOp::JoystickOn(){
//  if(m_bIsCalibrating == TRUE) return;
//  if(m_bControlOK == FALSE){
//    DisplayError(1);
//    return;
//  }//endif
//  m_bJoystickStatus = !m_bJoystickStatus;
//  if(m_bJoystickStatus == TRUE){  // Joystick einschalten
//    SendData("1 joystick ");
//  }else{  // Joystick ausschalten
//    SendData("0 joystick ");
//  }//endif
//}
//
//void CStepOp::JoystickOff(){
//  if(m_bIsCalibrating == TRUE) return;
//  if(m_bControlOK == FALSE){
//    DisplayError(1);
//    return;
//  }//endif
//  m_bJoystickStatus = !m_bJoystickStatus;
//  if(m_bJoystickStatus == TRUE){  // Joystick einschalten
//    SendData("1 joystick ");
//  }else{  // Joystick ausschalten
//    SendData("0 joystick ");
//  }//endif
//}

// -------------------------------------------- Joystick-Geschwindigkeit setzen
void CStepOp::SetJoystickSpeed(int iSpeed){
  ExecuteCommand(iSpeed, string("setjoyspeed"));
  m_iJoystickSpeed = iSpeed;
}

// ----------------------------------- Beschleunigung für Verfahrbefehle setzen
void CStepOp::SetMoveAccel(int iAccel){
  ExecuteCommand(iAccel, string("setaccel"));
  m_iMoveAccel = iAccel;
}

// --------------------------------- Geschwindigkeit von Verfahrbefehlen setzen
void CStepOp::SetMoveSpeed(int iSpeed){
  ExecuteCommand(iSpeed, string("setvel"));
  m_iMoveSpeed = iSpeed;
}

// ------------ Objektträger in/aus Objektiv-Zwischenraum herein/heraus bewegen
void CStepOp::MoveOSInOut(){
}

// ================================================= Interne (Hilfs-)Funktionen
// -------------------------------------- Kommando für Schrittmotoren ausführen
void CStepOp::ExecuteCommand(int iNumber, string strCommand){
  // Diese Funktion führt einfache Kommandos mit einem Parameter für die
  // Schrittmotor-Steuerung aus.
  char strNumber[20];
  if(m_bIsCalibrating == TRUE) return;
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  sprintf(strNumber,"%2d", iNumber);
  strCommand = string("") + strNumber + " " + strCommand + " ";
  SendData(strCommand.c_str());
}

// --------------- Aktuelle Schrittmotor-Position aus Puffer-String extrahieren
void CStepOp::ExtractPos()
{
	//HORRIBLE !!! String tokenisation & functions such as atof exist & are well docced
	//Will try and rewrite using class library
	/*m_fPosX   = GetValue(m_strInputBuffer, 0);
	m_fPosY   = GetValue(m_strInputBuffer, 1);
	m_fPosZ   = GetValue(m_strInputBuffer, 2);
	m_strPosX = GetString(m_strInputBuffer, 0);
	m_strPosY = GetString(m_strInputBuffer, 1);
	m_strPosZ = GetString(m_strInputBuffer, 2);*/

	int pos = 0;

	istrstream tok(m_strInputBuffer.c_str());

	/*m_strPosX = m_strInputBuffer.Tokenize(" ",pos);
	m_fPosX = atof(m_strPosX);
	m_strPosY = m_strInputBuffer.Tokenize(" ",pos);
	m_fPosY = atof(m_strPosY);
	m_strPosZ = m_strInputBuffer.Tokenize(" ",pos);
	m_fPosZ = atof(m_strPosZ);*/

	tok >> m_fPosX >> m_fPosY >> m_fPosZ;
}

// ----------------------------------------- Parameter für Verfahrbereich laden
bool CStepOp::LoadAreaValues(){
  FILE          *cfFile;
  //CFileException cfeException;
  bool           bResult;
  int            iSize;
  string        strError;
  
  strError = m_strMessage[14];
  iSize = sizeof(float);

  cfFile= fopen(m_chParPath, "r");
  bResult = (cfFile != NULL);
  if(bResult == TRUE){  // Datei erfolgreich geöffnet
    try{
      fread((void*)&m_fMinXS, 1, iSize, cfFile);
      fread((void*)&m_fMaxXS, 1, iSize, cfFile);
      fread((void*)&m_fMinYS, 1, iSize, cfFile);
      fread((void*)&m_fMaxYS, 1, iSize, cfFile);
      fread((void*)&m_fMinZS, 1, iSize, cfFile);
      fread((void*)&m_fMaxZS, 1, iSize, cfFile);
    }catch(void* err){  // Dateifehler aufgetreten
      //pFileError->Delete();
      bResult = FALSE;
    }//endcatch
    fclose(cfFile);
  }//endif
  if(bResult == FALSE) DisplayError(14); // AfxMessageBox(strError);
  return bResult;
}

// ------------------------------------- Parameter für Verfahrbereich speichern
bool CStepOp::SaveAreaValues(){
  FILE          *cfFile;
  //CFileException cfeException;
  bool           bResult;
  string        strError;
  int            iSize;
  
  strError = m_strMessage[15];
  iSize = sizeof(float);

  cfFile = fopen(m_chParPath, "w");
  bResult = (cfFile != NULL);
  if(bResult == TRUE){  // Datei erfolgreich geöffnet
    try{
      fwrite((void*)&m_fMinXS, iSize, 1, cfFile);
      fwrite((void*)&m_fMaxXS, iSize, 1, cfFile);
      fwrite((void*)&m_fMinYS, iSize, 1, cfFile);
      fwrite((void*)&m_fMaxYS, iSize, 1, cfFile);
      fwrite((void*)&m_fMinZS, iSize, 1, cfFile);
      fwrite((void*)&m_fMaxZS, iSize, 1, cfFile);
    }catch(void * err){  // Dateifehler aufgetreten
      //pFileError->Delete();
      bResult = FALSE;
    }//endcatch
    fclose(cfFile);
  }//endif
  if(bResult == FALSE) DisplayError(15); // AfxMessageBox(strError);
  return bResult;
}
