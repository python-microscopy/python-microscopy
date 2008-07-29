// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * PiezoOp.cpp: Implementierung der Klasse CPiezoOp.                       *
// * (C) 1999/2000 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am 30. März      2000                                      *
// * 10. Änderung am 26. September 2001                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#include "stdafx.h"
#include "stdafx.h"
// #include "SMIProjectDoc.h"
#include "PiezoOp.h"
#include "MsgPiezo.h"

#define bool bool

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif


// =================================================== Konstruktion/Destruktion
CPiezoOp::CPiezoOp(){
  m_strDeviceName  = _MSG_PIEZO_DEVICE;
  // m_iPortNr = 2;
  // -------------------------------- Strings für (Fehler-)Meldungen:
  // Allgemeine Kommandos:
  m_strMessageKey  = _MSG_PIEZO_KEY;
  m_strMessage[0]  = _MSG_PIEZO_0;
  m_strMessage[1]  = _MSG_PIEZO_1;
  m_strMessage[2]  = _MSG_PIEZO_2;
  m_strMessage[3]  = _MSG_PIEZO_3;
  m_strMessage[4]  = _MSG_PIEZO_4;
  m_strMessage[5]  = _MSG_PIEZO_5;
  // Spezielle Kommandos für Piezosteuerung:
  m_strMessage[10] = _MSG_PIEZO_10;

  // ----------------------------------------------------------------
  // Parameter für die Schnittstelle (= Mitglieder der Basisklasse CSerialOp):
  m_bControlOK = FALSE;
  m_bConnected = FALSE;
  m_dwBaudRate = CBR_9600;
  m_btByteSize = 8;
  m_btParity   = NOPARITY;
  m_btStopBits = ONESTOPBIT;
  m_btFlowCtrl = FC_XONXOFF;

  // Parameter für Piezo-Steuerung:
  for(int I = 0; I < 32; I++) m_strInitCom[I] = "";
  m_iComNumber     = 0;
  m_strInputBuffer = string("");
  m_bTimeOut       = FALSE;
  m_iTimeCounter   = 0;
  m_fPos[0]        = 0;
  m_fPos[1]        = 0;
  m_fPos[2]        = 0;
  //InitUserValues();
  
  m_iChannelObject = 2;
  m_iChannelPhase  = 1;  // oder = 3

  //m_fHardMin[0]    = 0;    // Kanal 1
  //m_fHardMax[0]    = 100;
  //m_fHardMin[1]    = 0;    // Kanal 2
  //m_fHardMax[1]    = 100;
  //m_fHardMin[2]    = 0;    // Kanal 3
  //m_fHardMax[2]    = 10;

  

  m_chInitComPath  = "PiezoInit.txt"; // geändert C. Wagner
}

CPiezoOp::~CPiezoOp(){
SetAllToNull(); //C. Wagner
}

// ------------------------------------------------------------- Serialisierung


// =================================== Kommandos für Piezosteuerung/Menübefehle
// ---------------------------------------------- Piezosteuerung initialisieren
int CPiezoOp::Init(int iMode){
  // iMode = 1: Default-Werte verwenden
  //       = 2: andere/geladene Werte übernehmen
  // Alle an die Piezosteuerung gesendeten Kommandos müssen mit mit LineFeed
  // (= ASCII 10 = Hex 00A) abgeschlossen werden!
  
  bool bResult;
  float   fResult;
  m_strInputBuffer = "";	

  if((iMode == 2) && (m_bControlOK == FALSE)) return -1;  // Liegt bereits beim
    // Laden eines Dokuments bzw. der Einstellungen ein Fehler vor, nicht ini-
    // tialisieren

  if(iMode < 2){  // Initialisierungs- und Kalibrierungsprozedur nur bei Pro-
    // grammstart oder ausdrücklicher Initialisierung (aus Hauptmenü) durch-
    // führen (also nicht bei Dokument-Laden):
    bResult = LoadInitCom();  // Port-Nr. laden

    if(bResult == FALSE){
//      AfxMessageBox(m_strMessage[10]);
//		string openFileErrorMsg = m_strMessage[10] + "\nFile '" + m_chInitComPath + "'";
//	  ((CSMIProjectApp*)AfxGetApp())->MsgLog.ShowMsg(m_strDeviceName,openFileErrorMsg);
		DisplayError(10,m_chInitComPath);
      return -1;
    }//endif
    bResult = OpenConnection(m_iPortNr);  // Verbindung zu Port mit Piezo-
      // steuerung einrichten
    if(bResult == FALSE) return -1;

    if (m_strController=="E500.00")
    {
	  m_iChannelPhase=3;
	  m_iChannelObject=2;	

	  m_fHardMin[0]    = 0;    // Kanal 1
      m_fHardMax[0]    = 100;
      m_fHardMin[1]    = 0;    // Kanal 2
      m_fHardMax[1]    = 100;
      m_fHardMin[2]    = 0;    // Kanal 3
      m_fHardMax[2]    = 10;

      InitChannels();  // Initialisierungskommandos senden
      // Hier noch weitere Initialisierungsbefehle z. B. zur Genauigkeit senden?
      InitUserValues();
    }//endif

	if (m_strController=="E255.60")
    {
	  Calibrate(); // Kalibrierungsdaten senden
      m_iChannelPhase=1; // bei E255.60 ist der Phasenspiegel an Kanal 1 angeschlossen
	  m_iChannelObject=3; // bei E255.60 ist der Objektverschiebetisch an Kanal 3 angeschlossen
	  InitUserValues();

	  m_fHardMin[0] = 0;
	  m_fHardMax[0] = 6; // Bei E255.60 max. Phasenverschiebung 8 Mikrometer
	  m_fHardMin[1] = 0; 
	  m_fHardMax[1] = 100;
	  m_fHardMin[2] = 0;
      m_fHardMax[2] = 100;
	}

	
  }//endif

  MoveTo(m_iChannelObject, (m_fMin[m_iChannelObject - 1] +
    m_fMax[m_iChannelObject - 1])/2, FALSE);
  Wait(100); // kurz warten, sonst wird der folgende Befehl bei E255.60 nicht ausgeführt
  MoveTo(m_iChannelPhase, (m_fMin[m_iChannelPhase - 1] +
    m_fMax[m_iChannelPhase - 1])/2, FALSE);
    // Piezoaktuatoren in die Mitte der spezifizierten Intervalle bewegen
  Wait(100);  // kurz warten, sodass Piezos Endposition erreicht haben
  fResult = GetOnePosition(1);  // prüfen, ob Verbindung steht und Positionen
    // ermitteln
  if(fResult != -1) fResult = GetOnePosition(2);
  if(fResult != -1) fResult = GetOnePosition(3);

  if (m_strController=="E500.00" && m_bExternControl == TRUE) SendData("DEV:CONT LOC\x00A");
  return 0;
}

// ------------------------- Vom Benutzer einstellbare Parameter initialisieren
void CPiezoOp::InitUserValues(){
  m_bExternControl = FALSE;
  
  m_fMin[0]        = 0;      // Hier bitte nur ganze Zahlen einsetzen!
  m_fMin[1]        = 0;
  m_fMin[2]        = 0;
  
  if (m_strController=="E255.60") 
  {
    m_fMax[0]=5.0;
    m_fMax[1]=40.0;
    m_fMax[2]=40.0;
  }
  else
  {
	m_fMax[0]        = 40.00;
    m_fMax[1]        = 40.00;
    m_fMax[2]        = 10.00;
  }//endif
}

// --------------------------- Kontinuierliche Kommunikation mit Piezosteuerung
void CPiezoOp::ContIO(){
  // Diese Routine wird ca. alle 100 ms vom Timer der Ansicht aufgerufen.
  // m_iTimeCounter++;
  // if(m_iTimeCounter > 1){
  m_bTimeOut     = FALSE;
  //   m_iTimeCounter = 0;
  // }//endif
}

// --------------------------------- Positionierbefehl für Piezotisch ausführen
void CPiezoOp::MoveTo(int iChannel, float fPos, bool bTimeOut){
  // iChannel = 1..3: Kanalnummer
  // fPos     = x [µm]: Zielkoordinate
  // bTimeOut = TRUE: Befehl nur bei m_bTimeOut = FALSE ausführen
  if((m_bTimeOut == TRUE) && (bTimeOut == TRUE)) return;
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  iChannel--;  // intern iChannel = 0..2
  // Grenzen dürfen nicht überschritten werden:
  if(fPos < m_fHardMin[iChannel]) fPos = m_fHardMin[iChannel];
  if(fPos > m_fHardMax[iChannel]) fPos = m_fHardMax[iChannel];
  char strCommand[255];

  if (m_strController=="E500.00")
  {
  sprintf(strCommand,"INST:SEL Ch%1d\x00A",iChannel + 1);  // Kanal auswählen
  SendData(strCommand);
  //strCommand = "";
  sprintf(strCommand, "SOUR:POS %3.2f\x00A", fPos);
  SendData(strCommand);
  }//endif
  
  // C. Wagner, Anfang
  int imV;
  if (m_strController=="E255.60")
  {
	imV=MikroToVolt(iChannel+1, fPos);
	sprintf(strCommand,"%dSO%d\r", iChannel+1, imV);  
    SendData(strCommand);
	Wait(50); // notwendig, sonst beeinflusst Phasen-Kanal den Objekt-Kanal
  }//endif
  // C. Wagner, Ende

  m_bTimeOut = TRUE;  // Nächster Aufruf dieser Funktion erst nach erneutem
    // Timeraufruf von PiezoContIO()! Sonst können zu viele Kommandos auf
    // einmal kommen und die Perfomance des Programms, insbes. der Kamera-Life-
    // Vorschau beeinträchtigen.
  m_fPos[iChannel] = fPos;  // Position merken
}

// ---------------------------------------------------------- fragt Position ab
float CPiezoOp::GetOnePosition(int iChannel){
  char strCommand[255];
  iChannel--;

  if (m_strController=="E500.00")
  {
  sprintf(strCommand,"INST:SEL Ch%1d\x00A", iChannel + 1);  // Kanal auswählen
  SendData(strCommand);
  SendData("MEAS:POS?\x00A");  // Position abfragen
  Wait(200);
  int iInputNumber = GetData();

  int iFindPos = m_strInputBuffer.find("\x00A");  // Nach Zeichenkette
    // \x00A (= Abschlußzeichen der gesendeten Daten = ASCII(10)) suchen
  if(iFindPos == -1){
    DisplayError(3);
    return -1;  // nicht gefunden => Aussteigen
  }//endif
  m_strInputBuffer.resize(iFindPos);  // Das letzte Zeichen
    // ("\x00A") abschneiden
  m_fPos[iChannel] = GetValue(m_strInputBuffer, 0);  // Position merken
  m_strInputBuffer = "";
  }//endif

  if (m_strController=="E255.60")
  {
	m_strInputBuffer = "";
    sprintf(strCommand,"%dTO\r", iChannel + 1); // Spannung von Kanal iChannel+1 anfordern
    SendData(strCommand);
    Wait(200);

	GetData();
	int iPosO = m_strInputBuffer.find("O");
	if(iPosO == -1){
    DisplayError(3);
    return -1;  // nicht gefunden => Aussteigen
    }//endif
	m_strInputBuffer = m_strInputBuffer.substr(iPosO+1, 6); // mV aus Gesamtstring extrahieren
	m_fPos[iChannel] = VoltToMikro(iChannel+1, atoi(m_strInputBuffer.c_str()));
	m_strInputBuffer = "";
  }//endif


  return m_fPos[iChannel];
}

// ------------------------ Externe Steuerung ein-/ausschalten (Piezosteuerung)
void CPiezoOp::SetExtCtrlOnOff(){
  
  if(m_bControlOK == FALSE){
    DisplayError(1);
    return;
  }//endif
  m_bExternControl = !m_bExternControl;
  if(m_bExternControl == TRUE){  // Kontrolle an Frontpanel übergeben
    SetAllToNull();  // Alle Kanäle auf U = Pos = 0 setzen, um Spannungs-
      // spitzen beim Wiedereinschalten der Regelung zu vermeiden!
  }else{  // Computer übernimmt Kontrolle
    InitChannels();
  }//endif
  Wait(1000);  // Warten, bis Piezoaktuatoren stationäre Position erreicht
    // haben  
}

// -------- Kanäle der Piezosteuerung auf Closed-Loop-Betrieb(Position-Control)
// ----------------------------------------------------------------- einstellen
void CPiezoOp::InitChannels(){
  if(m_strController=="E500.00")
  {
  if(m_bControlOK == FALSE) return;
  SendData("DEV:CONT REM\x00A");  // Steuerung mitteilen, dass Computer die
    // Kontrolle übernimmt (Remote-Control).
  Wait(200);
  SendData("INST:SEL Ch1\x00A");  // Kanal 1 auswählen
  SendData("DEV:SERV ON\x00A");  // Kanal 1 im Closed-Loop-Modus (Position-Con-
    // trol) betreiben
  SendData("INST:SEL Ch2\x00A");  // Das Gleiche mit Kanal 2
  SendData("DEV:SERV ON\x00A");   // und 3
  SendData("INST:SEL Ch3\x00A");
  SendData("DEV:SERV ON\x00A");
  }//endif
}

// -------------- Alle Piezo-Kanäle auf 0 setzen und Remote-Control ausschalten
void CPiezoOp::SetAllToNull(){
  // Durch das Setzen aller Kanäle auf U = 0 V und Pos = 0 µm wird die durch
  // das Setzen der Position auf die zuletzt eingestellten Werte beim Wieder-
  // einschalten der Remote-Control verursachte heftige Bewegung und das damit
  // verbundene starke Klick-Geräusch vermieden!

  if(m_bControlOK == FALSE) return;
  char strCommand[80];
  
  if(m_strController=="E500.00")
  {
  for(int I = 1; I <= 3; I++){
    //strCommand = "";
    sprintf(strCommand,"INST:SEL Ch%1d\x00A", I);  // Kanal auswählen
    SendData(strCommand);
    SendData("SOUR:POS 0.0\x00A");
    SendData("DEV:SERV OFF\x00A");  // Open-Loop-Modus einstellen
    SendData("SOUR:VOLT 0.0\x00A");
  }//next I
  Wait(500);
  SendData("DEV:CONT LOC\x00A");  // LOC = Local Control
  }//endif
	
  
  if(m_strController=="E255.60")
  {
	SendData("SO0\r"); // Spannung aller Kanäle auf Null setzen
    Wait(500);
  }//endif
}

// C. Wagner, Anfang
int CPiezoOp::Calibrate()
{
   //Kalibrierungsdaten an Steuerung senden
	SendData("1DO-3, 2DO14, 3DO-1, 1DG16234, 2DG16190, 3DG16255\r");
	//SendData("2DO14\r");
	//SendData("3DO-1\r");
	//SendData("1DG16234\r");
	//SendData("2DG16190\r");
	//SendData("3DG16255\r");

    return 0;
}

int CPiezoOp::MikroToVolt(int iChannel, float fMikro)
{
	// Berechnung der Spannung in mV aus entsprechender Verschiebung in um
	// für P-915.723
	if (iChannel==m_iChannelObject) return (int)(((fMikro-0.055)/19.999)*1000.0);

	// Umrechnung für Phasenspiegel (P-770.10)
	if (iChannel==m_iChannelPhase) return (int)(((fMikro+0.033)/0.702)*1000.0);
	return 0;
}


float CPiezoOp::VoltToMikro(int iChannel, int iVolt)
{	
   // Berechnung der Auslenkung in um bei eingelesener Spannung in mV für 
   // P-915.723
   if (iChannel==m_iChannelObject) return (float)(0.019999*iVolt+0.055);

   // Umrechnung für Phasenspiegel (P-770.10)
   if (iChannel==m_iChannelPhase) return (float)(0.000702*iVolt-0.033);   
   return 0;
}

string CPiezoOp::GetFirmware()
{
	if (m_strController=="E255.60") return ("");
    return ("Firmware Version 3.0");
}

string CPiezoOp::GetHardRange()
{
	char HardRange[256];
	sprintf(HardRange,"%s (0..%d):", _STR_PIEZO_0, (int)m_fHardMax[m_iChannelPhase-1]);

	return string(HardRange);
}


string CPiezoOp::GetRangeError()
{

	// Fehlermeldung für Bereichsüberschreitung zur Verfügung stellen

	char Error[256];
	sprintf(Error,_MSG_PIEZO_11,
    (int) m_fHardMin[m_iChannelPhase-1], (int) m_fHardMax[m_iChannelPhase-1]); 

	return string(Error);
}
// C. Wagner, Ende