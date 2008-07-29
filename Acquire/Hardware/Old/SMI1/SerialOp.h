// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * SerialOp.h: Schnittstelle für die Klasse CSerialOp.                     *
// * (C) 1999-2001 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am 10. Dezember  1999                                      *
// * 38. Änderung am 26. September 2001                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Basisklasse für die Klassen CStepOp und CPiezoOp zur Ansteuerung der seriel-
// len Schnittstellen COM1-4

#if !defined(AFX_SERIALOP_H__310CAB23_AD4B_11D3_995D_0000C0E169AB__INCLUDED_)
#define AFX_SERIALOP_H__310CAB23_AD4B_11D3_995D_0000C0E169AB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <string>
using namespace std;

#include <windows.h>

#define LPSTR char*

class CSerialOp
{
public:
	string GetControllerName();
	CSerialOp();
	virtual ~CSerialOp();

// Elementzugriffsfunktionen:
  void SetMessage(int iIndex, string strMessage){
    m_strMessage[iIndex] = strMessage;
  }
  string GetMessageKey(){return m_strMessageKey;}
  void SetPortNr(int iNr){m_iPortNr = iNr;}
  int GetPortNr()        {return m_iPortNr;}
  string GetPortName()  {return m_chPort[m_iPortNr - 1];}
  string GetPortName(int ip)  {return m_chPort[ip - 1];}
  bool GetControlReady() {return m_bControlOK;}
  void SetControlReady(bool bError){m_bControlOK = bError;}

  // Kommandos:
  void DisplayError(int iErr);         // Fehlermeldung ausgeben
  void DisplayError(int iErr, LPSTR sz,...); // Fehlermeldung im printf Format ausgeben
                                             // die Fehlermeldung dient als Format-String
  bool CloseConnection();              // Serielle Schnittstelle schliessen
  
  // Start USp    20.10.2002
  //void SetDocument(CDocument* pDoc){ m_pDoc = pDoc; };
  //CDocument* GetSMIDocument() {return m_pDoc;}
  // End   USp

  

protected:

	// C. Wagner, Anfang, weitere Variablen zur Speicherung der Hardware-Konfiguration
	string m_strPZT;
	string m_strController;
	// C. Wagner, Ende

	// Start USp    20.10.2002
	//CDocument* m_pDoc;
    // End   USp

  // Variablen/Attribute:
  string m_strDeviceName;             // device name für Error Messages
  string m_strMessageKey;             // Schlüsselwort für Messages-Datei
  string m_strMessage[100];           // Messages
  bool m_bControlOK;                   // = TRUE, falls keine Fehler aufge-
                                       //         treten
  int m_iPortNr;                       // Port-Nummer = 1..4
  string m_strVersion;                // enthält Antwort der Schrittmotor-
                                       // Steuerung auf Kommando "version"
  
  HANDLE m_hCOM;                       // Handle für Schnittstelle
  static const char* m_chPort[5];      // Bezeichner für Schnittstellen
  char* m_chInitComPath;               // Pfad für Initialisierungskommandos
  string m_strInitCom[32];            // enthalten Initialisierungskommandos
  int m_iComNumber;                    // Anzahl der Initialisierungskommandos
  DWORD m_dwBaudRate;                  // Baudraten
  BYTE  m_btByteSize;                  // Anzahl Bits pro Byte
  BYTE  m_btParity;                    // Parität
  BYTE  m_btStopBits;                  // Anzahl verwendeter Stop-Bits
  BYTE  m_btFlowCtrl;                  // für Hardware-Flow Control
  bool  m_bConnected;                  // = TRUE: Verbindung zu COMx steht

  string m_strInputBuffer;            // Pufferstring für eingehende Daten von
                                       // Piezosteuerung
  char m_pInputBuffer[4096];  // Eingangs-Puffer für Piezo-
                                       // Steuerung

  // ASCII-Definitionen:
  #define ASCII_XON  0x11
  #define ASCII_XOFF 0x13
  // Flow-Control Flags:
  #define FC_DTRDSR  0x01
  #define FC_RTSCTS  0x02
  #define FC_XONXOFF 0x04

  // -------------------------------------------------------------- Funktionen:
  bool LoadInitCom();                  // Initialisierungskommandos laden
  bool OpenConnection(int iCOMNr);     // Serielle Schnittstelle öffnen
  bool SetupConnection();              // Serielle Schnittstelle konfigurieren
  bool SendData(const char* chData);         // Daten auf Schnittstelle ausgeben
  int GetData();                       // Daten von Schnittstelle empfangen
  int WaitForData();                   // Wartet eine gewisse Zeit auf Daten
  int GetLength(const char* chString);       // Länge einer Char-Zeichenkette er-
                                       // mitteln  
  void Wait(int iTime);                // Wartet iTime ms
  float GetValue(string strNumber, int iValPos);  // Zahlenwert aus String
                                       // auslesen
  string GetString(string strNumber, int iValPos);  // analog zu GetValue(,)
  int GetStartPos(string strNumber, int iValPos);  // Anfang der Zahl suchen

  

};

#endif // !defined(AFX_SERIALOP_H__310CAB23_AD4B_11D3_995D_0000C0E169AB__INCLUDED_)
