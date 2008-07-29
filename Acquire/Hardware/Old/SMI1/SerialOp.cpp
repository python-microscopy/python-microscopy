// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * SerialOp.cpp: Implementierung der Klasse CSerialOp.                     *
// * (C) 1999-2001 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am 10. Dezember  1999                                      *
// * 50. Änderung am 20. Oktober   2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#include "stdafx.h"
#include "stdafx.h"
#include "SerialOp.h"
#include <windows.h>

//#include "SMIProjectDoc.h"
#include <io.h> 
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#define bool bool

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

// =================================================== Konstruktion/Destruktion
CSerialOp::CSerialOp(){
	m_iPortNr = 5;
} 

CSerialOp::~CSerialOp(){
}

// --------------------------------------------------------- Weitere Konstanten
const char* CSerialOp::m_chPort[5] = {  // Port-Bezeichnungen
	           // Port5 is the default port (m_iPortNr)
	           // in the constructor of CSerialPort
  "COM1", "COM2", "COM3", "COM4", "UNDEFINED"
};

// ============================================================ Hilfsfunktionen
// ----------------------------------------------------- Fehlermeldung ausgeben
void CSerialOp::DisplayError(int iErr){
	DisplayError(iErr, "");
}

// Start USp    20.10.2002
// allow additional error parameters
// as in printf or str.Format

void CSerialOp::DisplayError(int iErr, LPSTR sz,...)
{
    va_list args;
    va_start(args, sz);
	
	m_bControlOK = FALSE;

	char msgbuf[255];
	vsprintf(msgbuf, m_strMessage[iErr].c_str(), args);
//  AfxMessageBox(m_strMessage[5] + string(m_chPort[m_iPortNr - 1]) +
//    ":\n\n" + m_strMessage[iErr]);
	string serialOpErrorMsg = m_strMessage[5] + string(m_chPort[m_iPortNr - 1]) +
		":\n\n" + msgbuf;
	//if(iErr == 1) // Initialisierungs-Aufforderung, kein Eintrag in ErrorList
		MessageBox(NULL,serialOpErrorMsg.c_str(),"Piezo Error", MB_OK);
	//else
		//((CSMIProjectDoc*)GetSMIDocument())->MsgLog.ShowMsg(m_strDeviceName,serialOpErrorMsg);
}
// End   USp    20.10.2002

// --------------------------------------------------------- Warten iTime in ms
void CSerialOp::Wait(int iTime){
  DWORD dwFirstTime = GetTickCount();
  while(GetTickCount() < (dwFirstTime + (DWORD)iTime));
}

// --------------------------------------------- Zahlenwert aus String auslesen
float CSerialOp::GetValue(string strNumber, int iValPos){
  // Die Funktion erwartet einen String, der durch Leerzeichen getrennte
  // Dezimal-Zahlen enthält
  // iValPos = 0: 1. Zahl; = 1: 2. Zahl usw.
  bool bSign    = TRUE;
  char chNumber;
  float fValue  = 0;
  int iPos      = 0;
  int iPointPos = 0;
  int iLength   = strNumber.length();
  int iFactor   = 1;
  int I;

  iPos = GetStartPos(strNumber, iValPos);
  if(strNumber.substr(iPos, 1) == "-"){  // Vorzeichen feststellen
    bSign = FALSE;
    iPos++;
  }//endif
  if(strNumber.substr(iPos, 1) == "+") iPos++;

  // Position des Komma-Punktes suchen:
  if(iPos < (iLength - 1)){
    iPointPos = iPos + 1;  // Punkt frühestens an 2. Stelle
    while((strNumber.substr(iPointPos, 1) != ".") && (iPointPos < (iLength) - 1)){
      iPointPos++;
    }//wend
  }//endif

  if(iPointPos < (iLength - 1)){
    for(I = iPointPos - 1; I >= iPos; I--){  // Vorkomma-Anteil bestimmen
      chNumber = (char)*(strNumber.c_str() + I) - 48;  // 48 = ASCII("0")
      fValue += (float)(chNumber*(exp(log((float)10)*(iPointPos - I - 1))));
    }//next I
  }//endif
  iPos = iPointPos + 1;  // Ende der Zahl suchen
  if(iPos < (iLength - 1)){
    while((strNumber.substr(iPos, 1) != " ") && (iPos < (iLength - 1))) iPos++;

    for(I = iPointPos + 1; I < iPos; I++){  // Nachkomma-Anteil bestimmen
      chNumber = (char)*(strNumber.c_str() + I) - 48;  // 48 = ASCII("0")
      fValue += (float)(chNumber*(exp(log((float)10)*(iPointPos - I))));
    }//next I
  }//endif
  if(bSign == FALSE) fValue *= -1;

  return fValue;
}

// --------------------------------------------- Zahlstring aus String auslesen
string CSerialOp::GetString(string strNumber, int iValPos){
  // Diese Funktion verhält sich analog zu GetValue, liefert aber den ent-
  // sprechenden String.
  int iStartPos  = 0;
  int iEndPos    = 0;
  int iLength    = strNumber.length();
  int iMaxLength = 8;  // maximale Länge des Rückgabe-Strings
  string strReturn = "";
  
  iStartPos = GetStartPos(strNumber, iValPos);
  iEndPos = iStartPos;  // Ende der Zahl suchen
  while((strNumber.substr(iEndPos, 1) != " ") && (iEndPos < iLength)) iEndPos++;

  iLength = iEndPos - iStartPos;
  if(iLength > iMaxLength) iLength = iMaxLength;
  strReturn = strNumber.substr(iStartPos, iLength);
  return strReturn;
}

// ------------------------- Anfang des Zahlstring an der Stelle iValPos suchen
int CSerialOp::GetStartPos(string strNumber, int iValPos){
  int iPos    = 0;
  int iLength = strNumber.length();

  while(strNumber.substr(iPos, 1) == " "){  // evtl. vorkommende Leer-
    // zeichen überspringen:
    iPos++;
    if(iPos > iLength) return -1;
  }//wend
  if(iValPos > 0){
    for(int I = 0; I < iValPos; I++){
      while(strNumber.substr(iPos, 1) != " "){  // 1. Leerzeichen suchen
        iPos++;
        if(iPos > iLength) return -1;  // Fehler
      }//wend
      while(strNumber.substr(iPos, 1) == " "){  // evtl. vorkommende weitere Leer-
        // zeichen überspringen:
        iPos++;
        if(iPos > iLength) return -1;
      }//wend
    }//next I
  }//endif
  return iPos;
}

// -------------------------------------------- Initialisierungskommandos laden
bool CSerialOp::LoadInitCom(){
  FILE          *cfFile;
  //CFileException cfeException;
  bool           bResult;
  bool           bReturn = FALSE;
  bool           bLineComplete = FALSE;
  bool           bFound = FALSE;
  bool           bQuote = FALSE;
  bool           bStringFound = FALSE;
  char           chBuffer;
  int            iFilePos = 0;
  int            iFileLength = 0;
  int            iLines = 0;
  int            iSemicolonPos;
  int            iEqualPos;
  int            iComIndex = 0;
  int            iLength = 0;

  string        strBuffer = "";
  string        fullFileName = /*(((CSMIProjectDoc*)m_pDoc)->GetSMIAppPath()) + */m_chInitComPath;

  cfFile = fopen(fullFileName.c_str(), "r");
  bResult = (cfFile != NULL);
  if(bResult)// Datei erfolgreich geöffnet
  {  
    //iFileLength = (int)cfFile.GetLength();
    try{
      while(!feof(cfFile))
	  {

        fread((void*)&chBuffer, 1,1,cfFile);  // Zeichen einlesen

        // Prüfen, ob Zeilenende erreicht:
        if((((int)chBuffer == 13) || ((int)chBuffer == 10)) && (bReturn == FALSE))
        {  // Zeilenende erreicht, dabei ist zu beachten: Unter Windows/MS-DOS
          // werden Zeilen mit /013/010 abgeschlossen, unter UNIX nur mit \010.
          bLineComplete = TRUE;
          bReturn       = TRUE;
        }else{
          bReturn = FALSE;
        }//endif
        iFilePos++;

        if(chBuffer == '"'){
          if(bQuote == TRUE) bStringFound = TRUE;  // String gefunden
          bQuote = !bQuote;
        }//endif
        if((bLineComplete == FALSE) && ((int)chBuffer != 10) &&
          ((chBuffer != ' ') || (bQuote == TRUE)) && (chBuffer != '"'))
          strBuffer += chBuffer;  // Zeilenende noch nicht erreicht
          // und Zeichen kein Leerzeichen => eingelesenes Zeichen an Puffer an-
          // hängen

        if((bLineComplete == TRUE) || (iFilePos == iFileLength)){  // ganze
          // Zeile eingelesen oder Dateiende erreicht => Zeile interpretieren:
          iSemicolonPos = strBuffer.find(";", 0);  // Semikolon schließt Befehl
            // ab
          bFound = FALSE;
          if((strBuffer.substr(0,2) != "//") && (strBuffer != "")){  // "//" = Kom-
            // mentar, Leerzeile, 1. Zeichen = Space
            if(iSemicolonPos <= 0) bResult = FALSE;  // Falls kein Semikolon
              // gefunden wurde, Fehlermeldung ausgeben
            if(bStringFound == FALSE){
              if(strBuffer.substr(0,7) == "PortNr="){  // Port-Nummer zuweisen
                chBuffer = *(strBuffer.c_str() + iSemicolonPos - 1);
                m_iPortNr = (int)chBuffer - 48;
                  // Port-Nummer soll nur aus einer Ziffer bestehen
                bFound = TRUE;
              }//endif
              if(strBuffer.substr(0,8) == "Version="){  // Version zuweisen
                iEqualPos = strBuffer.find("=", 0);
                iLength   = strBuffer.length();
                m_strVersion = strBuffer.substr(8, iLength - 9);
                // m_strVersion = strBuffer.Mid(iEqualPos + 2,
                //   iSemicolonPos - iEqualPos - 2);
                bFound = TRUE;
              }//endif

			  // C. Wagner, Anfang
			  if(strBuffer.substr(0,11) == "Controller="){ // Steuerungseinheit identifizieren
			    iLength = strBuffer.length();
			    m_strController = strBuffer.substr(11, iLength - 12);
			    bFound=TRUE;
			  }//endif
				
              if(strBuffer.substr(0,4) == "PZT="){ // Piezo-Verschiebetisch identifizieren
			    iLength = strBuffer.length();
			    m_strPZT = strBuffer.substr(4, iLength - 5);
			    bFound=TRUE;
			  }//endif
			  //C. Wagner, Ende

            }else{
              if(bQuote == FALSE){  // bStringFound == TRUE
                m_strInitCom[iComIndex] = strBuffer.substr(0,iSemicolonPos);
                iComIndex++;
                bFound = TRUE;
              }//endif
            }//endif
            if(bFound == FALSE) bResult = FALSE;
          }//endif
          strBuffer = "";         // Puffer leeren
          bLineComplete = FALSE;  // Zeile abgearbeitet
          bQuote        = FALSE;
          bStringFound  = FALSE;
        }//endif
        m_iComNumber = iComIndex;  // Anzahl Kommandos merken
      }//wend
    }catch(void* pFileError){  // Dateifehler aufgetreten
      //pFileError->Delete();
      bResult = FALSE;
    }//endcatch
    fclose(cfFile);
  }//endif
  return bResult;
}

// ===================================================== Serielle Schnittstelle
// ---------------------------------------------- Serielle Schnittstelle öffnen
bool CSerialOp::OpenConnection(int iCOMNr){
  // iCOMNr = Nr. der Schnittstelle = 1 oder 2 für COM1 oder COM2
  bool         bRes;
  COMMTIMEOUTS CommTimeOuts;
  iCOMNr--;  // Übergeben wird 1 oder 2, die Indizierung beginnt aber bei 0.
  if(m_bConnected == TRUE){  // wenn Verbindung offen, erst schließen
    CloseConnection();
  }//endif
  m_hCOM = CreateFile(m_chPort[iCOMNr], GENERIC_READ | GENERIC_WRITE,
    0, NULL, OPEN_EXISTING, FILE_FLAG_WRITE_THROUGH, NULL);  // COM-Device
    // öffnen
  if((m_hCOM == INVALID_HANDLE_VALUE) || (m_hCOM == NULL)){
    DisplayError(0);
    return FALSE; 
  }//endif

 

  SetCommMask(m_hCOM, EV_ERR | EV_TXEMPTY);  // Spezifikation der möglichen Er-
    // eignisse
  SetupComm(m_hCOM, 4096, 4096);  // Größe von Eingabe- und Ausgabepuffer fest-
    // legen
  PurgeComm(m_hCOM, PURGE_TXCLEAR | PURGE_RXCLEAR);  // Puffer leeren

  // Time-Out-Parameter festlegen:
  CommTimeOuts.ReadIntervalTimeout = 0xFFFFFFFF;
  CommTimeOuts.ReadTotalTimeoutMultiplier = 2;
  CommTimeOuts.ReadTotalTimeoutConstant = 1000;
  CommTimeOuts.WriteTotalTimeoutMultiplier = 2;
  CommTimeOuts.WriteTotalTimeoutConstant = 1000;
  bRes = SetCommTimeouts(m_hCOM, &CommTimeOuts);

  if(bRes == TRUE) m_bConnected = TRUE;  // Flag setzen für offene Verbindung
  bRes = SetupConnection();
  if(bRes == TRUE){  // wenn Verbindung erfolgreich konfiguriert
    m_bControlOK = TRUE;
  }else{
    DisplayError(0);
  }//endif
  return bRes;
}

// --------------------------------------- Serielle Schnittstelle konfigurieren
bool CSerialOp::SetupConnection(){
  int iCOMNr;
  bool bRes;
  DCB  dcbPort;   // Eine DCB-Struktur spezifiziert die Einstellungen für einen
                  // seriellen Anschluß.
  iCOMNr = m_iPortNr - 1;
  dcbPort.DCBlength = sizeof(DCB);
  GetCommState(m_hCOM, &dcbPort);
  dcbPort.BaudRate = m_dwBaudRate;
  dcbPort.ByteSize = m_btByteSize;
  dcbPort.Parity   = m_btParity;
  dcbPort.StopBits = m_btStopBits;
  dcbPort.fOutX    = TRUE;
  dcbPort.fInX     = TRUE;  // FALSE;
  
  bRes = SetCommState(m_hCOM, &dcbPort);  // Einstellungen zuweisen
  if(bRes == FALSE) CloseConnection();
  
  return bRes;
}

// ------------------------------------------- Serielle Schnittstelle schließen
bool CSerialOp::CloseConnection(){
  int iCOMNr = m_iPortNr - 1;
  if(m_bConnected == FALSE) return FALSE;  // Schnittstelle nur dann schließen,
    // wenn sie vorher geöffnet wurde!
  PurgeComm(m_hCOM, PURGE_TXABORT | PURGE_RXABORT | PURGE_TXCLEAR |
    PURGE_RXCLEAR);  // Alle ausstehenden Lese-/Schreib-Operationen löschen/
    // Puffer leeren
  CloseHandle(m_hCOM);   // Verbindung schließen
  m_bConnected = FALSE;  // Flag für offene Verbindung zurücksetzen.
    // Darf nur hier oder bei fehlerhaftem Öffnungsversuch erfolgen!
  m_bControlOK = FALSE;
  return TRUE;
}

// --------------------------------- Daten auf serieller Schnittstelle ausgeben
bool CSerialOp::SendData(const char* chData){
  bool bRes;
  int iCOMNr = m_iPortNr - 1;
  DWORD dwLength, dwBytesWritten;

  if(m_bControlOK == FALSE) return FALSE;  // Falls bereits ein Fehler vor-
    // liegt, nicht mehr versuchen, Daten zu schreiben

  dwLength = (DWORD)GetLength(chData);  // Länge der zu schreibenden Zeichen-
    // kette ermitteln
  bRes = WriteFile(m_hCOM, (LPSTR)chData, dwLength, &dwBytesWritten, NULL);
  if(bRes == FALSE) DisplayError(2);
  return bRes;
}

// ------------------------------- Auf Daten von serieller Schnittstelle warten
int CSerialOp::WaitForData(){
  DWORD iMaxTime = 1000;  // max. 1s Sekunde warten
  DWORD dwBeginTime = GetTickCount();
  int iPos = -1;
  int iInputNumber;
  m_strInputBuffer = "";
  while((GetTickCount() < (dwBeginTime + (DWORD)iMaxTime)) && (iPos < 0)){
    iInputNumber = GetData();
    if(iInputNumber > 0) iPos = m_strInputBuffer.find("\n\0");
  }//endif
  if(iPos >= 0){  // Daten sind angekommen
    m_strInputBuffer = m_strInputBuffer.substr(0,iPos);  // Die letzten
    // zwei Zeichen ("\n\0") abschneiden
    return 0;
  }else{
    return -1;
  }//endif
}

// -------------------------------- Daten von serieller Schnittstelle empfangen
int CSerialOp::GetData(){
  string strInput;
  bool    bResult;
  COMSTAT ComStat;
  DWORD   dwErrorFlags;
  DWORD   dwLength = 0;
  int     I;
  char* pInputBuffer;
  char* pInput;
  ComStat.cbInQue = 0;

  if(m_bControlOK == FALSE) return 0;  // Falls bereits ein Fehler vorliegt,
    // nicht mehr versuchen, Daten zu lesen

  pInputBuffer = m_pInputBuffer;
  pInput       = pInputBuffer;

  ClearCommError(m_hCOM, &dwErrorFlags, &ComStat);  // Anzahl Bytes im Puffer
    // ermitteln
  // dwLength = min((DWORD)nMaxLength, ComStat.cbInQue);
  dwLength = ComStat.cbInQue;  // Anzahl Bytes im Eingangspuffer
  if(dwLength > 0){
    bResult = ReadFile(m_hCOM, pInputBuffer, dwLength, &dwLength, NULL);
      // Daten werden bei pInputBuffer abgelegt
    // Zeichen aus Eingangspuffer in Eingangs-String kopieren bzw. anhängen:
    for(I = 0; I < (int)dwLength; I++){
      m_strInputBuffer += (*pInput); 
      pInput++;
    }//next I
  }//endif
  return (int)dwLength;  // liefert Anzahl der gelesen Bytes
}

// ----------------------------------------- Länge einer Zeichenkette ermitteln
int CSerialOp::GetLength(const char* chString){
  int iLength = 0;
  while(*chString != '\0'){
    iLength++;
    chString++;
  }//wend
  return iLength;
}

// C. Wagner, Anfang
string CSerialOp::GetControllerName()
{
	return m_strController;
}
// C. Wagner, Ende