// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * PiezoOp.h: Schnittstelle für die Klasse CPiezoOp.                       *
// * (C) 1999/2000 by Benno Albrecht, Kirchhoff-Institut für Physik          *
// *     Erstellt am 30. März      2000                                      *
// *  9. Änderung am 26. September 2001                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// Klasse zur Ansteuerung der Hardware zur Piezosteuerung (Mitglied der
// Dokumentklasse)

#if !defined(AFX_PIEZOOP_H__C8EA0594_0622_11D4_99AE_0000C0E169AB__INCLUDED_)
#define AFX_PIEZOOP_H__C8EA0594_0622_11D4_99AE_0000C0E169AB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "SerialOp.h"

class CPiezoOp: public CSerialOp{
public:
	string GetRangeError();
	string GetHardRange();
	string GetFirmware();
	float VoltToMikro(int iChannel, int iVolt);
	int MikroToVolt (int iChannel, float iMikro);
	int Calibrate();
  CPiezoOp();
  virtual ~CPiezoOp();

// Elementzugriff:
  bool GetCtrlStatus()      {return m_bExternControl;}
  float GetPos(int iChannel){return m_fPos[iChannel - 1];}
  void SetMin(int iChannel, float fMin){m_fMin[iChannel - 1] = fMin;}
  float GetMin(int iChannel){return m_fMin[iChannel - 1];}
  void SetMax(int iChannel, float fMax){m_fMax[iChannel - 1] = fMax;}
  float GetMax(int iChannel){return m_fMax[iChannel - 1];}

  void SetChannelObject(int iChannel){m_iChannelObject = iChannel;}
  int GetChannelObject()   {return m_iChannelObject;}
  int GetChannelPhase()    {return m_iChannelPhase;}

  float GetHardMin(int iChannel){return m_fHardMin[iChannel - 1];}
  float GetHardMax(int iChannel){return m_fHardMax[iChannel - 1];}


// Kommandos:
  int Init(int iMode);                 // Intialisierung der Piezosteuerung
  void ContIO();                       // Setzt Verfahr-Flag zurück
  void MoveTo(int iChannel, float fPos, bool bTimeOut);  // Positionierbefehl
                                       // für Piezotisch ausführen
  float GetOnePosition(int iChannel);  // liefert Position
  void SetExtCtrlOnOff();              // Externe Steuerung ein-/ausschalten
                                       // (Piezosteuerung)
  void SetAllToNull();                 // Alle Piezo-Kanäle auf 0 setzen

private:
  // Variablen/Attribute:
  string m_strInitCom[32];            // enthalten Initialisierungskommandos
  int m_iComNumber;                    // Anzahl der Initialisierungskommandos
  bool m_bExternControl;               // = TRUE: Piezos werden extern ge-
                                       //         steuert
  bool m_bTimeOut;                     // solange TRUE, keine Kommandos an
                                       // Piezosteuerung senden
  int m_iTimeCounter;
  float m_fPos[3];                     // aktuelle Position Piezo an Kanal 1-3
  float m_fMin[3];                     // minimale Position Piezo-Kanal 1-3
  float m_fMax[3];                     // maximale Position Piezo-Kanal 1-3

  int m_iChannelObject;                // Kanal für Objektpositionierung (nur 1
                                       // oder 2 möglich)
  int m_iChannelPhase;                 // Kanal für Phase (nur 3)

  float m_fHardMin[3];                 // minimale Position der Hardware
  float m_fHardMax[3];                 // maximale Position der Hardware
  int m_iDummy;                        // Platzhalter für Serialisierung
  
  // Interne Funktionen:
  void InitUserValues();               // Vom Benutzer einstellbare Parameter
                                       // initialisieren
  void InitChannels();                 // Kanäle der Piezosteuerung auf Closed-
                                       // Loop-Betrieb einstellen
};

#endif // !defined(AFX_PIEZOOP_H__C8EA0594_0622_11D4_99AE_0000C0E169AB__INCLUDED_)
