// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * MsgStep-de.h: Fehlermeldungen der Klasse CStepOp.                       *
// * (C) 2002/2004 by Udo Spöri, Kirchhoff-Institut für Physik               *
// *     Erstellt am 30. Oktober   2002                                      *
// *  2. Änderung am 30. Oktober   2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// die maximale Anzahl an Nachrichten ist definiert 
// in der Klasse SerialOp
// z.Z. m_strMessage[100]



// um die deutschen Fehlermeldungen einzubinden, ist es notwending
// den zusätzlichen compile-Parmeter _DEUTSCH einzufügen
// bei   -> Einstellungen
//         -> C++
//           -> Präprozessor definitionen
//              ..., _DEUTSCH

// Achtung: Die Fehlermeldungen können zusätzliche
//          Strings enthalten, diese sind durch 
//          Platzhalter, z.B. %s gekennzeichnet
#define _MSG_STEP_0 "Fehler beim Öffnen."
#define _MSG_STEP_1 "Fehler. Bitte zuerst initialisieren."
#define _MSG_STEP_2 "Fehler beim Schreiben."
#define _MSG_STEP_3 "Keine Antwort."
#define _MSG_STEP_4 "Keine Antwort oder\nfalsche Interpreter-Version."
#define _MSG_STEP_5 "Serielle Schnittstelle "
  // Spezielle Kommandos für Schrittmotor-Steuerung:
#define _MSG_STEP_10 "Fehler beim Laden der Initialisierungs-\nkommandos für die Schrittmotor-Steuerung.\nFile '%s'"
#define _MSG_STEP_11 "Keine Antwort oder falsche Interpreter-\nVersion der Schrittmotor-Steuerung.\n\nRückmeldung: "
#define _MSG_STEP_12 "ACHTUNG! Zur Kalibrierung und zum Ermitteln des\nmaximalen Verfahrbereichs der Schrittmotoren\nmuß die Objektträger-Halterung entfernt werden!\n\nSoll der Verfahrbereich jetzt ermittelt werden?"
#define _MSG_STEP_13 "Kann Intervall nicht setzen, da\nmomentane Position nicht\nzwischen den Grenzen."
#define _MSG_STEP_14 "Parameter für Verfahrbereich der\nSchrittmotoren konnten nicht\ngeladen werden."
#define _MSG_STEP_15 "Parameter für Verfahrbereich der\nSchrittmotoren konnten nicht\ngespeichert werden."

  // Spezielle Kommandos für die Ansicht:
#define _MSG_JOYSTICK_ON "EIN"
#define _MSG_JOYSTICK_OFF "AUS"
