// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * MsgPiezo-de.h: Fehlermeldungen der Klasse CPiezoOp.                     *
// * (C) 2002/2004 by Udo Spöri, Kirchhoff-Institut für Physik               *
// *     Erstellt am 30. Oktober   2002                                      *
// *  2. Änderung am 30. Oktober   2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// die maximale Anzahl an Nachrichten ist definiert 
// in der Klasse SerialOp
// z.Z. m_strMessage[100]

#define _MSG_PIEZO_KEY "PiezoOperations"

// Achtung: Die Fehlermeldungen können zusätzliche
//          Strings enthalten, diese sind durch 
//          Platzhalter, z.B. %s gekennzeichnet
#define _MSG_PIEZO_0 "Error on OPEN INTERFACE."
#define _MSG_PIEZO_1 "Error. Please initialize."
#define _MSG_PIEZO_2 "Error on WRITE TO INTERFACE."
#define _MSG_PIEZO_3 "No response."
#define _MSG_PIEZO_4 "No response or\nInterpreter-Version mismatch."
#define _MSG_PIEZO_5 "Serial Port "
  // Spezielle Kommandos für Piezosteuerung:
#define _MSG_PIEZO_10 "Error loading the initializing\ncommands for the piezo control.\nFile '%s'"
#define _MSG_PIEZO_11 "Error! Valid values range from %d µm to %d µm!"

  // anderweitig benutzer Strings
#define _STR_PIEZO_0 "adjusting range[µm]"
