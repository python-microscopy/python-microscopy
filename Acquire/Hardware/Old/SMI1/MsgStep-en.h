// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * MsgStep-en.h: Fehlermeldungen der Klasse CStepOp.                       *
// * (C) 2002/2004 by Udo Spöri, Kirchhoff-Institut für Physik               *
// *     Erstellt am 30. Oktober   2002                                      *
// *  2. Änderung am 30. Oktober   2002                                      *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// die maximale Anzahl an Nachrichten ist definiert 
// in der Klasse SerialOp
// z.Z. m_strMessage[100]

#define _MSG_STEP_KEY "StepOperations"

// Achtung: Die Fehlermeldungen können zusätzliche
//          Strings enthalten, diese sind durch 
//          Platzhalter, z.B. %s gekennzeichnet
#define _MSG_STEP_0 "Error on OPEN INTERFACE."
#define _MSG_STEP_1 "Error. Please initialize."
#define _MSG_STEP_2 "Error on WRITE TO INTERFACE."
#define _MSG_STEP_3 "No response."
#define _MSG_STEP_4 "No response or\nInterpreter-Version mismatch."
#define _MSG_STEP_5 "Serial Port "
  // Spezielle Kommandos für Schrittmotor-Steuerung:
#define _MSG_STEP_10 "Error loading the initializing\ncommands for the stepper motor control.\nFile '%s'"
#define _MSG_STEP_11 "No response or Interpreter-Version mismatch\nof the stepper motor control.\n\nResponse: "
#define _MSG_STEP_12 "WARNING! Remove object mount.\nFor calibration and for determination of the\nlimits of the stepper motor range\nremove the mounting plate!\n\nContinue determination of the maximum traversing range?"
#define _MSG_STEP_13 "Interval could not be set.\nCurrent position is not \nin-between the limits."
#define _MSG_STEP_14 "Parameter for traversing range of the\nstepper motors could not be loaded."
#define _MSG_STEP_15 "Parameter for traversing range of the\nstepper motors could not be stored."

  // Spezielle Kommandos für die Ansicht:
#define _MSG_JOYSTICK_ON "ON"
#define _MSG_JOYSTICK_OFF "OFF"
