//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
//
// MODULE:  ABFERROR.STR
// PURPOSE: Include file containing the string table for ABF error strings.
//


#include <map>
#include <string>
#include "./../Common/axodefn.h"
#include "abffiles.h"
#include "./../Common/resource.h"

void initErrorMap(std::map<int, std::string>& myMap) {
	#define STR(i, s) myMap[i]=s;
	#define BEGIN
	#define END
	#define STRINGTABLE  
	#define DISCARDABLE 

STRINGTABLE DISCARDABLE 
BEGIN
   STR(ABF_EUNKNOWNFILETYPE,  "File '%s' is of unknown file type.")
   STR(ABF_EBADFILEINDEX,     "INTERNAL ERROR: bad file index.")
   STR(ABF_TOOMANYFILESOPEN,  "INTERNAL ERROR: too many files open.")
   STR(ABF_EOPENFILE,         "Cannot open file '%s'.")
   STR(ABF_EBADPARAMETERS,    "File '%s' has invalid parameters.")
   STR(ABF_EREADDATA,         "File '%s': error reading data.")
   STR(ABF_OUTOFMEMORY,       "Out of memory reading file '%s'.")
   STR(ABF_EREADSYNCH,        "File '%s': error reading synch array.")
   STR(ABF_EBADSYNCH,         "File '%s': has a corrupted synch array.")
   STR(ABF_EEPISODERANGE,     "INTERNAL ERROR: File '%s': episode out of range.")
   STR(ABF_EINVALIDCHANNEL,   "INTERNAL ERROR: File '%s': invalid channel number.")
   STR(ABF_EEPISODESIZE,      "File '%s': has an invalid episode size.")
   STR(ABF_EREADONLYFILE,     "INTERNAL ERROR: file '%s' is read only.")
   STR(ABF_EDISKFULL,         "Insufficient disk space for file '%s'.")
   STR(ABF_ENOTAGS,           "INTERNAL ERROR: File '%s' does not contain any tag information.")
   STR(ABF_EREADTAG,          "Error reading tag from file '%s'.")
   STR(ABF_ENOSYNCHPRESENT,   "INTERNAL ERROR: No synch array is present in file '%s'.")
   STR(ABF_EREADDACEPISODE,   "Error reading DAC episode from file '%s'.")
   STR(ABF_ENOWAVEFORM,       "INTERNAL ERROR: No waveform defined.")
   STR(ABF_EBADWAVEFORM,      "INTERNAL ERROR: Bad waveform definition.")
   STR(ABF_BADMATHCHANNEL,    "INTERNAL ERROR: Bad math channel description.")
   STR(ABF_BADTEMPFILE,       "Could not create a temporary file.")
   STR(ABF_NODOSFILEHANDLES,  "No DOS file handles were available - too many files open.")
   STR(ABF_ENOSCOPESPRESENT,  "INTERNAL ERROR: File '%s' does not contain any scope configuration information.")
   STR(ABF_EREADSCOPECONFIG,  "Error reading scope configuration from file '%s'.")
   STR(ABF_EBADCRC,           "INTERNAL ERROR: Bad CRC on reading ABF file -- file may be corrupted.")
   STR(ABF_ENOCOMPRESSION,    "Data compression is not supported on this platform.")
   STR(ABF_EREADDELTA,        "Error reading delta from file '%s'.")
   STR(ABF_ENODELTAS,         "INTERNAL ERROR: File '%s' does not contain any delta information.")
   STR(ABF_EBADDELTAID,       "INTERNAL ERROR: ABFDelta has an unknown parameter ID.")
   STR(ABF_EWRITEONLYFILE,    "INTERNAL ERROR: file '%s' is write only.")
   STR(ABF_ENOSTATISTICSCONFIG,  "INTERNAL ERROR: File '%s' does not contain a statistics window configuration structure.")
   STR(ABF_EREADSTATISTICSCONFIG, "INTERNAL ERROR: Error reading statistics window configuration from file '%s'.")
   STR(ABF_EWRITERAWDATAFILE, "INTERNAL ERROR: Cannot modify raw data files.")
   STR(ABF_EWRITEMATHCHANNEL, "INTERNAL ERROR: Cannot modify math channels.")

   STR(ABFH_EHEADERREAD,      "Error reading file header.")
   STR(ABFH_EHEADERWRITE,     "Error writing file header.")
   STR(ABFH_EINVALIDFILE,     "Not a valid ABF file.")
   STR(ABFH_EUNKNOWNFILETYPE, "Not a valid ABF file.")
   STR(ABFH_CHANNELNOTSAMPLED,"The requested Analog IN channel was not sampled.")
   STR(ABFH_EPOCHNOTPRESENT,  "The requested epoch is not present.")
   STR(ABFH_ENOWAVEFORM,      "No waveform was defined.")
   STR(ABFH_EDACFILEWAVEFORM, "Waveform was defined by a DAC file.")
   STR(ABFH_ENOMEMORY,        "Out of memory!")

   STR(ABFH_BADSAMPLEINTERVAL,      "Invalid sample interval.")
   STR(ABFH_BADSECONDSAMPLEINTERVAL,"Invalid second sample interval.")
   STR(ABFH_BADSAMPLEINTERVALS,     "The first and second sampling intervals must be integer multiples of each other.")
   STR(ABFH_ENOCONDITTRAINS,        "There is an error in the User List.\n\nConditioning trains must be enabled in order to vary this parameter.")
   STR(ABFH_EMETADURATION,          "There is an error in the User List.\n\nThe conditioning train duration is too long.  Either reduce the duration or increase the sweep start to start time.")
   STR(ABFH_ECONDITNUMPULSES,       "There is an error in the User List.\n\nThe number of pulses in the conditioning train must be between 1 and 10000.")
   STR(ABFH_ECONDITBASEDUR,         "There is an error in the User List.\n\nThe conditioning train baseline duration must be between 0.01 and 10000 ms.")
   STR(ABFH_ECONDITBASELEVEL,       "There is an error in the User List.\n\nThe conditioning train baseline level is out of range.")
   STR(ABFH_ECONDITPOSTTRAINDUR,    "There is an error in the User List.\n\nThe conditioning train post-train duration must be between 0 and 10000 ms.")
   STR(ABFH_ECONDITPOSTTRAINLEVEL,  "There is an error in the User List.\n\nThe conditioning train post-train level is out of range.")
   STR(ABFH_ESTART2START,           "There is an error in the User List.\n\nThe time between sweep starts is too short.")
   STR(ABFH_EINACTIVEHOLDING,       "There is an error in the User List.\n\nThe inactive analog OUT holding level is out of range.")
   STR(ABFH_EINVALIDCHARS,          "The user list contains invalid characters or is empty.\n\nValid characters are: 0..9 E e + - .\n\nValid characters for binary values: 0 1 *.")
   STR(ABFH_EINVALIDBINARYCHARS,    "The user list contains invalid characters.\n\nValid characters for binary values are: 0 1 *,\n\nA * is valid for defining a digital train.")
   STR(ABFH_ENODIG,                 "There is an error in the User List.\n\nThe waveform digital outputs must be enabled in order to vary this parameter.")
   STR(ABFH_EDIGHOLDLEVEL,          "There is an error in the User List.\n\nThe waveform digital inter-sweep holding level is out of range.")
   STR(ABFH_ENOPNPULSES,            "There is an error in the User List.\n\nP/N leak subtraction must be enabled in order to vary this parameter.")
   STR(ABFH_EPNNUMPULSES,           "There is an error in the User List.\n\nThe number of P/N leak subtraction sub-sweeps must be between 1 and 8.")
   STR(ABFH_ENOEPOCH,               "There is an error in the User List.\n\nThe waveform epoch requested in the user list must be enabled in order to vary this parameter.")
   STR(ABFH_EEPOCHLEN,              "There is an error in the User List.\n\nThe waveform duration is too long.  Either reduce the waveform duration or increase the sweep duration.")
   STR(ABFH_EEPOCHINITLEVEL,        "There is an error in the User List.\n\nThe waveform initial level is out of range.")
   STR(ABFH_EDIGLEVEL,              "There is an error in the User List.\n\nThe waveform digital pattern is out of range.")
   STR(ABFH_ECONDITSTEPLEVEL,       "There is an error in the User List.\n\nThe conditioning train step level is out of range.")
   STR(ABFH_ECONDITSTEPDUR,         "There is an error in the User List.\n\nThe conditioning train step duration must be between 0.01 and 10000 ms.")
   STR(ABF_ECRCVALIDATIONFAILED,    "The Cyclic Redundancy Code (CRC) validation failed while opening the file.")

   STR(IDS_ENOMESSAGESTR,     "INTERNAL ERROR: No message string assigned to error %d.")
   STR(IDS_EPITAGHEADINGS,    "Tag #    Time (s)  Episode  Comment")
   STR(IDS_CONTTAGHEADINGS,   "Tag #    Time (s)   Comment")
   STR(IDS_NONE, "<none>")
   STR(IDS_CANNOTLOAD,		  "This version of ABFInfo is out of date.")

   STR(IDS_MATHCHANNEL, "Math")
END
}

extern "C"	INT   WINAPI c_LoadString( HINSTANCE instance, UINT resource_id,
                               char* buffer, INT buflen )
	{
		std::map<int,std::string> errorMap;
		initErrorMap(errorMap);
		strcpy(buffer,errorMap[resource_id].c_str());
		return (int)errorMap[resource_id].length();
	}
