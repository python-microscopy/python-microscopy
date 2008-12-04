//***********************************************************************************************
//
//    Copyright (c) 1993-1997 Axon Instruments.
//    All rights reserved.
//    Permission is granted to freely to use, modify and copy the code in this file.
//
//***********************************************************************************************
// This is ABFOLDNX.H; a header file containing definition of parameter
// numbers for old pCLAMP binary data files
//
// NOTE: arrays are 0 relative in C, but 1 relative in BASIC

#ifndef __ABFOLDNX_H__
#define __ABFOLDNX_H__ 1

#define ABF_OLDUNITLEN        16      // length of old ADC/DAC units strings
#define ABF_OLDCOMMENTLEN     77      // length of old descriptor comment
#define ABF_OLDCONDITLEN      64      // length of old conditioning string

#define F53_EXPERIMENTTYPE          0
#define F53_ADCNUMCHANNELS          1
#define F53_SAMPLESPEREPISODE       2
#define F53_ACTUALEPISODESPERFILE   3
#define F53_ADCSAMPLEINTERVAL       4
//      F53_UNUSED6                 5
#define F53_FILESTARTTIME           6
#define F53_FILEELAPSEDTIME         7
#define F53_FILEVERSIONNUMBER       8
#define F53_FILESTARTDATE           9
                                    
//      F53_UNUSED11                10
#define F53_SEGMENTSPEREPISODE      11
#define F53_OLDSAMPLESPEREPISODE    11
#define F53_EPISODESPERFILE         12
#define F53_REQUESTEDSAMPLEINTERVAL 13
#define F53_OPERATIONMODE           14
//      F53_UNUSED16                15
//      F53_UNUSED17                16
#define F53_POSTTRIGGERPORTION      17
//      F53_UNUSED19                18
//      F53_UNUSED20                19
#define F53_NUMPOINTSIGNORED        20
//      F53_UNUSED22                21  // LongDescriptionPtr ___ never used
#define F53_TAGSECTIONPTR           22
#define F53_TIMEOUTWAIT             23
#define F53_DAC0HOLDINGLEVEL        24
#define F53_NUMTAGENTRIES           25   // Was unused, used in translation
//      F53_UNUSED27                26
//      F53_UNUSED28                27
//      F53_UNUSED29                28
//      F53_UNUSED30                29
#define F53_ADCNUMBERINGSTRATEGY    30
#define F53_ADCFIRSTLOGICALCHANNEL  31
#define F53_GAINDACTOCELL           32
#define F53_TAPEWINDUPTIME          33
#define F53_ENVIRONMENTALINFO       34
#define F53_CELLID1                 35
#define F53_CELLID2                 36
//      F53_UNUSED38                37
#define F53_THRESHOLDCURRENT        38
#define F53_ADDITINSTGAIN           39
#define F53_INSTRUMENTFILTER        40
#define F53_AUTOSAMPLEINSTRUMENT    41
//      F53_UNUSED43                42
#define F53_UNUSED44                43
#define F53_UNUSED45                44
#define F53_UNUSED46                45
#define F53_UNUSED47                46
#define F53_UNUSED48                47  // was copy of ADCNUMCHANNELS
#define F53_CHANNEL0GAIN            48
#define F53_INVALIDLASTDATA         49
#define F53_UNUSED51                50
//      F53_UNUSED52                51
#define F53_ADCRANGE                52
#define F53_DACRANGE                53
#define F53_ADCRESOLUTION           54
#define F53_DACRESOLUTION           55
//      F53_UNUSED57                56
//      F53_UNUSED58                57
//      F53_UNUSED59                58
//      F53_UNUSED60                59
#define F53_AMPLIFICATIONFACTOR     60
#define F53_VERTICALOFFSET          61
//      F53_UNUSED63                62
//      F53_UNUSED64                63
//      F53_UNUSED65                64
//      F53_UNUSED66                65
//      F53_UNUSED67                66
//      F53_UNUSED68                67
//      F53_UNUSED69                68
#define F53_DATADISPLAYMODE         69
//      F53_UNUSED71                70
//      F53_UNUSED72                71
//      F53_UNUSED73                72
//      F53_UNUSED74                73
//      F53_UNUSED75                74
//      F53_UNUSED76                75
//      F53_UNUSED77                76
//      F53_UNUSED78                77
//      F53_UNUSED79                78
//      F53_UNUSED80                79
                                    
#define F53_INSTOFFSET              96
#define F53_INSTSCALEFACTOR         112
#define F53_ADCDISPLAYGAIN          128
#define F53_ADCDISPLAYOFFSET        144


// =============================================================================
// CLAMPEX Data file version 5.2 Param() index #defines

#define C52_EXPERIMENTTYPE          0
#define C52_ADCNUMCHANNELS          1
#define C52_SAMPLESPEREPISODE       2
#define C52_EPISODESPERFILE         3
#define C52_FIRSTCLOCKPERIOD        4  
#define C52_SECONDCLOCKPERIOD       5   
#define C52_FILESTARTTIME           6 
#define C52_FILEELAPSEDTIME         7
#define C52_FILEVERSIONNUMBER       8
#define C52_FILESTARTDATE           9
#define C52_INTEREPISODETIME        10  
#define C52_RUNSPERFILE             11  
#define C52_EPISODESPERRUN          12  
#define C52_FIRSTCLOCKINTERVAL      13  
#define C52_STARTDELAY              14  
#define C52_NUMTRIALS               15  
#define C52_STARTEPISODENUM         16  
#define C52_GAINMULTIPLIER          17  
#define C52_OLDPROTOCOLTYPE         17
#define C52_TRIGGERMODE             18  
#define C52_PULSESAMPLECH1          19  
#define C52_FIRSTTRIGGEREPISODE     20  
#define C52_LASTTRIGGEREPISODE      21  
#define C52_PULSESAMPLECH2          22  
#define C52_SEGMENTSPEREPISODE      23  
#define C52_DAC0HOLDINGLEVEL        24
#define C52_EPOCHALEVELINIT         25  
#define C52_EPOCHAINCREMENT         26  
#define C52_EPOCHBLEVELINIT         27  
#define C52_EPOCHBINCREMENT         28  
#define C52_EPOCHAINITDURATION      29  
#define C52_UNUSED31                30  
#define C52_ADCFIRSTLOGICALCHANNEL  31
#define C52_OLDEPOCHALEVEL          31
#define C52_GAINDACTOCELL           32
#define C52_PULSESINTRAIN           33  
#define C52_PRECONDURATION          34  
#define C52_PRECONLEVEL             35  
#define C52_CONDURATION             36  
#define C52_CONLEVEL                37  
#define C52_POSTCONDURATION         38  
#define C52_POSTCONLEVEL            39  
#define C52_FILTERCUTOFF            40  
#define C52_CH1PULSE                41  
#define C52_CH2PULSE                42  
#define C52_EPOCHCLEVELINIT         43  
#define C52_EPOCHCINCREMENT         44  
#define C52_EPOCHCINITDURATION      45  
#define C52_SECONDCLOCKRATE         46
#define C52_OLDMULTIPLEXCODE        47  
#define C52_AUTOSAMPLEINSTRUMENT    48
#define C52_INTEREPISODEAMP         49
#define C52_OLDCHANNEL0GAIN         48
#define C52_OLDCHANNEL1GAIN         49
//      C52_UNUSED51                50
#define C52_INTEREPISODEWRITE       51
#define C52_ADCRANGE                52
#define C52_DACRANGE                53
#define C52_ADCRESOLUTION           54
#define C52_DACRESOLUTION           55
#define C52_EPOCHBINITDURATION      56
#define C52_EPOCHBDURATIONINC       57
#define C52_CONDITVARIABLE          58
#define C52_EPOCHADURATIONINC       59
#define C52_CH0DISPLAYAMPLIFICATION 60
#define C52_CH0DISPLAYOFFSET        61
#define C52_CH1DISPLAYAMPLIFICATION 62
#define C52_CH1DISPLAYOFFSET        63
#define C52_AUTOPEAKCHANNEL         63
#define C52_AVERAGEDDATADISPLAY     64
#define C52_AUTOPEAKSEARCHMODE      65
#define C52_AUTOPEAKCENTER          66
#define C52_AUTOPEAKAVPOINTS        67
#define C52_AUTOERASE               68
#define C52_DATADISPLAYMODE         69
#define C52_BASELINECALCULATION     70
#define C52_AUTOPEAKDESTINATION     71
#define C52_DISPLAYSEGMENTNUM       72
#define C52_PLOTDENSITY             73
//      C52_UNUSED75                74
//      C52_UNUSED76                75
//      C52_UNUSED77                76
//      C52_UNUSED78                77
//      C52_UNUSED79                78
//      C52_UNUSED80                79
#define C52_EPOCHCDURATIONINC       80
#define C52_EPOCHDLEVELINIT         81
#define C52_EPOCHDINCREMENT         82
#define C52_EPOCHDINITDURATION      83
#define C52_EPOCHDDURATIONINC       84
//      C52_UNUSED86                85
#define C52_EPOCHATYPE              86
#define C52_EPOCHBTYPE              87
#define C52_EPOCHCTYPE              88
#define C52_EPOCHDTYPE              89
#define C52_PNNUMPULSES             90
#define C52_PNADCNUM                91
#define C52_PNHOLDINGLEVEL          92
#define C52_PNSETTLINGTIME          93
#define C52_PNINTERPULSE            94
#define C52_AUTOSAMPLEADCNUM        95
                                    
#define C52_INSTOFFSET              96
#define C52_INSTSCALEFACTOR         112
#define C52_ADCDISPLAYGAIN          128
#define C52_ADCDISPLAYOFFSET        144

#endif   /* __ABFOLDNX_H__ */
