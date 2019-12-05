/*
 IO slave:

use the arduino as a serial io slave. Command format is:

S[modifier] [pin #] [value]\n  -- set a value
Q[modifier] [pin #]\n          -- query a value

Commands currently supported are:

SD  -- Set a digital io pin
SA  -- Set an analog (PWM) output value 
SF  -- Set a pin to flash - in this case value is half period in us
QA  -- Read an analog value (from ADC)
SS  -- Set servo value

Set commands typically echo something back, which will either need to be read and discarded or flushed. 
 */
#include <string.h>
#include <stdio.h>
#include <Servo.h>

Servo servos[13];


#define INPUT_BUFFER_SIZE 200

char inputBuffer[INPUT_BUFFER_SIZE];         // a string to hold incoming data
char inputString[INPUT_BUFFER_SIZE];
int inputLength = 0;
boolean stringComplete = false;  // whether the string is complete

char buffer[200]; //a buffer for various string operations
char out[16];



void setup() {
  // initialize serial:
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }
  
  // Attach and null servos
  servos[9].attach(9);
  servos[10].attach(10);
  servos[11].attach(11);
  servos[0].write(0);
  servos[10].write(0);
  servos[11].write(0);
}



void parseCommandArgs(char* command, int *chan, float *val)
{
  char *p = command + 2; 
  char *str;
      
  str = strtok_r(p, " ", &p);
  *chan = atoi(str);
      
  str = strtok_r(p, " ", &p);
  *val = atof(str);
}

boolean flashChan[14];
boolean flashStates[14];
unsigned long flashPeriod[14];
unsigned long lastMicr[14];

void flash()
{
  unsigned long micr;
  micr = micros();
  for (int i = 0; i < 14; i++)
  {
    if (flashChan[i] && ((micr - lastMicr[i]) > flashPeriod[i]))
    {
      flashStates[i] = !flashStates[i];
      digitalWrite(i, flashStates[i]);
      lastMicr[i] = micr;
    }
  }
}  

void loop() {
  flash();
  if (stringComplete) {
    stringComplete = false;
    
    //querying an analog value
    if (strncmp(inputString, "QA", 2) == 0)
    {
      int chan = int(inputString[2]) - 48;
      int val = analogRead(chan);

      float volts = val*(5.0/1023.0);
      sprintf(out, "%d.%04d", (int)floor(volts), (int)trunc((volts - floor(volts))*10000));
      Serial.println(out);
    }
    
    
    //Setting a digital value
    if (strncmp(inputString, "SD", 2) == 0)
    {
      int chan;
      float val;
      
      parseCommandArgs(inputString, &chan, &val);
      
      pinMode(chan, OUTPUT);
      digitalWrite(chan, int(val));
      Serial.println(val);
    }
    
    //Setting an analog / PWM value
    if (strncmp(inputString, "SA", 2) == 0)
    {
      int chan;
      float val;
      
      parseCommandArgs(inputString, &chan, &val);
      
      pinMode(chan, OUTPUT);
      
      analogWrite(chan, int(255.0*val/5.0));

      Serial.println(val);
    }
    
    //Setting a (the) servo
    if (strncmp(inputString, "SS", 2) == 0)
    {
      int chan;
      float val;
      
      parseCommandArgs(inputString, &chan, &val);

      Serial.println(val);
      servos[chan].write(val);
    }
    
    //Setting a pin to flash
    if (strncmp(inputString, "SF", 2) == 0)
    {
      int chan;
      float val;
      unsigned long period;
      
      parseCommandArgs(inputString, &chan, &val);
      
      period = long(val);
      
      if (period != 0)
      {
        pinMode(chan, OUTPUT);
     
        flashChan[chan] = 1;
        flashPeriod[chan] = period;
      } else flashChan[chan] = 0;
    }
    
    /*
    //querying the temperature
    if (strncmp(inputString, "QT", 2))
    {
      int chan = int(inputString[2]) - 48;

      float val = sensors.getTempC(thermAddresses[chan]);

      Serial.println(val);
    }
    
    //querying the temperature
    if (strncmp(inputString, "S?", 2))
    {
      int nTemp = sensors.getDeviceCount();
      
      sprintf(out, "Num Sensors: %d", nTemp);
      Serial.println(out);
    }*/
    
    // clear the string:
    for (int i = 0; i < INPUT_BUFFER_SIZE; i++) inputString[i] = 0;
  }
  
  serialPoll();
}

/*
  SerialEvent occurs whenever a new data comes in the
 hardware serial RX.  This routine is run between each
 time loop() runs, so using delay inside loop can delay
 response.  Multiple bytes of data may be available.
 */
void serialPoll() {
  while (Serial.available() && (inputLength < INPUT_BUFFER_SIZE)) {
    // get the new byte:
    char inChar = (char)Serial.read(); 
    // add it to the inputBuffer:
    inputBuffer[inputLength] = inChar;
    inputLength++;
    // if the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') {
      //copy from our buffer into the input string
      strcpy(inputString, inputBuffer);
      stringComplete = true;
      
      //Zero our buffer
      for (int i = 0; i < INPUT_BUFFER_SIZE;i++) inputBuffer[i] = 0;
      //reset the counter
      inputLength = 0;
    } 
  }
}


