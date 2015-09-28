int led = 13;
int strobe = 49; // no dedicated rstart pin for Blackrock
int rstart = 49;
int di0 = 36;
char c;
char d;
int en = 0;
char dio_data[2];
int data_pins[] = {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48};

int rstart_init = 0;

void setup () {
  Serial.begin(9600);

  // initialize the output pins
  for (int k = 0; k < (sizeof(data_pins)/sizeof(int)); k += 1) {
    pinMode(data_pins[k], OUTPUT);
  }
  
  pinMode(led, OUTPUT);

}

int rstart_count = 0;
int half_word = 0;

void loop() {
  if (Serial.available() >= 1) {
    c = Serial.read();
  
    // Start recording
    if ((c == 'r') && (en == 0)) {
        if (rstart_init == 0) {
          pinMode(strobe, OUTPUT);        
          rstart_init = 1;
        }
        
        // positive edge for rstart
        digitalWrite(data_pins[15], HIGH);
        delay(10);
        digitalWrite(rstart, HIGH);
        delay(200);
        digitalWrite(rstart, LOW);
  
        en = 1;    
        
        // turn on LED (debugging)
        digitalWrite(led, HIGH);
        delay(500);
        digitalWrite(led, LOW);        
    }
    
    // Stop recording
    else if ((c == 'p') && (en == 1)) {
        // positive edge for rstart
        digitalWrite(data_pins[15], LOW);
        delay(10);
        digitalWrite(rstart, HIGH);
        delay(200);
        digitalWrite(rstart, LOW);
        
        en = 0;
        c = ' ';
        
        // turn on LED (debugging)
        digitalWrite(led, HIGH);
        delay(500);
        digitalWrite(led, LOW);
    }
    
    // Digital data
    else if (c == 'd') {
      handle_word();
    }  
  }
}

void handle_word() { 
  Serial.readBytes(dio_data, 2);
  char d1 = dio_data[0];
  char d0 = dio_data[1];
  
  
    // set all the data bits
    for (int byte_idx = 0; byte_idx < 2; byte_idx += 1) {
      byte data_byte = dio_data[byte_idx];
      for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
        int pin_idx = 8*byte_idx + bit_idx;
        byte mask = 1 << bit_idx;
        if (mask & data_byte) {
          digitalWrite(data_pins[pin_idx], HIGH);
        } else {
          digitalWrite(data_pins[pin_idx], LOW);         
        }
      }
    }  

  digitalWrite(data_pins[15], HIGH);

  digitalWrite(strobe, HIGH);
  delay(0.5);
  digitalWrite(strobe, LOW);  
}
