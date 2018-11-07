int led = 13;
int strobe = 49; // no dedicated rstart pin for Blackrock
int rstart = 47;
int di0 = 36;
char c;
char d;
int en = 0;
char dio_data[2];
int data_pins[] = {32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};

int rstart_init = 0;

char coda_data[1];
int coda_rec = 30;
int coda_pins[] = {27, 28, 29};

void setup () {
  Serial.begin(115200);

  // initialize the output pins
  for (int k = 0; k < (sizeof(data_pins)/sizeof(int)); k += 1) {
    pinMode(data_pins[k], OUTPUT);
  }
  
  pinMode(led, OUTPUT);
  pinMode(coda_rec, OUTPUT);
  
  for (int k = 0; k < (sizeof(coda_pins)/sizeof(int)); k += 1) {
    pinMode(coda_pins[k], OUTPUT);
  }
  

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
        digitalWrite(rstart, HIGH);
        Serial.println("blackrock recording started"); 
        delay(10);
        digitalWrite(strobe, HIGH);
        delay(200);
        digitalWrite(strobe, LOW);
  
        en = 1;    
        
        // turn on LED (debugging)
        //digitalWrite(led, HIGH);
        //delay(500);
        //digitalWrite(led, LOW);        
    }
    
    // Stop recording
    else if ((c == 'p') && (en == 1)) {
        // positive edge for rstart
        digitalWrite(rstart, LOW);
        Serial.println("blackrock recording stopped"); 
        delay(10);
        digitalWrite(strobe, HIGH);
        delay(200);
        digitalWrite(strobe, LOW);
        
        en = 0;
        c = ' ';
        
        // turn on LED (debugging)
        //digitalWrite(led, HIGH);
        //delay(500);
        //digitalWrite(led, LOW);
    }

    // Set CODA pin to default value (not recording)
    else if ((c == 'g')) {
      digitalWrite(coda_rec, HIGH);
      Serial.println("coda pin set to 1 - default value, not recording"); 
    }
    
    // Start CODA
    else if ((c == 'h')) {
      digitalWrite(coda_rec, LOW);
      digitalWrite(rstart, HIGH);
      Serial.println("coda recording started"); 
    }

    // Digital data
    else if (c == 'd') {
      handle_word();
    }  

    // Coda trial data
    else if (c == 'c') {
      Serial.readBytes(coda_data, 1);
      byte coda_byte_data = coda_data[0];
      handle_coda(coda_byte_data);
      send_coda_to_br(coda_byte_data);
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

  digitalWrite(rstart, HIGH);

  digitalWrite(strobe, HIGH);
  delay(0.5);
  digitalWrite(strobe, LOW);  
}

void handle_coda(byte coda_byte_data) {

  //Set bits: 
  for (int bit_idx = 0; bit_idx < 3; bit_idx +=1) {
    byte mask = 1 << bit_idx;
    if (mask & coda_byte_data) {
      digitalWrite(coda_pins[bit_idx], HIGH);
      Serial.println("HIGH");
      Serial.println(coda_pins[bit_idx]);
    } else {
      digitalWrite(coda_pins[bit_idx], LOW);
      Serial.println("LOW");
      Serial.println(coda_pins[bit_idx]);
    }
  }
}

void send_coda_to_br(byte coda_byte_data) {
  // Set bits:
  for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
    byte mask = 1 << bit_idx;
    if (mask & coda_byte_data) {
      digitalWrite(data_pins[bit_idx], HIGH);
    } else {
      digitalWrite(data_pins[bit_idx], LOW);
    }
  }
 digitalWrite(rstart, HIGH);
 digitalWrite(strobe, HIGH);
 delay(0.5);
 digitalWrite(strobe, LOW);
}
    
    
