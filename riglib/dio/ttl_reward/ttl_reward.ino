// Code to send data to reward system
// the pin that will attach to the BNC --> reward

int outputPin = 8;       
char c;

void setup() {
  Serial.begin(115200);
  // initialize the output:
  pinMode(outputPin, OUTPUT);
}


void loop() {
  if (Serial.available() >=1) {
    c = Serial.read();
    
    // Start reward:
    if (c == 'a') {
      digitalWrite(outputPin, HIGH);
    }
    
    else if (c == 'b') {
      digitalWrite(outputPin, LOW);
      
      
    }
  }
}



