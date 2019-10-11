int sensorPin = A0;
int sensorValue = 0;
unsigned long start_time = 0;
unsigned long dt = 0;
unsigned long time1 = 0;
unsigned long time2 = 0;

void setup() {
  Serial.begin(115200);
  start_time = micros();
}

bool stream = false;
String rx;
char STREAM_ON = 'a';
char STREAM_OFF = 'b';

void loop() {
  if(Serial.available()) {
    rx = Serial.readString();
    if (rx[0] == STREAM_ON) {
      stream = true;
    } else if (rx[0] == STREAM_OFF) {
      stream = false;
    }
  }
  time2 = micros();
  dt = time2 - time1; // without com, this is about 5 ms
  time1 = time2;

  if (stream) {
    sensorValue = analogRead(sensorPin);
  
    Serial.print(dt);
    Serial.print(", ");
    Serial.println(sensorValue);    
  }
}
