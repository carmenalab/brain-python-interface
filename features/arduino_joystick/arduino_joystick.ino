/*
  AnalogReadSerial
 Reads an analog input on pin 0, prints the result to the serial monitor 
 
 This example code is in the public domain.
 */

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(A0);
  int sensorValue2 = analogRead(A1);
  
  Serial.println(sensorValue);
  Serial.println(sensorValue2);
  delay(1);
  
}
