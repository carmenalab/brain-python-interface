int nsleep = 2;
int step_x = 3;
int dir_x = 4;
int step_y = 5;
int dir_y = 6;
int step_z = 7;
int dir_z = 8;

int min_limit_x = 9;
int max_limit_x = 10;
int min_limit_y = 11;
int max_limit_y = 12;
int min_limit_z = 13;
int max_limit_z = 13;

void setup() {
  Serial.begin(115200);
  
  pinMode(nsleep, OUTPUT);
  pinMode(step_x, OUTPUT);
  pinMode(dir_x, OUTPUT);
  pinMode(min_limit_x, INPUT);
  pinMode(max_limit_x, INPUT);
  
  pinMode(step_y, OUTPUT);
  pinMode(dir_y, OUTPUT);
  pinMode(min_limit_y, INPUT);
  pinMode(max_limit_y, INPUT);
  
  pinMode(step_z, OUTPUT);
  pinMode(dir_z, OUTPUT);
  pinMode(min_limit_z, INPUT);
  pinMode(max_limit_z, INPUT);
}

int mask_step_x = 0x01; // 0b00000001
int mask_step_y = 0x02; // 0b00000010
int mask_step_z = 0x04; // 0b00000100
int mask_dir_x = 0x10; // 0b00010000
int mask_dir_y = 0x20; // 0b00100000
int mask_dir_z = 0x40; // 0b01000000

String inData;


void loop() {
  while (Serial.available() > 0)  {
    char recieved = Serial.read();
    inData += recieved; 
      // Process message when new line character is recieved
      if (recieved == '\n') {
        // Read the limit switches
        int room_to_decrease_x = digitalRead(min_limit_x);
        int room_to_increase_x = digitalRead(max_limit_x);
        int room_to_decrease_y = digitalRead(min_limit_y);
        int room_to_increase_y = digitalRead(max_limit_y);
        int room_to_decrease_z = 1; //digitalRead(min_limit_z);
        int room_to_increase_z = 1; //digitalRead(max_limit_z);      
              
        char cmd = inData[0];
        if (cmd == 'w') { // wake up motor
          digitalWrite(nsleep, HIGH);        
        }
        else if (cmd == 's') { // put motors to sleep
          digitalWrite(nsleep, LOW);        
        }
        else if (cmd == 'm') { // move
          char step_dir_data = inData[1];
          //char step_dir_data = Serial.read();
          
          // set motor directions
          digitalWrite(dir_x, step_dir_data & mask_dir_x);
          digitalWrite(dir_y, step_dir_data & mask_dir_y);
          digitalWrite(dir_z, step_dir_data & mask_dir_z);
          
          // step the motors, if the limit switches are high (room to move)
    
          // TODO make use of the limit switches!
          // TODO need to map out which direction corresponds to which switch!
          
          // Step the motors
          digitalWrite(step_x, step_dir_data & mask_step_x);
          digitalWrite(step_y, step_dir_data & mask_step_y);
          digitalWrite(step_z, step_dir_data & mask_step_z);      
          
          delay(10);
          
          digitalWrite(step_x, LOW);      
          digitalWrite(step_y, LOW);            
          digitalWrite(step_z, LOW);
          
        }
        
          // Respond with the status of the limit switches 
          Serial.print("Limit switches read: ");
          Serial.print(room_to_decrease_x, DEC);
          Serial.print(room_to_increase_x, DEC);      
          Serial.print(room_to_decrease_y, DEC);
          Serial.print(room_to_increase_y, DEC);
          Serial.println();    
    
          inData = "";
      
     }
  }
}  

