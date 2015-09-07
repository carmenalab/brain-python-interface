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
int max_limit_z = A0; // using analog pin because UNO doesn't haven enough functioning digital pins!

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
//  pinMode(max_limit_z, INPUT);


  // Sleep motors on startup
  digitalWrite(nsleep, LOW);        
}

int mask_step_x = 0x01; // 0b00000001
int mask_step_y = 0x02; // 0b00000010
int mask_step_z = 0x04; // 0b00000100
int mask_dir_x = 0x10; // 0b00010000
int mask_dir_y = 0x20; // 0b00100000
int mask_dir_z = 0x40; // 0b01000000

//String inData;


int inData[14];
int indata_idx = 0;

void loop() {
  while (Serial.available() > 0)  {
    char recieved = Serial.read();
    //inData += recieved;

    inData[indata_idx % 14] = (int) recieved;
    indata_idx += 1;
    //if (recieved == '\n') {
    //  Serial.println("newline");
    //} else {
    //  Serial.println(recieved);
    //}
    
      // Process message when new line character is recieved
      if (recieved == '\n') {
        //Serial.println("starting things");
        // Read the limit switches
        int room_to_decrease_x = digitalRead(min_limit_x);
        int room_to_increase_x = digitalRead(max_limit_x);
        int room_to_decrease_y = digitalRead(min_limit_y);
        int room_to_increase_y = digitalRead(max_limit_y);
        int room_to_decrease_z = digitalRead(min_limit_z);
        
        int room_to_increase_z_sensor_val = analogRead(max_limit_z);
        int room_to_increase_z = (int) room_to_increase_z_sensor_val > 512;
              
        char cmd = inData[0];
        //Serial.print("cmd = ");
        //Serial.print(cmd);
        //Serial.println();
        if (cmd == 'w') { // wake up motor
          digitalWrite(nsleep, HIGH);        
        }
        else if (cmd == 's') { // put motors to sleep
          digitalWrite(nsleep, LOW);        
        }
        else if (cmd == 'm') { // move max 1 step in each dimension
          //Serial.println("sending step command");
          char step_dir_data = inData[1];
          //Serial.println(step_dir_data);
          
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
        
        else if (cmd == 'c') { // continuous motion for a number of steps in each dimension
          // Read number of steps needed
          int n_steps_x = (inData[1] << 8) | inData[2] & 0xff;
          int n_steps_y = (inData[3] << 8) | inData[4] & 0xff;
          int n_steps_z = (inData[5] << 8) | inData[6] & 0xff;
          
          // set motor directions
          digitalWrite(dir_x, n_steps_x >= 0);
          digitalWrite(dir_y, n_steps_y >= 0);
          digitalWrite(dir_z, n_steps_z < 0);          
          
          int n_steps_to_travel_x = abs(n_steps_x);
          int n_steps_to_travel_y = abs(n_steps_y);          
          int n_steps_to_travel_z = abs(n_steps_z);          
          
          int n_steps_commanded_x = 0;
          int n_steps_commanded_y = 0;
          int n_steps_commanded_z = 0;
          
          int x_going = (n_steps_commanded_x  < n_steps_to_travel_x);
          int y_going = (n_steps_commanded_y < n_steps_to_travel_y);
          int z_going = (n_steps_commanded_z < n_steps_to_travel_z);
          
          while (x_going || y_going || z_going) {
            
            // Read the limit switches
            int room_to_decrease_x = digitalRead(min_limit_x);
            int room_to_increase_x = digitalRead(max_limit_x);
            int room_to_decrease_y = digitalRead(min_limit_y);
            int room_to_increase_y = digitalRead(max_limit_y);
            int room_to_decrease_z = digitalRead(min_limit_z);
             
            int room_to_increase_z_sensor_val = analogRead(max_limit_z);
            int room_to_increase_z = (int) room_to_increase_z_sensor_val > 512;            
            
            // Move the x-stage
            int room_to_move_x = ((n_steps_x >= 0) && room_to_increase_x) || ((n_steps_x < 0 ) && room_to_decrease_x);
            if (x_going && (n_steps_commanded_x  < n_steps_to_travel_x) && room_to_move_x ) {
              digitalWrite(step_x, HIGH);
              //delay(10);
              //digitalWrite(step_x, LOW); 
              n_steps_commanded_x += 1;              
            } else {
              x_going = 0;
            }

            // Move the y-stage
            int room_to_move_y = ((n_steps_y >= 0) && room_to_increase_y) || ((n_steps_y < 0 ) && room_to_decrease_y);
            if (y_going && (n_steps_commanded_y < n_steps_to_travel_y) && room_to_move_y ) {
              digitalWrite(step_y, HIGH);
              //delay(10);
              //digitalWrite(step_y, LOW);
              n_steps_commanded_y += 1;              
            } else {
              y_going = 0;
            }

            // Move the z-stage
            int room_to_move_z = ((n_steps_z >= 0) && room_to_increase_z) || ((n_steps_z < 0 ) && room_to_decrease_z);
            if (z_going && (n_steps_commanded_z < n_steps_to_travel_z) && room_to_move_z ) {
              digitalWrite(step_z, HIGH);
              //delay(10);
              //digitalWrite(step_z, LOW); 
              n_steps_commanded_z += 1;
            } else {
              z_going = 0;
            }

            // Turn off step
            delay(10);
            digitalWrite(step_x, LOW); 
            digitalWrite(step_y, LOW);     
            digitalWrite(step_z, LOW);  
          }
          
          Serial.print("c move: ");
          Serial.print(n_steps_commanded_x, DEC);
          Serial.print(", ");
          Serial.print(n_steps_commanded_y, DEC);      
          Serial.print(", ");        
          Serial.print(n_steps_commanded_z, DEC);          
          Serial.println();
                  
        }
        else if (cmd == '\n') {        
          // Respond with the status of the limit switches 
          Serial.print("Limit switches read: ");
          Serial.print(room_to_decrease_x, DEC);
          Serial.print(room_to_increase_x, DEC);      
          Serial.print(room_to_decrease_y, DEC);
          Serial.print(room_to_increase_y, DEC);
          Serial.print(room_to_decrease_z, DEC);
          Serial.print(room_to_increase_z, DEC);          
          Serial.println();    
        }
        //inData = "";
        indata_idx = 0;
      
     }
  }
}  

