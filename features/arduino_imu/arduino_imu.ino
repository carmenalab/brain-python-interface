/* MPU9250 Basic Example Code
 by: Kris Winer
 date: April 1, 2014
 license: Beerware - Use this code however you'd like. If you
 find it useful you can buy me a beer some time.
 Modified by Brent Wilkins July 19, 2016

 Demonstrate basic MPU-9250 functionality including parameterizing the register
 addresses, initializing the sensor, getting properly scaled accelerometer,
 gyroscope, and magnetometer data out. Added display functions to allow display
 to on breadboard monitor. Addition of 9 DoF sensor fusion using open source
 Madgwick and Mahony filter algorithms. Sketch runs on the 3.3 V 8 MHz Pro Mini
 and the Teensy 3.1.

 SDA and SCL should have external pull-up resistors (to 3.3V).
 10k resistors are on the EMSENSR-9250 breakout board.

 Hardware setup:
 MPU9250 Breakout --------- Arduino
 VDD ---------------------- 3.3V
 VDDI --------------------- 3.3V
 SDA ----------------------- A4
 SCL ----------------------- A5
 GND ---------------------- GND
 */
#include <Wire.h>
#include <SPI.h>
#include "quaternionFilters.h"
#include "MPU9250.h"

#define AHRS false         // Set to false for basic data read
#define SerialDebug false  // Set to true to get Serial output for debugging

// Pin definitions
int intPin = 12;  // These can be changed, 2 and 3 are the Arduinos ext int pins
int myLed  = 13;  // Set up pin 13 led for toggling
char cc;
MPU9250 myIMU = MPU9250(-1);

void setup()
{
  Serial.begin(115200);
  //myIMU.begin();
  //myIMU.ak8963WhoAmI_SPI();
  // Serial.println("Testing MPU9250::readBytes...");
  byte fc = 0;
  // Serial.println( myIMU.readBytes(MPU9250_ADDRESS, WHO_AM_I_MPU9250, 1, &fc) );
  // Serial.println(fc, HEX);
  Serial.flush();
  byte fd = 0;
  myIMU.readBytes(AK8963_ADDRESS, WHO_AM_I_AK8963, 1, &fd);
  // Serial.println(fd, HEX);

  // Read the WHO_AM_I register, this is a good test of communication
  byte c = myIMU.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
  // Serial.print(F("MPU9250 I AM 0x"));
  // Serial.print(c, HEX);
  // Serial.print(F(" I should be 0x"));
  // Serial.println(0x71, HEX);
  Serial.flush();

  if (c == 0x71) // WHO_AM_I should always be 0x68
  {
    // Serial.println("MPU9250 is online...");

    // Start by performing self test and reporting values
//      myIMU.MPU9250SelfTest(myIMU.selfTest);
//    Serial.print(F("x-axis self test: acceleration trim within : "));
//    Serial.print(myIMU.selfTest[0],1); Serial.println("% of factory value");
//    Serial.print(F("y-axis self test: acceleration trim within : "));
//    Serial.print(myIMU.selfTest[1],1); Serial.println("% of factory value");
//    Serial.print(F("z-axis self test: acceleration trim within : "));
//    Serial.print(myIMU.selfTest[2],1); Serial.println("% of factory value");
//    Serial.print(F("x-axis self test: gyration trim within : "));
//    Serial.print(myIMU.selfTest[3],1); Serial.println("% of factory value");
//    Serial.print(F("y-axis self test: gyration trim within : "));
//    Serial.print(myIMU.selfTest[4],1); Serial.println("% of factory value");
//    Serial.print(F("z-axis self test: gyration trim within : "));
//    Serial.print(myIMU.selfTest[5],1); Serial.println("% of factory value");

    // Calibrate gyro and accelerometers, load biases in bias registers
    myIMU.calibrateMPU9250(myIMU.gyroBias, myIMU.accelBias);
    myIMU.initMPU9250();
    // Initialize device for active mode read of acclerometer, gyroscope, and
    // temperature
    //Serial.println("MPU9250 initialized for active data mode....");

    // Read the WHO_AM_I register of the magnetometer, this is a good test of
    // communication
    byte d = myIMU.readByte(AK8963_ADDRESS, WHO_AM_I_AK8963);
    //Serial.print("AK8963 "); Serial.print("I AM "); Serial.print(d, HEX);
    //Serial.print(" I should be "); Serial.println(0x48, HEX);

    // Get magnetometer calibration from AK8963 ROM
    myIMU.initAK8963(myIMU.factoryMagCalibration);
    // Initialize device for active mode read of magnetometer
    //Serial.println("AK8963 initialized for active data mode....");
    if (SerialDebug)
    {
      //  Serial.println("Calibration values: ");
      Serial.print("X-Axis factory sensitivity adjustment value ");
      Serial.println(myIMU.factoryMagCalibration[0], 2);
      Serial.print("Y-Axis factory sensitivity adjustment value ");
      Serial.println(myIMU.factoryMagCalibration[1], 2);
      Serial.print("Z-Axis factory sensitivity adjustment value ");
      Serial.println(myIMU.factoryMagCalibration[2], 2);
    }

  } // if (c == 0x71)
  else
  {
    Serial.print("Could not connect to MPU9250: 0x");
    Serial.println(c, HEX);
    while(1) ; // Loop forever if communication doesn't happen
  }
}

void loop()
{
  // If intPin goes high, all data registers have new data
  // On interrupt, check if data ready interrupt
  // if (myIMU.readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)
  if (Serial.available() >=1)
  {  
    cc = Serial.read();
    if (cc=='d') {
    
    myIMU.readAccelData(myIMU.accelCount);  // Read the x/y/z adc values
    myIMU.getAres();

    // Now we'll calculate the accleration value into actual g's
    // This depends on scale being set
    myIMU.ax = (float)myIMU.accelCount[0]*myIMU.aRes; // - accelBias[0];
    myIMU.ay = (float)myIMU.accelCount[1]*myIMU.aRes; // - accelBias[1];
    myIMU.az = (float)myIMU.accelCount[2]*myIMU.aRes; // - accelBias[2];

    myIMU.readGyroData(myIMU.gyroCount);  // Read the x/y/z adc values
    myIMU.getGres();

    // Calculate the gyro value into actual degrees per second
    // This depends on scale being set
    myIMU.gx = (float)myIMU.gyroCount[0]*myIMU.gRes;
    myIMU.gy = (float)myIMU.gyroCount[1]*myIMU.gRes;
    myIMU.gz = (float)myIMU.gyroCount[2]*myIMU.gRes;

    myIMU.readMagData(myIMU.magCount);  // Read the x/y/z adc values
    myIMU.getMres();
    // User environmental x-axis correction in milliGauss, should be
    // automatically calculated
    myIMU.magBias[0] = +470.;
    // User environmental x-axis correction in milliGauss TODO axis??
    myIMU.magBias[1] = +120.;
    // User environmental x-axis correction in milliGauss
    myIMU.magBias[2] = +125.;

    // Calculate the magnetometer values in milliGauss
    // Include factory calibration per data sheet and user environmental
    // corrections
    // Get actual magnetometer value, this depends on scale being set
    myIMU.mx = (float)myIMU.magCount[0] * myIMU.mRes
               * myIMU.factoryMagCalibration[0] - myIMU.magBias[0];
    myIMU.my = (float)myIMU.magCount[1] * myIMU.mRes
               * myIMU.factoryMagCalibration[1] - myIMU.magBias[1];
    myIMU.mz = (float)myIMU.magCount[2] * myIMU.mRes
               * myIMU.factoryMagCalibration[2] - myIMU.magBias[2];    

    // Must be called before updating quaternions!
    myIMU.updateTime();

  // Sensors x (y)-axis of the accelerometer is aligned with the y (x)-axis of
  // the magnetometer; the magnetometer z-axis (+ down) is opposite to z-axis
  // (+ up) of accelerometer and gyro! We have to make some allowance for this
  // orientationmismatch in feeding the output to the quaternion filter. For the
  // MPU-9250, we have chosen a magnetic rotation that keeps the sensor forward
  // along the x-axis just like in the LSM9DS0 sensor. This rotation can be
  // modified to allow any convenient orientation convention. This is ok by
  // aircraft orientation standards! Pass gyro rate as rad/s
//  MadgwickQuaternionUpdate(ax, ay, az, gx*PI/180.0f, gy*PI/180.0f, gz*PI/180.0f,  my,  mx, mz);
  MahonyQuaternionUpdate(myIMU.ax, myIMU.ay, myIMU.az, myIMU.gx*DEG_TO_RAD,
                         myIMU.gy*DEG_TO_RAD, myIMU.gz*DEG_TO_RAD, myIMU.my,
                         myIMU.mx, myIMU.mz, myIMU.deltat);

    myIMU.delt_t = millis() - myIMU.count;
    // if (myIMU.delt_t > 500)
  
      
      //if(SerialDebug)
      //{
        // Print acceleration values in milligs!
        //Serial.print("X-acceleration: "); 
        Serial.println(1000*myIMU.ax);
        //Serial.print(" mg ");
        //Serial.print("Y-acceleration: "); 
        Serial.println(1000*myIMU.ay);
        //Serial.print(" mg ");
        //Serial.print("Z-acceleration: "); 
        Serial.println(1000*myIMU.az);
        //Serial.println(" mg ");

        // Print gyro values in degree/sec
        //Serial.print("X-gyro rate: "); 
        Serial.println(myIMU.gx, 3);
        //Serial.print(" degrees/sec ");
        //Serial.print("Y-gyro rate: "); 
        Serial.println(myIMU.gy, 3);
        //Serial.print(" degrees/sec ");
        //Serial.print("Z-gyro rate: "); 
        Serial.println(myIMU.gz, 3);
        //Serial.println(" degrees/sec");

        // Print mag values in degree/sec
        //Serial.print("X-mag field: "); 
        //Serial.println(myIMU.mx);
        //Serial.print(" mG ");
        //Serial.print("Y-mag field: "); 
        //Serial.println(myIMU.my);
        //Serial.print(" mG ");
        //Serial.print("Z-mag field: "); 
        //Serial.println(myIMU.mz);
        //Serial.println(" mG");

        myIMU.tempCount = myIMU.readTempData();  // Read the adc values
        // Temperature in degrees Centigrade
        myIMU.temperature = ((float) myIMU.tempCount) / 333.87 + 21.0;
        // Print temperature in degrees Centigrade
        //Serial.print("Temperature is ");  
        //Serial.println(myIMU.temperature, 1);
        //Serial.println(" degrees C");

        myIMU.count = millis();
        digitalWrite(myLed, !digitalRead(myLed));  // toggle led
    } 
  }
} 

