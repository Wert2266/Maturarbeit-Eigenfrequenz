#define DATA_SIZE 200
short int data[DATA_SIZE];
void setup() {
  Serial.begin(115200);
}

void loop() {
  for(int i = 0;i<DATA_SIZE; i++) {
    int sensorValue = analogRead(A1);
    data[i] = sensorValue;
  }
  for(int i = 0;i<DATA_SIZE; i++) {
    Serial.println(data[i]);
  }
  }



  
