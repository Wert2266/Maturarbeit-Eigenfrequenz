float Ana_out = 0;
void setup() {
  pinMode(A0,INPUT);
  Serial.begin(57600);
}

void loop() {
  Ana_out = analogRead(A0);
  Serial.println(Ana_out);
}
