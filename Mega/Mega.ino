void setup() {
  Serial.begin(115200);
  Serial.println("Arduino Mega bat dau gui du lieu qua Serial1...");

  // TX1 (chân 18), RX1 (chân 19)
  Serial1.begin(115200); 
}

void loop() {
  // Serial1.print("Hello ESP32 from Mega!");
  // Serial.println("Da gui: Hello ESP32 from Mega!");
  
  // Kiểm tra nếu có dữ liệu nhận được từ Serial1 (RX1)
  if (Serial1.available()) {
    String receivedData = Serial1.readStringUntil('\n');
    Serial.print("Da nhan tu ESP32: ");
    Serial.println(receivedData);
  }
  
  delay(2000); // Đợi 2 giây
}