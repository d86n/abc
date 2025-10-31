#include <esp_now.h>
#include <WiFi.h>

#define TXD2 17
#define RXD2 16

void OnDataRecv(const esp_now_recv_info_t *, const uint8_t *data, int len) {
  String msg = "";
  for (int i = 0; i < len; i++) msg += (char)data[i];

  Serial.print("ESP32 nhận từ ESP-NOW: ");
  Serial.println(msg);

  Serial2.println(msg);  // Gửi sang Mega
  Serial2.println("Đã chuyển sang Mega qua Serial2");
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init thất bại");
    while (1);
  }

  esp_now_register_recv_cb(OnDataRecv);
  Serial.println("ESP32 Slave sẵn sàng...");
}

void loop() {}
