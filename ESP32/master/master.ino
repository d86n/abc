#include <esp_now.h>
#include <WiFi.h>

char c;
char str[255];
uint8_t idx = 0;

uint8_t mac1[] = {0x14, 0x33, 0x5C, 0x04, 0x61, 0x18};  // MAC của slave 1
uint8_t mac2[] = {0x1C, 0x69, 0x20, 0xA4, 0xD0, 0x58};  // MAC của slave 2
uint8_t mac3[] = {0x94, 0xB9, 0x7E, 0xFB, 0x23, 0xF0};  // MAC của slave 3

void sendTo(uint8_t *mac, const String &msg) {
  esp_now_send(mac, (const uint8_t *)msg.c_str(), msg.length());
}

void OnDataSent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  Serial.print("\r\nTrạng thái gửi cuối: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Thành công" : "Thất bại");
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init thất bại");
    while (1);
  }

  esp_now_peer_info_t peerInfo = {};
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  memcpy(peerInfo.peer_addr, mac1, 6);
  esp_now_add_peer(&peerInfo);

  memcpy(peerInfo.peer_addr, mac2, 6);
  esp_now_add_peer(&peerInfo);

  memcpy(peerInfo.peer_addr, mac3, 6);
  esp_now_add_peer(&peerInfo);

  esp_now_register_send_cb(OnDataSent);

  Serial.println("ESP32 Master sẵn sàng...");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    // 1. Tìm vị trí của dấu phẩy
    int commaIndex = input.indexOf(',');

    // 2. Kiểm tra định dạng có hợp lệ không
    if (commaIndex <= 0 || commaIndex == input.length() - 1) {
      Serial.println("Định dạng không hợp lệ. Phải là: [index],[message]");
      return;
    }

    // 3. Tách chuỗi thành chỉ số (index) và tin nhắn (message)
    String macIndexStr = input.substring(0, commaIndex);
    String msg = input.substring(commaIndex + 1);

    int macIndex = macIndexStr.toInt();

    // Chuyển đổi tin nhắn sang định dạng để gửi
    const uint8_t *data = (const uint8_t *)msg.c_str();
    int len = msg.length();

    // 4. Gửi tin nhắn đến đúng MAC dựa trên chỉ số
    Serial.print("Đang gửi '" + msg + "' đến MAC index " + macIndexStr + "... ");

    switch (macIndex) {
      case 1:
        esp_now_send(mac1, data, len);
        break;
      case 2:
        esp_now_send(mac2, data, len);
        break;
      case 3:
        esp_now_send(mac3, data, len);
        break;
      case 12:
      case 21:
        esp_now_send(mac1, data, len);
        esp_now_send(mac2, data, len);
        break;
      case 13:
      case 31:
        esp_now_send(mac1, data, len);
        esp_now_send(mac3, data, len);
        break;
      case 23:
      case 32:
        esp_now_send(mac2, data, len);
        esp_now_send(mac3, data, len);
        break;
      case 123:
      case 132:
      case 213:
      case 231:
      case 312:
      case 321:
        esp_now_send(mac1, data, len);
        esp_now_send(mac2, data, len);
        esp_now_send(mac3, data, len);
        break;
      default:
        Serial.println("Chỉ số MAC không hợp lệ. Chỉ chấp nhận 1, 2, hoặc 3.");
        break;
    }
  }
}