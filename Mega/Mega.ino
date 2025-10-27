const size_t output[4][4] = {{24, 25, 26, 27},
                             {28, 29, 30, 31},
                             {32, 33, 34, 35},
                             {36, 37, 38, 39}};

void setup() {
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      pinMode(output[i][j], OUTPUT);
    }
  }
} 

// MAIN FUNCTION
void loop() {
  goStraight(output[4][4]);
}

void clockwise(const size_t output_i[4]) {
  for (size_t i = 0; i < 4; i++) {
    size_t halfStep = 0;
    for (size_t j = 0; j < 4; j++) {
      if (halfStep) {
        digitalWrite(output_i[(j + 1) % 4], (i == j) ? HIGH : LOW);
      }
      digitalWrite(output_i[j], (i == j) ? HIGH : LOW);

      halfStep = 1;
    }
    delay(2);
  }
}

void anticlockwise(const size_t output_i[4]) {
  for (size_t i = 4; i-- > 0;) {
    size_t halfStep = 0;
    for (size_t j = 0; j < 4; j++) {
      if (halfStep) {
        digitalWrite(output_i[(j + 1) % 4], (i == j) ? HIGH : LOW);
      }
      digitalWrite(output_i[j], (i == j) ? HIGH : LOW);

      halfStep = 1;
    }
    delay(2);
  }
}

void goStraight(const size_t output_i[4][4]) {
  anticlockwise(output[0]);
  anticlockwise(output[1]);
  clockwise(output[2]);
  clockwise(output[3]);
}

void rotate(const size_t output[4][4]) {
  for (size_t i = 0; i < 4; i++) {
    clockwise(output[i]);
  }
}