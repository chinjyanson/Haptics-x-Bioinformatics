/*
 * BCI FYP — Arduino Uno R3 peripheral sketch
 *
 * Inputs:
 *   Rotary encoder phase A → pin 2 (INT0), phase B → pin 3 (INT1)
 * Outputs:
 *   Vibration motor 1 (via transistor / motor driver) on PWM pin 9
 *   Vibration motor 2 (via transistor / motor driver) on PWM pin 10
 *
 * Serial protocol (115200 baud, JSON lines):
 *
 *   Python → Arduino:
 *     {"cmd":"start","task":1}                        — begin task N; ack sent back
 *     {"cmd":"stop"}                                  — stop, zero both motors; ack sent back
 *     {"cmd":"set_vibration","motor":1,"intensity":N} — set motor 1 PWM 0-255; no ack
 *     {"cmd":"set_vibration","motor":2,"intensity":N} — set motor 2 PWM 0-255; no ack
 *     {"cmd":"set_report_interval","ms":N}            — set encoder report interval (10-1000 ms)
 *
 *   Arduino → Python:
 *     {"type":"ready"}                    — sent once on boot
 *     {"type":"encoder","delta":N,"position":P} — delta ticks + absolute position (every 50 ms if non-zero)
 *     {"type":"ack","cmd":"start","task":N} — start command acknowledgement (also resets position to 0)
 *     {"type":"ack","cmd":"stop"}           — stop command acknowledgement
 *
 *   Pin assignments: ENC_A=2(INT0)  ENC_B=3(INT1)  VIB1=9(PWM)  VIB2=10(PWM)
 */

#include <Arduino.h>
#include <ArduinoJson.h>

// ── Pin assignments ───────────────────────────────────────────────────────────
static const uint8_t ENC_A   = 2;   // INT0 — encoder phase A
static const uint8_t ENC_B   = 3;   // INT1 — encoder phase B
static const uint8_t VIB1    = 9;   // PWM pin for vibration motor 1
static const uint8_t VIB2    = 10;  // PWM pin for vibration motor 2

// ── Encoder state (volatile — shared with ISR) ────────────────────────────────
volatile int32_t encoderDelta = 0;
volatile uint8_t lastEncoded  = 0;

// ── Absolute encoder position (accumulated, can be reset via command) ─────────
int32_t encoderPosition = 0;

// ── Encoder reporting interval (adjustable via set_report_interval command) ───
uint32_t encoderReportMs = 50;
uint32_t lastEncoderReport = 0;

// ── Encoder ISR — called on any edge of phase A or B ─────────────────────────
void encoderISR() {
    uint8_t a       = digitalRead(ENC_A);
    uint8_t b       = digitalRead(ENC_B);
    uint8_t encoded = (a << 1) | b;
    uint8_t sum     = (lastEncoded << 2) | encoded;

    // Gray-code transition table
    if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) {
        encoderDelta++;
    } else if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) {
        encoderDelta--;
    }
    lastEncoded = encoded;
}

// ── Command handler ───────────────────────────────────────────────────────────
void handleCommand(const char* jsonStr) {
    StaticJsonDocument<128> doc;
    if (deserializeJson(doc, jsonStr) != DeserializationError::Ok) return;

    const char* cmd = doc["cmd"];
    if (!cmd) return;

    if (strcmp(cmd, "start") == 0) {
        int task = doc["task"] | 0;
        // Reset encoder position at task start
        encoderPosition = 0;
        StaticJsonDocument<64> ack;
        ack["type"] = "ack";
        ack["cmd"]  = "start";
        ack["task"] = task;
        serializeJson(ack, Serial);
        Serial.println();

    } else if (strcmp(cmd, "stop") == 0) {
        analogWrite(VIB1, 0);
        analogWrite(VIB2, 0);
        StaticJsonDocument<48> ack;
        ack["type"] = "ack";
        ack["cmd"]  = "stop";
        serializeJson(ack, Serial);
        Serial.println();

    } else if (strcmp(cmd, "set_report_interval") == 0) {
        uint32_t ms = doc["ms"] | 50;
        encoderReportMs = constrain(ms, 10, 1000);

    } else if (strcmp(cmd, "set_vibration") == 0) {
        int motor     = doc["motor"] | 1;  // default motor 1 if omitted
        int intensity = constrain(doc["intensity"] | 0, 0, 255);
        if (motor == 2) {
            analogWrite(VIB2, (uint8_t)intensity);
        } else {
            analogWrite(VIB1, (uint8_t)intensity);
        }
        // No ack — high-frequency motor commands must not flood the serial link
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    pinMode(ENC_A, INPUT_PULLUP);
    pinMode(ENC_B, INPUT_PULLUP);
    pinMode(VIB1,  OUTPUT);
    pinMode(VIB2,  OUTPUT);
    analogWrite(VIB1, 0);
    analogWrite(VIB2, 0);

    // Reduce stream timeout so readBytesUntil() doesn't block the loop for 1 s
    Serial.setTimeout(50);

    // Attach interrupts on both encoder pins (CHANGE catches all edges)
    attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_B), encoderISR, CHANGE);

    lastEncoded = (digitalRead(ENC_A) << 1) | digitalRead(ENC_B);

    // Announce readiness so Python can confirm the connection is live
    StaticJsonDocument<48> ready;
    ready["type"] = "ready";
    serializeJson(ready, Serial);
    Serial.println();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
    uint32_t now = millis();

    // 1. Read incoming JSON command (non-blocking — readBytesUntil returns 0 if nothing)
    if (Serial.available()) {
        char buf[128];
        uint8_t len = Serial.readBytesUntil('\n', buf, sizeof(buf) - 1);
        buf[len] = '\0';
        if (len > 0) {
            handleCommand(buf);
        }
    }

    // 2. Report accumulated encoder delta every encoderReportMs
    if (now - lastEncoderReport >= encoderReportMs) {
        lastEncoderReport = now;

        noInterrupts();
        int32_t delta = encoderDelta;
        encoderDelta  = 0;
        interrupts();

        if (delta != 0) {
            // Update absolute position
            encoderPosition += delta;
            
            StaticJsonDocument<64> doc;
            doc["type"]     = "encoder";
            doc["delta"]    = delta;
            doc["position"] = encoderPosition;
            serializeJson(doc, Serial);
            Serial.println();
        }
    }
}
