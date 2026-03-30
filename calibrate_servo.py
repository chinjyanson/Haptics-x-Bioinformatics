"""
calibrate_servo.py — Set servo to 90° (neutral centre) for physical calibration.

Mount the servo arm while this script holds the shaft at 90°, so the arm
sits centred within the 20°–160° working range.

Usage:
    python calibrate_servo.py
    python calibrate_servo.py --port /dev/tty.usbmodem1401
"""

import argparse
import json
import sys
import time

import serial

DEFAULT_PORT = "/dev/tty.usbmodem1401"
BAUD = 115200


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold servo at 90° for calibration")
    parser.add_argument("--port", default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"[calibrate] Opening {args.port}...")
    try:
        ser = serial.Serial(port=args.port, baudrate=BAUD, timeout=2.0)
    except serial.SerialException as e:
        print(f"[calibrate] Could not open port: {e}")
        sys.exit(1)

    print("[calibrate] Waiting for Arduino boot...")
    time.sleep(2.0)
    ser.reset_input_buffer()

    cmd = json.dumps({"cmd": "set_servo", "angle": 90}) + "\n"
    ser.write(cmd.encode('utf-8'))
    ser.flush()
    print("[calibrate] Servo set to 90°. Mount the arm now.")
    print("[calibrate] Press Ctrl+C when done.")

    try:
        while True:
            # Re-send every second in case the Arduino resets
            ser.write(cmd.encode('utf-8'))
            ser.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\n[calibrate] Done.")


if __name__ == "__main__":
    main()
