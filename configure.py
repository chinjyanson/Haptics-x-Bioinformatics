"""
configure.py — Arduino hardware test / configuration utility

Opens a direct serial connection to the Arduino Uno R3 and lets you
interactively test both vibration motors and observe encoder events.
Run this before the main experiment to verify wiring is correct.

Usage:
    python configure.py --port /dev/ttyACM0
    python configure.py --port COM3 --baud 115200
"""

import argparse
import csv
import json
import os
import sys
import threading
import time

import serial

from arduino.bridge import ARDUINO_DEFAULT_PORT, ARDUINO_DEFAULT_BAUD

MENU = """
Commands:
  v1 <0-255>         Set motor 1 intensity       (e.g. v1 128)
  v1 on              Motor 1 full vibration      (intensity 255)
  v1 off             Motor 1 off                 (intensity 0)
  v2 <0-255>         Set motor 2 intensity       (e.g. v2 200)
  v2 on              Motor 2 full vibration      (intensity 255)
  v2 off             Motor 2 off                 (intensity 0)
  enc verify         Stream encoder deltas for 5 s (rotate to test)
  enc record <s>     Record encoder for N seconds, display + save CSV
  enc record <s> -p  Record and print only (no CSV saved)
  enc calibrate      Count ticks for one full revolution
  enc interval N     Set report interval to N ms (default 50, range 10-1000)
  start              Send start command          (task 1)
  stop               Send stop command           (zeros both motors)
  q / quit           Exit
"""


# ── Background reader ─────────────────────────────────────────────────────────

def _reader(ser: serial.Serial, stop: threading.Event) -> None:
    """Print incoming JSON lines from the Arduino."""
    while not stop.is_set():
        try:
            raw = ser.readline()
            if raw:
                line = raw.decode('utf-8', errors='replace').strip()
                if line:
                    print(f"\n[Arduino] {line}\n> ", end='', flush=True)
        except Exception:
            break


# ── Command sender ────────────────────────────────────────────────────────────

def _send(ser: serial.Serial, cmd: dict) -> None:
    payload = json.dumps(cmd) + "\n"
    ser.write(payload.encode('utf-8'))
    ser.flush()


# ── Encoder utilities ─────────────────────────────────────────────────────────

def _enc_verify(ser: serial.Serial, duration: float = 5.0) -> None:
    """Stream encoder deltas for `duration` seconds and print a summary."""
    print(f"Encoder verify — rotate the encoder for {duration:.0f} s  (Ctrl-C to stop early)")
    print(f"{'Time':>6}  {'Delta':>6}  {'Cumulative':>10}")
    print("-" * 28)

    ser.reset_input_buffer()
    cumulative = 0
    deadline = time.time() + duration
    try:
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "encoder":
                continue
            delta = msg.get("delta", 0)
            cumulative += delta
            elapsed = duration - (deadline - time.time())
            print(f"{elapsed:6.2f}s  {delta:+6}  {cumulative:+10}")
    except KeyboardInterrupt:
        print()

    direction = "CW (+)" if cumulative > 0 else "CCW (-)" if cumulative < 0 else "no movement"
    print(f"\nSummary: {abs(cumulative)} total ticks, net direction {direction}")


def _enc_record(ser: serial.Serial, duration: float, save_csv: bool = True) -> None:
    """
    Record encoder deltas for `duration` seconds, print a live table,
    and optionally save to encoder_record_<timestamp>.csv.
    Columns: timestamp, delta, cumulative
    """
    print(f"Encoder record — {duration:.0f} s  (Ctrl-C to stop early)")
    print(f"{'Time':>8}  {'Delta':>6}  {'Cumulative':>10}")
    print("-" * 30)

    ser.reset_input_buffer()
    rows: list[tuple[float, int, int]] = []
    cumulative = 0
    start = time.time()
    deadline = start + duration

    try:
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "encoder":
                continue
            ts = time.time() - start
            delta = msg.get("delta", 0)
            cumulative += delta
            rows.append((round(ts, 4), delta, cumulative))
            print(f"{ts:8.3f}s  {delta:+6}  {cumulative:+10}")
    except KeyboardInterrupt:
        print()

    total_time = time.time() - start
    print(f"\n{len(rows)} events in {total_time:.2f} s  |  net cumulative: {cumulative:+}")

    if not rows:
        return

    if save_csv:
        fname = f"encoder_record_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "delta", "cumulative"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {os.path.abspath(fname)}")


def _enc_calibrate(ser: serial.Serial) -> None:
    """
    Guided one-revolution calibration.
    Tells the user to rotate exactly one full turn, counts ticks, and saves
    ticks_per_rev to gsr_calibration.json (appended alongside GSR fields).
    """
    import os

    print("Encoder calibrate — turn the encoder exactly ONE full revolution,")
    print("then press Enter to stop counting.\n")
    input("Press Enter when ready to start counting...")
    ser.reset_input_buffer()
    print("Counting ticks — rotate one full revolution now...")

    cumulative = 0
    stop_flag = threading.Event()

    def _count():
        nonlocal cumulative
        while not stop_flag.is_set():
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "encoder":
                cumulative += abs(msg.get("delta", 0))

    counter_thread = threading.Thread(target=_count, daemon=True)
    counter_thread.start()

    try:
        input("Press Enter after completing one full revolution...")
    except KeyboardInterrupt:
        print()
    finally:
        stop_flag.set()

    ticks = cumulative
    print(f"\nTicks counted: {ticks}")

    if ticks == 0:
        print("No ticks detected — check wiring and try again.")
        return

    cal_path = "gsr_calibration.json"
    cal = {}
    if os.path.exists(cal_path):
        try:
            with open(cal_path) as f:
                cal = json.load(f)
        except Exception:
            pass

    cal["encoder_ticks_per_rev"] = ticks
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved encoder_ticks_per_rev={ticks} to {cal_path}")


# ── Input parser ──────────────────────────────────────────────────────────────

def _parse(text: str) -> dict | None:
    """
    Parse user input and return the JSON command dict, or None if unrecognised.
    """
    parts = text.strip().lower().split()
    if not parts:
        return None

    if parts[0] in ('v1', 'v2'):
        motor = 1 if parts[0] == 'v1' else 2
        if len(parts) == 1:
            print(f"Usage: {parts[0]} <0-255> | {parts[0]} on | {parts[0]} off")
            return None
        arg = parts[1]
        if arg == 'on':
            return {"cmd": "set_vibration", "motor": motor, "intensity": 255}
        if arg == 'off' or arg == '0':
            return {"cmd": "set_vibration", "motor": motor, "intensity": 0}
        try:
            intensity = max(0, min(255, int(arg)))
            return {"cmd": "set_vibration", "motor": motor, "intensity": intensity}
        except ValueError:
            print(f"Invalid intensity '{arg}' — use a number 0-255, 'on', or 'off'")
            return None

    if parts[0] == 'enc':
        if len(parts) < 2:
            print("Usage: enc verify | enc record [s] [-p] | enc calibrate | enc interval <ms>")
            return None
        sub = parts[1]
        if sub == 'verify':
            return {"cmd": "_enc_verify"}
        if sub == 'record':
            duration = 10.0
            save = True
            if len(parts) >= 3:
                try:
                    duration = float(parts[2])
                except ValueError:
                    print(f"Invalid duration '{parts[2]}' — use a number of seconds")
                    return None
            if '-p' in parts:
                save = False
            return {"cmd": "_enc_record", "duration": duration, "save": save}
        if sub == 'calibrate':
            return {"cmd": "_enc_calibrate"}
        if sub == 'interval':
            if len(parts) < 3:
                print("Usage: enc interval <ms>  (range 10-1000)")
                return None
            try:
                ms = max(10, min(1000, int(parts[2])))
                return {"cmd": "set_report_interval", "ms": ms}
            except ValueError:
                print(f"Invalid interval '{parts[2]}' — use a number 10-1000")
                return None
        print(f"Unknown enc sub-command '{sub}'.")
        return None

    if parts[0] == 'start':
        return {"cmd": "start", "task": 1}

    if parts[0] == 'stop':
        return {"cmd": "stop"}

    if parts[0] in ('q', 'quit'):
        return {"cmd": "_quit"}

    print(f"Unknown command '{text.strip()}'. Type 'q' to quit.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Interactive Arduino hardware test for BCI FYP.'
    )
    parser.add_argument(
        '--port', '-p',
        default=ARDUINO_DEFAULT_PORT,
        help='Serial port (e.g. /dev/ttyACM0 or COM3)'
    )
    parser.add_argument(
        '--baud', '-b',
        type=int,
        default=ARDUINO_DEFAULT_BAUD,
        help=f'Baud rate (default: {ARDUINO_DEFAULT_BAUD})'
    )
    args = parser.parse_args()

    if not args.port:
        print("Error: no serial port specified. Use --port /dev/ttyACM0 (or COM3 on Windows).")
        sys.exit(1)

    print(f"Opening {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"Error: could not open port — {e}")
        sys.exit(1)

    print("Waiting 2 s for Arduino to boot (DTR reset)...")
    time.sleep(2.0)
    ser.reset_input_buffer()

    stop = threading.Event()
    reader_thread = threading.Thread(target=_reader, args=(ser, stop), daemon=True)
    reader_thread.start()

    print(f"\nArduino Configure — connected to {args.port} @ {args.baud}")
    print(MENU)

    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break

            cmd = _parse(text)
            if cmd is None:
                continue
            if cmd.get("cmd") == "_quit":
                break
            if cmd.get("cmd") == "_enc_verify":
                _enc_verify(ser)
                print(MENU)
                continue
            if cmd.get("cmd") == "_enc_record":
                _enc_record(ser, cmd["duration"], cmd["save"])
                continue
            if cmd.get("cmd") == "_enc_calibrate":
                _enc_calibrate(ser)
                continue
            _send(ser, cmd)

    except KeyboardInterrupt:
        print()

    finally:
        print("Stopping motor and closing port...")
        stop.set()
        try:
            _send(ser, {"cmd": "stop"})
            time.sleep(0.1)
            ser.close()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
