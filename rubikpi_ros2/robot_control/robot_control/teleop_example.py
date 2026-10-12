#!/usr/bin/env python3
import os, serial, json, time, threading
from pynput.keyboard import Key, Listener

# ---- Serial ----
try:
    robot = serial.Serial('/dev/ttyUSB0', 115200, timeout=0)
except Exception:
    robot = None

command = {"T": 1, "L": 0.0, "R": 0.0}
pressed_keys = set()
running = True

def on_press(key):
    global command
    try:
        if hasattr(key, "char") and key.char:
            c = key.char.lower()
            pressed_keys.add(c)
            if c == 'w': command = {"T":1,"L":0.3,"R":0.3}
            elif c == 's': command = {"T":1,"L":-0.3,"R":-0.3}
            elif c == 'a': command = {"T":1,"L":-0.5,"R":0.5}
            elif c == 'd': command = {"T":1,"L":0.5,"R":-0.5}
            elif c == 'x':
                command = {"T":1,"L":0.0,"R":0.0}
                pressed_keys.clear()
    except Exception:
        pass

def on_release(key):
    global command, running
    try:
        if key == Key.esc:
            running = False
            return False  # stop listener
        if hasattr(key, "char") and key.char:
            c = key.char.lower()
            pressed_keys.discard(c)
            if not any(k in pressed_keys for k in 'wasd'):
                command = {"T":1,"L":0.0,"R":0.0}
    except Exception:
        pass

def send_commands():
    while running:
        if robot:
            try:
                robot.write((json.dumps(command) + '\n').encode('utf-8'))
            except Exception:
                pass
        time.sleep(0.05)

print("""
========================================
   🚗  Robot Controller Ready
----------------------------------------
   W : Forward
   S : Backward
   A : Turn Left
   D : Turn Right
   X : Stop
  ESC: Quit
========================================
""")

tx = threading.Thread(target=send_commands, daemon=True)
tx.start()

with Listener(on_press=on_press, on_release=on_release, suppress=True) as listener:
    listener.join()

# 退出时停车
try:
    if robot:
        robot.write(b'{"T":1,"L":0.0,"R":0.0}\n')
except Exception:
    pass
