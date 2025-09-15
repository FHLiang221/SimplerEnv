#!/usr/bin/env python3
"""
EE_modded.py - Nintendo Switch Pro Controller for Robot Teleoperation with End-Effector Frame Control
Stream 8-float action packets via SSH tunnel at optimized rate.

SSH Tunnel Command:
ssh -N -L 60000:localhost:5555 -o Compression=no -o TCPKeepAlive=yes -o ServerAliveInterval=10 fhliang@10.136.109.136 -p 42022

Controls (End-Effector Frame Translation + World Frame Rotation)
──────────────────────────────────────────────────────────────────────────
LEFT HAND - TRANSLATION (relative to gripper orientation):
  Left Stick X          : Left/Right relative to gripper
  Left Stick Y          : Forward/Backward relative to gripper
  L Button              : Up relative to gripper
  ZL Trigger            : Down relative to gripper

RIGHT HAND - ROTATION (world frame):
  Right Stick X         : Yaw rotate (left/right)
  Right Stick Y         : Pitch rotate (up/down) - normal
  R Button              : Roll rotate LEFT
  ZR Trigger            : Roll rotate RIGHT

GRIPPER & CONTROL:
  A Button              : Close gripper
  B Button              : Open gripper
  + Button              : Start episode / Abort episode / Save failure
  - Button              : Abort episode / Discard failure  
  Home Button           : (unused)

SPEED CONTROL:
  Y Button (hold)       : Precision mode (50% speed)
  X Button (hold)       : Fast mode (200% speed)
  Normal                : 100% speed

NOTE: EE.py will handle the frame transformation. This controller still sends
      world-frame commands, but they represent desired movement relative to 
      gripper orientation.
"""

import time, socket, struct, pygame, numpy as np, sys

# Connection settings
DEST_IP   = "127.0.0.1"   # SSH tunnel endpoint
DEST_PORT = 60000
HZ        = 5            # Optimized update rate

# ── Controller initialization ──────────────────────────────────────────
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No Nintendo Switch controller found on the Mac!")

js = pygame.joystick.Joystick(0)
js.init()

print(f"Controller connected: {js.get_name()}")
print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}")

# ── Controller mapping (Nintendo Switch Pro Controller) ────────────────
# Axes
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3  # Left stick XY, Right stick XY

# Buttons  
BTN_A = 0          # Close gripper
BTN_B = 1          # Open gripper
BTN_X = 2          # Fast mode modifier
BTN_Y = 3          # Precision mode modifier
BTN_MINUS = 4      # Discard episode
BTN_HOME = 5       # Home button
BTN_PLUS = 6       # Quit episode
BTN_L = 9          # Forward relative to gripper (Left Bumper)
BTN_R = 10         # Yaw left (Right Bumper)

# Trigger Axes (analog 0.0 to 1.0)
AXIS_ZL = 4        # Backward relative to gripper (Left Trigger)
AXIS_ZR = 5        # Yaw right (Right Trigger)

# ── Control parameters (tuned for EE frame control) ────────────────────
VEL_TRANSLATE = 0.004   # Base translation speed (50% slower than original 0.008)
VEL_ROTATE = 0.02      # Base rotation speed (50% slower than original 0.04)
VEL_GRIPPER = 0.015    # Gripper sensitivity (50% slower than original 0.03)
DEADZONE = 0.05        # Stick deadzone

# Speed modifiers
PRECISION_MULTIPLIER = 0.5   # Y button: 50% speed
FAST_MULTIPLIER = 2.0        # X button: 200% speed

def axis(i):
    """Get axis value with deadzone filtering."""
    v = js.get_axis(i)
    return v if abs(v) > DEADZONE else 0.0

def get_speed_multiplier():
    """Calculate speed multiplier based on modifier buttons."""
    if js.get_button(BTN_Y):     # Y button: precision mode
        return PRECISION_MULTIPLIER
    elif js.get_button(BTN_X):   # X button: fast mode  
        return FAST_MULTIPLIER
    else:
        return 1.0               # Normal speed

def get_action():
    """Generate 10-float action vector from controller input for EE frame control."""
    pygame.event.pump()
    
    # Get speed modifier
    speed_mult = get_speed_multiplier()
    
    # ── LEFT HAND: TRANSLATION (will be transformed to EE frame by EE.py) ──
    # These commands represent desired movement relative to gripper orientation
    # but are sent in world frame coordinates for EE.py to transform
    
    # Up/down movement on left shoulder buttons (relative to gripper)
    dz = 0.0
    if js.get_button(BTN_L):                            # L button → up relative to gripper
        dz = VEL_TRANSLATE * speed_mult
    elif js.get_axis(AXIS_ZL) > 0.1:                    # ZL trigger → down relative to gripper
        dz = -VEL_TRANSLATE * speed_mult
    
    # Left/right movement on left stick X (relative to gripper)
    dy = VEL_TRANSLATE * (axis(AX_LX)) * speed_mult      # Left stick left/right → Y left/right (normal)
    
    # Forward/back movement on left stick Y (relative to gripper)
    dx = VEL_TRANSLATE * (axis(AX_LY)) * speed_mult      # Left stick up/down → X forward/back (inverted)
    
    # ── RIGHT HAND: ROTATION (world frame - swapped controls) ─────────────────
    # Roll on right shoulder buttons
    droll = 0.0
    if js.get_button(BTN_R):                            # R button → roll left
        droll = VEL_ROTATE * speed_mult
    elif js.get_axis(AXIS_ZR) > 0.1:                    # ZR trigger → roll right
        droll = -VEL_ROTATE * speed_mult

    dpitch = VEL_ROTATE * (-axis(AX_RY)) * speed_mult   # Right stick up/down → pitch (normal, not inverted)
    dyaw = VEL_ROTATE * axis(AX_RX) * speed_mult        # Right stick left/right → yaw
    
    # ── GRIPPER & CONTROL ───────────────────────────────────────────────
    grip = 0.0
    if js.get_button(BTN_A):                            # A button → close
        grip = -VEL_GRIPPER
    elif js.get_button(BTN_B):                          # B button → open
        grip = VEL_GRIPPER
    
    # Control flags
    plusf = 1.0 if js.get_button(BTN_PLUS) else 0.0    # + button → quit/save
    startf = 1.0 if js.get_button(BTN_HOME) else 0.0   # Home button → start episode
    minusf = 1.0 if js.get_button(BTN_MINUS) else 0.0  # - button → discard
    
    return np.array([dx, dy, dz, droll, dpitch, dyaw, grip, plusf, startf, minusf], dtype=np.float32)

def connect():
    """Establish optimized TCP connection with retry logic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Optimize for low latency
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)  # Small send buffer
    
    while True:
        try:
            s.connect((DEST_IP, DEST_PORT))
            print(f"Connected to {DEST_IP}:{DEST_PORT}")
            return s
        except OSError as e:
            print(f"Waiting for server ({e.strerror}) ...")
            time.sleep(1)

def print_controls():
    """Display control mapping for reference."""
    print("\n" + "="*70)
    print("NINTENDO SWITCH PRO CONTROLLER - EE FRAME ROBOT TELEOPERATION")
    print("="*70)
    print("LEFT HAND (Translation - relative to gripper orientation):")
    print("  Left Stick ←→  → Left/Right relative to gripper")
    print("  Left Stick ↑↓  → Forward/Backward relative to gripper") 
    print("  L Button       → Up relative to gripper")
    print("  ZL Trigger     → Down relative to gripper")
    print()
    print("RIGHT HAND (Rotation - world frame):")
    print("  Right Stick ←→ → Yaw Left/Right")
    print("  Right Stick ↑↓ → Pitch Up/Down (normal)")
    print("  R Button       → Roll Left")
    print("  ZR Trigger     → Roll Right")
    print()
    print("GRIPPER & CONTROL:")
    print("  A Button       → Close Gripper")
    print("  B Button       → Open Gripper")
    print("  + Button       → Start Episode / Abort Episode / Save Failure")
    print("  - Button       → Abort Episode / Discard Failure")
    print("  Home Button    → (unused)")
    print()
    print("SPEED CONTROL:")
    print("  Y Button (hold) → Precision Mode (50% speed)")
    print("  X Button (hold) → Fast Mode (200% speed)")
    print("  Normal          → 100% speed")
    print()
    print("🧭 END-EFFECTOR FRAME MODE:")
    print("   Translation commands adapt to gripper orientation!")
    print("   'Forward' always moves in gripper's forward direction")
    print("   EE.py handles the coordinate transformation automatically")
    print("="*70)

def main():
    """Main teleoperation loop."""
    print_controls()
    
    # Establish connection
    sock = connect()
    frame_time = 1.0 / HZ
    
    print(f"\nStreaming at {HZ} Hz for EE Frame Control — Press Ctrl+C to stop")
    print("🧭 EE FRAME MODE: Translations adapt to gripper orientation!")
    print("Move controller to start sending commands...")
    
    try:
        while True:
            # Get controller input
            action_packet = get_action()
            
            # Debug output with EE frame indicator
            print(f"\r🧭 EE-SEND: [{action_packet[0]:+.2f}, {action_packet[1]:+.2f}, {action_packet[2]:+.2f}, "
                  f"{action_packet[3]:+.2f}, {action_packet[4]:+.2f}, {action_packet[5]:+.2f}, "
                  f"{action_packet[6]:+.2f}, {action_packet[7]:.0f}, {action_packet[8]:.0f}, {action_packet[9]:.0f}]", end="")
            
            # Send data
            try:
                sock.sendall(struct.pack("<10f", *action_packet))  # Now 10 floats
            except (BrokenPipeError, ConnectionResetError):
                print("\nConnection lost — reconnecting...")
                sock.close()
                sock = connect()
                continue
            
            # Rate limiting
            time.sleep(frame_time)
            
    except KeyboardInterrupt:
        print("\n\nEE Frame Teleoperation stopped by user.")
    finally:
        sock.close()
        pygame.quit()
        print("EE Frame Controller disconnected. Goodbye!")

if __name__ == "__main__":
    main()

# Usage Instructions:
# 1. Open terminal and establish SSH tunnel:
#
# for Tengen:
    # ssh -N -L 60000:localhost:5555 fhliang@10.136.109.136 -p 42522
# for Primus:
    # ssh -N -L 60000:localhost:5555 fhliang@10.136.109.136 -p 42022
# for blackCoffee:
    # ssh -N -L 60000:localhost:5555 fhliang@10.136.109.242 -p 42522
#
# 2. Open another terminal and run:
# conda activate switch
# python EE_modded.py
#
# 3. In robot terminal, run:
# python EE.py google_robot_pick_horizontal_coke_can 10
#
# 🧭 EE Frame Control Benefits:
# - Translation commands always relative to gripper orientation
# - "Forward" button always moves gripper forward regardless of robot pose  
# - Much more intuitive control when gripper is rotated
# - Rotation commands remain in world frame for consistent behavior