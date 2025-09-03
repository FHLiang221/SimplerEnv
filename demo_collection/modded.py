#!/usr/bin/env python3
"""
mac_controller.py - Nintendo Switch Pro Controller for Robot Teleoperation
Stream 8-float action packets via SSH tunnel at optimized rate.

SSH Tunnel Command:
ssh -N -L 60000:localhost:5555 -o Compression=no -o TCPKeepAlive=yes -o ServerAliveInterval=10 fhliang@10.136.109.136 -p 42522
s
Controls (Intuitive Left/Right Hand Separation)
────────────────────────────────────────────────────────────
LEFT HAND - TRANSLATION:
  Left Stick X/Y        : X/Y translate (forward/back, left/right)
  L Button              : Z translate UP
  ZL Trigger            : Z translate DOWN

RIGHT HAND - ROTATION:
  Right Stick X         : Yaw rotate (turn left/right)
  Right Stick Y         : Pitch rotate (nose up/down)
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
BTN_L = 9          # Z up (Left Bumper)
BTN_R = 10         # Roll left (Right Bumper)

# Trigger Axes (analog 0.0 to 1.0)
AXIS_ZL = 4        # Z down (Left Trigger)
AXIS_ZR = 5        # Roll right (Right Trigger)

# ── Control parameters ─────────────────────────────────────────────────
VEL_TRANSLATE = 0.03    # Base translation speed
VEL_ROTATE = 0.3       # Base rotation speed  
VEL_GRIPPER = 0.1     # Gripper sensitivity
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
    """Generate 10-float action vector from controller input."""
    pygame.event.pump()
    
    # Get speed modifier
    speed_mult = get_speed_multiplier()
    
    # ── LEFT HAND: TRANSLATION ─────────────────────────────────────────
    dx = VEL_TRANSLATE * (-axis(AX_LY)) * speed_mult    # Left stick up/down → forward/back
    dy = VEL_TRANSLATE * (-axis(AX_LX)) * speed_mult     # Left stick left/right → left/right
    
    # Z movement on left shoulder buttons
    dz = 0.0
    if js.get_button(BTN_L):                            # L button → up
        dz = VEL_TRANSLATE * speed_mult
    elif js.get_axis(AXIS_ZL) > 0.1:                    # ZL trigger (analog) → down
        dz = -VEL_TRANSLATE * speed_mult
    
    # ── RIGHT HAND: ROTATION ────────────────────────────────────────────
    dyaw = VEL_ROTATE * axis(AX_RX) * speed_mult        # Right stick left/right → yaw
    dpitch = VEL_ROTATE * (-axis(AX_RY)) * speed_mult   # Right stick up/down → pitch
    
    # Roll on right shoulder buttons
    droll = 0.0
    if js.get_button(BTN_R):                            # R button → roll left
        droll = -VEL_ROTATE * speed_mult
    elif js.get_axis(AXIS_ZR) > 0.1:                    # ZR trigger (analog) → roll right
        droll = VEL_ROTATE * speed_mult
    
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
    print("\n" + "="*60)
    print("NINTENDO SWITCH PRO CONTROLLER - ROBOT TELEOPERATION")
    print("="*60)
    print("LEFT HAND (Translation):")
    print("  Left Stick ↑↓  → Forward/Backward")
    print("  Left Stick ←→  → Left/Right") 
    print("  L Button       → Move UP")
    print("  ZL Trigger     → Move DOWN")
    print()
    print("RIGHT HAND (Rotation):")
    print("  Right Stick ←→ → Turn Left/Right (Yaw)")
    print("  Right Stick ↑↓ → Pitch Up/Down")
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
    print("="*60)

def main():
    """Main teleoperation loop."""
    print_controls()
    
    # Establish connection
    sock = connect()
    frame_time = 1.0 / HZ
    
    print(f"\nStreaming at {HZ} Hz — Press Ctrl+C to stop")
    print("Move controller to start sending commands...")
    
    try:
        while True:
            # Get controller input
            action_packet = get_action()
            
            # Debug output (remove this line when satisfied)
            print(f"\rSEND: [{action_packet[0]:+.2f}, {action_packet[1]:+.2f}, {action_packet[2]:+.2f}, "
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
        print("\n\nTeleoperation stopped by user.")
    finally:
        sock.close()
        pygame.quit()
        print("Controller disconnected. Goodbye!")

if __name__ == "__main__":
    main()

# 1. Open terminal

# for Tengen:
    # ssh -N -L 60000:localhost:5555 fhliang@10.136.109.136 -p 42522
# for blackCoffee:
    # ssh -N -L 60000:localhost:5555 fhliang@10.136.109.242 -p 42522

# 2. Open another terminal
# conda env: switch, modded.py