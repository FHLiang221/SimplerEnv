# Jaco Arm Data Collection Commands

## **Prerequisites**

### 1. Hardware Setup
```bash
# Ensure Nintendo Switch controller is paired and connected
# Controller should be running the TCP bridge on port 5555
# Verify connection: controller should send action data to 0.0.0.0:5555
```

### 2. Environment Setup
```bash
cd /project/fhliang/projects/SimplerEnv/demo_collection
# Ensure simpler_env is installed and working
python -c "import simpler_env; print('SimplerEnv ready')"
```

---

## **Picking Tasks** (50 episodes total)

```bash
# Horizontal coke can pickup (good for grasp diversity)
python rlds_jaco.py google_robot_pick_horizontal_coke_can 10

# Vertical coke can pickup (different orientation)
python rlds_jaco.py google_robot_pick_vertical_coke_can 10

# Standing coke can pickup (upright grasp)
python rlds_jaco.py google_robot_pick_standing_coke_can 10

# General coke can pickup (randomized orientation)
python rlds_jaco.py google_robot_pick_coke_can 10

# Generic object pickup (diverse objects)
python rlds_jaco.py google_robot_pick_object 20
```

## **Drawer Tasks**

### Opening drawers (different levels) - 60 episodes
```bash
# Top drawer opening (highest reach)
python rlds_jaco.py google_robot_open_top_drawer 20

# Middle drawer opening (mid-level reach)
python rlds_jaco.py google_robot_open_middle_drawer 20

# Bottom drawer opening (low reach, challenging)
python rlds_jaco.py google_robot_open_bottom_drawer 20
```

### Closing drawers (different levels) - 60 episodes
```bash
# Top drawer closing
python rlds_jaco.py google_robot_close_top_drawer 20

# Middle drawer closing
python rlds_jaco.py google_robot_close_middle_drawer 20

# Bottom drawer closing
python rlds_jaco.py google_robot_close_bottom_drawer 20
```

## **Placement Tasks** - 40 episodes

```bash
# Place apple in closed top drawer (pick + place + open sequence)
python rlds_jaco.py google_robot_place_apple_in_closed_top_drawer 20

# General placement in closed drawer
python rlds_jaco.py google_robot_place_in_closed_drawer 20
```

## **Moving Tasks** - 20 episodes

```bash
# Move objects near target locations
python rlds_jaco.py google_robot_move_near 20
```

---

## **Jaco-Specific Features**

### **Robot Configuration**
- **Action Space**: 7D (6D end-effector pose delta + 1D gripper)
- **Proprioception**: 8D (EEF pose 7D + gripper state 1D)
- **Action Chunks**: 8 steps (matches LIBERO format for OpenVLA compatibility)

### **Data Organization**
```bash
# Data saved to: /project/fhliang/projects/SimplerEnv/demo_collection/jaco_data/
# Structure:
#   jaco_{env_name}_{num_trajs}trajs_switch_{timestamp}_all_episodes/
#   ├── successes/     # Training data (successful demonstrations)
#   ├── failures/      # Analysis data (failure cases for debugging)
#   └── jaco_metadata.json  # Dataset statistics and configuration
```

### **Quality Control**
- **Success Episodes**: Automatically saved for training
- **Failure Episodes**: Interactive decision (+ to save, - to discard)
- **Empty Episodes**: Automatically discarded
- **Metadata Tracking**: Instructions, success rates, episode counts

---

## **Controls Reminder for Jaco**

- **Left stick + L/ZL**: Translation (xyz movement)
- **Right stick + R/ZR**: Rotation (roll/pitch/yaw)
- **A/B buttons**: Gripper open/close
- **+ button**: Start episode / End episode / Save failure
- **- button**: Abort episode / Discard failure

---

## **Post-Collection Workflow**

### 1. **Verify Data Quality**
```bash
# Check collected data statistics
ls -la /project/fhliang/projects/SimplerEnv/demo_collection/jaco_data/
cat /path/to/your/dataset/jaco_metadata.json | jq '.num_successful_episodes'
```

### 2. **Build RLDS Dataset**
```bash
cd /project/fhliang/projects/rlds_dataset_builder/jaco_dataset
tfds build --overwrite

# Verify dataset creation
python ../visualize_dataset.py jaco_dataset
```

### 3. **Dataset Location**
```bash
# Built dataset location:
~/tensorflow_datasets/jaco_dataset/1.0.0/
```

### 4. **Training Integration**
```bash
cd /project/fhliang/projects/openvla-oft

# Fine-tune with Jaco data (auto-detects "jaco" in command)
python vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --dataset_name "jaco_dataset" \
    --run_root_dir "./runs/jaco_finetune" \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 50000
```

---

## **Collection Strategy Recommendations**

### **Balanced Dataset (230 total episodes)**
1. **Picking**: 50 episodes (diverse grasping patterns)
2. **Drawer Opening**: 60 episodes (spatial reasoning)
3. **Drawer Closing**: 60 episodes (precision movements)
4. **Placement**: 40 episodes (complex sequences)
5. **Moving**: 20 episodes (object manipulation)

### **Session Planning**
```bash
# Session 1: Picking tasks (easier warm-up)
python rlds_jaco.py google_robot_pick_standing_coke_can 10
python rlds_jaco.py google_robot_pick_object 20

# Session 2: Drawer operations (moderate difficulty)  
python rlds_jaco.py google_robot_close_top_drawer 20
python rlds_jaco.py google_robot_open_middle_drawer 20

# Session 3: Complex tasks (higher difficulty)
python rlds_jaco.py google_robot_place_apple_in_closed_top_drawer 20
python rlds_jaco.py google_robot_move_near 20
```

---

## **Troubleshooting**

### **Controller Connection Issues**
```bash
# Verify TCP connection
netstat -an | grep 5555
# Should show: tcp 0.0.0.0:5555 LISTEN
```

### **Data Collection Issues**
```bash
# Check episode statistics in real-time
tail -f /path/to/your/dataset/jaco_metadata.json

# Monitor success rates during collection
# Target: >60% success rate for training effectiveness
```

### **OpenVLA Integration**
```bash
# Verify Jaco constants are loaded correctly
cd /project/fhliang/projects/openvla-oft
python -c "from prismatic.vla.constants import *; print(f'Action: {ACTION_DIM}, Proprio: {PROPRIO_DIM}')"
# Should output: Action: 7, Proprio: 8
```