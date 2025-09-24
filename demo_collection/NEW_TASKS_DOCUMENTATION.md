# New Tasks with Distractor Objects - Documentation

This document describes the new tasks created with distractor objects placed on top of drawers. These tasks are designed to make robot manipulation more challenging by introducing visual and physical distractors while maintaining the core task objectives.

## Comparison: Original vs New Tasks with Distractors

| **Aspect** | **Original Tasks** | **New Tasks with Distractors** | **Key Differences** |
|------------|-------------------|--------------------------------|---------------------|
| **Pick Tasks** | | | |
| Environment Names | `PickCokeCanInScene-v0`<br>`PickSpongeInScene-v0`<br>`PickAppleInScene-v0` | `PickCokeCanOnClosedDrawerInScene-v0`<br>`PickSpongeOnClosedDrawerInScene-v0`<br>`PickAppleOnClosedDrawerInScene-v0` | ✅ Objects now placed **on top of closed drawer**<br>✅ **2 distractor objects** added per task |
| Object Placement | Objects placed on **table surface** | Target + distractors placed on **closed drawer top** | ✅ More constrained workspace<br>✅ Realistic household scenario |
| Scene Complexity | **Single target object** only | **1 target + 2 distractors** (3 objects total) | ✅ Visual confusion added<br>✅ Requires object discrimination |
| Language Instructions | `"pick coke can"`<br>`"pick sponge"`<br>`"pick apple"` | Same: `"pick coke can"`<br>`"pick sponge"`<br>`"pick apple"` | ✅ **Identical instructions** maintained |
| Success Criteria | Grasp and lift target object | Grasp and lift **target object only** (ignore distractors) | ✅ Same success definition<br>✅ Distractors don't affect success |
| **Open Drawer Tasks** | | | |
| Environment Names | `OpenDrawerCustomInScene-v0`<br>`OpenTopDrawerCustomInScene-v0`<br>`OpenMiddleDrawerCustomInScene-v0`<br>`OpenBottomDrawerCustomInScene-v0` | `OpenDrawerWithObjectsCustomInScene-v0`<br>`OpenTopDrawerWithObjectsCustomInScene-v0`<br>`OpenMiddleDrawerWithObjectsCustomInScene-v0`<br>`OpenBottomDrawerWithObjectsCustomInScene-v0` | ✅ **3 distractor objects** on drawer top<br>✅ Objects may fall/move during opening |
| Initial State | **Clean closed drawer** | **Closed drawer with objects on top** | ✅ More realistic scenario<br>✅ Physics interactions |
| Task Complexity | Simple drawer opening | Drawer opening **with object dynamics** | ✅ Objects react to drawer motion<br>✅ More challenging manipulation |
| Language Instructions | `"open top drawer"`<br>`"open middle drawer"`<br>`"open bottom drawer"` | Same: `"open top drawer"`<br>`"open middle drawer"`<br>`"open bottom drawer"` | ✅ **Identical instructions** maintained |
| Success Criteria | Drawer position >= 0.15 | Same: Drawer position >= 0.15 | ✅ Objects on top **don't affect success**<br>✅ Pure drawer opening task |
| **Close Drawer Tasks** | | | |
| Environment Names | `CloseDrawerCustomInScene-v0`<br>`CloseTopDrawerCustomInScene-v0`<br>`CloseMiddleDrawerCustomInScene-v0`<br>`CloseBottomDrawerCustomInScene-v0` | `CloseDrawerWithObjectsCustomInScene-v0`<br>`CloseTopDrawerWithObjectsCustomInScene-v0`<br>`CloseMiddleDrawerWithObjectsCustomInScene-v0`<br>`CloseBottomDrawerWithObjectsCustomInScene-v0` | ✅ **3 distractor objects** on drawer top<br>✅ Objects may interfere with closing |
| Initial State | **Clean open drawer** | **Open drawer with objects on top** | ✅ Objects may obstruct closing<br>✅ Requires careful manipulation |
| Task Complexity | Simple drawer closing | Drawer closing **with potential obstructions** | ✅ Robot must work around objects<br>✅ More realistic household scenario |
| Language Instructions | `"close top drawer"`<br>`"close middle drawer"`<br>`"close bottom drawer"` | Same: `"close top drawer"`<br>`"close middle drawer"`<br>`"close bottom drawer"` | ✅ **Identical instructions** maintained |
| Success Criteria | Drawer position <= 0.05 | Same: Drawer position <= 0.05 | ✅ Objects **don't prevent success**<br>✅ Can push objects while closing |
| **Place in Drawer Tasks** | | | |
| Environment Names | `PlaceIntoClosedDrawerCustomInScene-v0`<br>`PlaceIntoClosedTopDrawerCustomInScene-v0`<br>`PlaceIntoClosedMiddleDrawerCustomInScene-v0`<br>`PlaceIntoClosedBottomDrawerCustomInScene-v0` | `PlaceIntoClosedDrawerWithObjectsCustomInScene-v0`<br>`PlaceIntoClosedTopDrawerWithObjectsCustomInScene-v0`<br>`PlaceIntoClosedMiddleDrawerWithObjectsCustomInScene-v0`<br>`PlaceIntoClosedBottomDrawerWithObjectsCustomInScene-v0` | ✅ **2 distractor objects** on drawer top<br>✅ Target object also starts on drawer |
| Object Placement | Target object on **table surface** | Target object + distractors **on closed drawer top** | ✅ All objects start in same location<br>✅ More cluttered initial state |
| Task Phases | 1. Open drawer<br>2. Pick object from table<br>3. Place in drawer | 1. Open drawer (move distractors)<br>2. Pick target from drawer top<br>3. Place in drawer | ✅ **Same two-phase structure**<br>✅ Pick location changed |
| Language Instructions | Phase 1: `"open [X] drawer"`<br>Phase 2: `"place [obj] into [X] drawer"` | Same: Phase 1: `"open [X] drawer"`<br>Phase 2: `"place [obj] into [X] drawer"` | ✅ **Identical instructions** maintained |
| Success Criteria | Sustained contact in open drawer | Same: Sustained contact in open drawer | ✅ **Same success definition**<br>✅ Distractors irrelevant for success |

## Key Innovation Summary

### **What Makes New Tasks More Challenging:**
1. **Visual Complexity**: Multiple objects create visual confusion and require object discrimination
2. **Physical Interactions**: Objects can interfere with each other and drawer mechanics
3. **Workspace Constraints**: Smaller surface area (drawer top vs full table)
4. **Dynamic Environments**: Objects move and react during task execution
5. **Realistic Scenarios**: Mimics real household environments with clutter

### **What Stays the Same:**
1. **Language Instructions**: Identical to original tasks for consistency
2. **Success Criteria**: Same evaluation metrics, distractors don't affect success
3. **Robot Control**: Same action spaces and control interfaces
4. **Episode Structure**: Same termination conditions and episode lengths
5. **Data Format**: Compatible with existing RLDS data collection pipeline

### **Distractor Object Details:**
- **Pick Tasks**: 2 distractors per task (selected from: coke can, sponge, apple)
- **Drawer Tasks**: 3 distractors per task (coke can + sponge + apple)
- **Place Tasks**: 2 distractors per task (sponge + apple, target varies)
- **Positioning**: Circular/grid patterns on drawer surface with physics settling
- **Behavior**: Objects can move, fall, or be pushed during task execution

## Overview of New Tasks

### 1. Pick Object Tasks on Closed Drawer
These tasks involve picking up a target object that is placed on top of a closed drawer, with distractor objects also present on the drawer surface.

**Available Environments:**
- `PickCokeCanOnClosedDrawerInScene-v0` - Pick a coke can with sponge and apple distractors
- `PickSpongeOnClosedDrawerInScene-v0` - Pick a sponge with coke can and apple distractors
- `PickAppleOnClosedDrawerInScene-v0` - Pick an apple with coke can and sponge distractors

### 2. Open Drawer Tasks with Objects
These tasks require opening a drawer while distractor objects are placed on top of the closed drawer.

**Available Environments:**
- `OpenDrawerWithObjectsCustomInScene-v0` - Open any drawer (top/middle/bottom) with objects on top
- `OpenTopDrawerWithObjectsCustomInScene-v0` - Open top drawer specifically with objects
- `OpenMiddleDrawerWithObjectsCustomInScene-v0` - Open middle drawer specifically with objects
- `OpenBottomDrawerWithObjectsCustomInScene-v0` - Open bottom drawer specifically with objects

### 3. Close Drawer Tasks with Objects
These tasks require closing an initially open drawer while distractor objects are placed on top of it.

**Available Environments:**
- `CloseDrawerWithObjectsCustomInScene-v0` - Close any drawer (top/middle/bottom) with objects on top
- `CloseTopDrawerWithObjectsCustomInScene-v0` - Close top drawer specifically with objects
- `CloseMiddleDrawerWithObjectsCustomInScene-v0` - Close middle drawer specifically with objects
- `CloseBottomDrawerWithObjectsCustomInScene-v0` - Close bottom drawer specifically with objects

### 4. Place in Closed Drawer Tasks with Objects
These are multi-step tasks that require: (1) opening a drawer, then (2) placing a target object into the drawer, while distractor objects are present on top of the closed drawer initially.

**Available Environments:**
- `PlaceIntoClosedDrawerWithObjectsCustomInScene-v0` - Place object into any drawer with distractors
- `PlaceIntoClosedTopDrawerWithObjectsCustomInScene-v0` - Place into top drawer with distractors
- `PlaceIntoClosedMiddleDrawerWithObjectsCustomInScene-v0` - Place into middle drawer with distractors
- `PlaceIntoClosedBottomDrawerWithObjectsCustomInScene-v0` - Place into bottom drawer with distractors

## Using rlds.py to Collect Demonstrations

### Basic Usage

To collect demonstrations using the rlds.py script, use the following command format:

```bash
cd SimplerEnv/demo_collection
conda activate simpler_env
python rlds.py <environment_name> <number_of_trajectories>
```

### Example Commands

#### Pick Tasks
```bash
# Collect 5 demonstrations of picking a coke can with distractors
# Language instruction: "pick coke can"
python rlds.py google_robot_pick_coke_can_on_drawer 5

# Collect 3 demonstrations of picking a sponge with distractors
# Language instruction: "pick sponge"
python rlds.py google_robot_pick_sponge_on_drawer 3

# Collect 10 demonstrations of picking an apple with distractors
# Language instruction: "pick apple"
python rlds.py google_robot_pick_apple_on_drawer 10
```

#### Open Drawer Tasks
```bash
# Collect 5 demonstrations of opening any drawer with objects on top
# Language instruction: "open [top/middle/bottom] drawer" (randomly selected)
python rlds.py google_robot_open_drawer_with_objects 5

# Collect 3 demonstrations of opening specifically the top drawer with objects
# Language instruction: "open top drawer"
python rlds.py google_robot_open_top_drawer_with_objects 3

# Collect 4 demonstrations of opening the middle drawer with objects
# Language instruction: "open middle drawer"
python rlds.py google_robot_open_middle_drawer_with_objects 4

# Collect 6 demonstrations of opening the bottom drawer with objects
# Language instruction: "open bottom drawer"
python rlds.py google_robot_open_bottom_drawer_with_objects 6
```

#### Close Drawer Tasks
```bash
# Collect 5 demonstrations of closing any drawer with objects on top
# Language instruction: "close [top/middle/bottom] drawer" (randomly selected)
python rlds.py google_robot_close_drawer_with_objects 5

# Collect 3 demonstrations of closing specifically the top drawer with objects
# Language instruction: "close top drawer"
python rlds.py google_robot_close_top_drawer_with_objects 3

# Collect 4 demonstrations of closing the middle drawer with objects
# Language instruction: "close middle drawer"
python rlds.py google_robot_close_middle_drawer_with_objects 4

# Collect 6 demonstrations of closing the bottom drawer with objects
# Language instruction: "close bottom drawer"
python rlds.py google_robot_close_bottom_drawer_with_objects 6
```

#### Place in Drawer Tasks
```bash
# Collect 8 demonstrations of placing object into any drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open [top/middle/bottom] drawer" (randomly selected)
#   Phase 2: "place [object_name] into [top/middle/bottom] drawer"
python rlds.py google_robot_place_in_closed_drawer_with_objects 8

# Collect 5 demonstrations of placing into top drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open top drawer"
#   Phase 2: "place [object_name] into top drawer"
python rlds.py google_robot_place_in_closed_top_drawer_with_objects 5

# Collect 6 demonstrations of placing into middle drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open middle drawer"
#   Phase 2: "place [object_name] into middle drawer"
python rlds.py google_robot_place_in_closed_middle_drawer_with_objects 6

# Collect 4 demonstrations of placing into bottom drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open bottom drawer"
#   Phase 2: "place [object_name] into bottom drawer"
python rlds.py google_robot_place_in_closed_bottom_drawer_with_objects 4
```

## Task Characteristics

### Distractor Objects
The default distractor objects used across tasks are:
- **Opened coke can** - Empty aluminum can (low density)
- **Sponge** - Soft cleaning sponge
- **Apple** - Toy apple (low density)

These objects are positioned randomly on top of the drawer surface using either:
- Circular arrangement around the drawer center (for few objects)
- Grid pattern (for many objects)

### Object Physics
- All objects have appropriate physics properties (mass, friction, damping)
- Objects fall and settle naturally on the drawer surface
- Collision detection prevents objects from intersecting
- Objects can be moved/displaced during task execution

### Success Criteria

#### Pick Tasks
- **Success**: Target object must be grasped and lifted significantly (>10cm) above its initial position
- **Failure**: Picking wrong object, dropping target object, or not lifting sufficiently

#### Open/Close Drawer Tasks
- **Open Success**: Drawer joint position >= 0.15 (fully open)
- **Close Success**: Drawer joint position <= 0.05 (fully closed)
- Objects on top may move during drawer operation but don't affect success criteria

#### Place in Drawer Tasks
- **Multi-step task**:
  1. First must open drawer (qpos >= 0.15)
  2. Then place target object with sustained contact inside drawer
- **Success**: Object maintains contact with drawer interior for 3+ timesteps while drawer is open

## Data Collection Notes

### Demonstration Quality
- The rlds.py script automatically separates successful and failed demonstrations
- Failed demonstrations can be saved for analysis if they contain useful partial progress
- All demonstrations include:
  - RGB images (224x224 for training, original resolution for visualization)
  - Robot proprioceptive state (8D: end-effector pose + gripper state)
  - Actions (7D: end-effector delta pose + gripper command)
  - Language instructions (automatically generated based on task)

### File Organization
Collected data is organized as:
```
collected_data/
└── <env_name>_<num_trajs>trajs_switch_<timestamp>_all_episodes/
    ├── successes/
    │   ├── success_episode_000001.npz
    │   ├── success_episode_000002.npz
    │   └── ...
    ├── failures/
    │   ├── failure_episode_000001.npz
    │   └── ...
    ├── captured_frames/  # Optional frame captures
    └── metadata.json
```

### Language Instructions
Each task generates appropriate language instructions based on the object naming system:

**Pick Tasks:**
- `PickCokeCanOnClosedDrawerInScene-v0`: "pick coke can"
- `PickSpongeOnClosedDrawerInScene-v0`: "pick sponge"
- `PickAppleOnClosedDrawerInScene-v0`: "pick apple"

**Open Drawer Tasks:**
- `OpenDrawerWithObjectsCustomInScene-v0`: "open [top/middle/bottom] drawer" (randomly selected each episode)
- `OpenTopDrawerWithObjectsCustomInScene-v0`: "open top drawer"
- `OpenMiddleDrawerWithObjectsCustomInScene-v0`: "open middle drawer"
- `OpenBottomDrawerWithObjectsCustomInScene-v0`: "open bottom drawer"

**Close Drawer Tasks:**
- `CloseDrawerWithObjectsCustomInScene-v0`: "close [top/middle/bottom] drawer" (randomly selected each episode)
- `CloseTopDrawerWithObjectsCustomInScene-v0`: "close top drawer"
- `CloseMiddleDrawerWithObjectsCustomInScene-v0`: "close middle drawer"
- `CloseBottomDrawerWithObjectsCustomInScene-v0`: "close bottom drawer"

**Place in Drawer Tasks (Two-Phase):**
All place tasks have two sequential language instructions:
- Phase 1 (opening): "open {drawer_id} drawer"
- Phase 2 (placing): "place {object_name} into {drawer_id} drawer"

The object names are automatically processed to remove prefixes like "opened", "light", "generated", etc., so "opened_coke_can" becomes "coke can" in instructions.

## Testing the New Environments

To verify the new environments work correctly, you can test them in Python:

```python
import simpler_env

# Test pick task
env = simpler_env.make("google_robot_pick_coke_can_on_drawer")
obs, info = env.reset()
instruction = env.get_language_instruction()
print(f"Task: {instruction}")
env.close()

# Test drawer task
env = simpler_env.make("google_robot_open_drawer_with_objects")
obs, info = env.reset()
instruction = env.get_language_instruction()
print(f"Task: {instruction}")
env.close()

# Test place task
env = simpler_env.make("google_robot_place_in_closed_drawer_with_objects")
obs, info = env.reset()
instruction = env.get_language_instruction()
print(f"Task: {instruction}")
env.close()
```

## Important Notes

1. **Environment Dependencies**: Make sure to activate the `simpler_env` conda environment before running rlds.py

2. **Hardware Requirements**: The tasks require a Nintendo Switch Pro Controller connected via the controller streaming setup

3. **Scene Compatibility**: All new environments are compatible with the existing ManiSkill2 robot and scene configurations

4. **Randomization**: Object positions, orientations, and drawer selection are randomized across episodes for better generalization

5. **Safety**: Objects are placed to avoid collisions with the robot's initial position and fall naturally onto surfaces

These new tasks provide increased complexity while maintaining the same interaction paradigms as the original SimplerEnv tasks, making them suitable for training more robust manipulation policies.