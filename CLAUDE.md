# SimplerEnv Demo Collection - Progress Summary

## ✅ Completed Tasks

### Demo Collection System Working for Coke Can Tasks
- **Environment Mapping**: Updated `rlds_jaco.py` to use consistent environment mappings that match the manual control setup
- **Scene Configuration**: Implemented dynamic scene selection based on task type:
  - Coke can tasks → `google_pick_coke_can_1_v4` scene
  - Drawer tasks → `frl_apartment_stage_simple` scene  
  - Bridge tasks → `bridge_table_1_v1`/`bridge_table_1_v2` scenes
- **Robot Positioning**: Added proper environment reset options for Jaco robot positioning (`init_xy: [-0.45, 0.6]`)
- **Visual Consistency**: Fixed coke can tasks to use `GraspSingleCokeCanInScene-v0` which includes custom visual elements (dark table + gray walls)

### Controller Improvements
- **Gripper Control**: Implemented discrete gripper actions with threshold detection for better control
- **Delta Target Control**: Added gripper action reset for delta target control mode to prevent continuous movement
- **Sensitivity Tuning**: Reduced controller sensitivity in `jaco_modded.py`:
  - Translation speed: `0.008` (reduced from `0.015`)
  - Rotation speed: `0.04` (reduced from `0.08`)
  - Gripper sensitivity: `0.03` (reduced from `0.05`)

### Environment Analysis & Visual Consistency Fixes
- **Scene Background Investigation**: Identified that `GraspSingleCokeCanInScene-v0` has special visual decorations that other environments lack
- **Manual Control Alignment**: Modified manual control script (`demo_manual2.py`) to use unified Jaco robot positioning across tasks
- **Real-World Visual Matching**: Added black table and gray walls to match real-world setup across all primary task environments

#### Black Table & Gray Walls Implementation - FIXED ✅
**Problem**: Only `GraspSingleCokeCanInScene-v0` had the black table and gray walls needed to match the real-world robot setup. Other key environments (`GraspSingleRandomObjectInScene-v0` and `MoveNearGoogleBakedTexInScene-v1`) were missing these visual elements.

**Solution**: Added programmatic visual elements to environments in their `_load_actors()` methods (not `__init__()` - timing issue):
```python
# Black table
builder.add_box_visual(half_size=[5, 5, 0.87], color=[0.05, 0.05, 0.05])
dummy_table.set_pose(sapien.Pose([0, 0, 0.019], [0, 0, 0, 1]))

# Gray walls (2 walls)  
builder.add_box_visual(half_size=[2, 0.01, 3], color=[0.48, 0.48, 0.48])
wall1.set_pose(sapien.Pose([0, -0.28, 0], [0, 0, 0, 1]))
builder.add_box_visual(half_size=[2, 0.01, 3], color=[0.48, 0.48, 0.48])
wall2.set_pose(sapien.Pose([0, 0.76, 0], [0, 0, 0, 1]))
```

**Key Fix**: Visual elements must be added in `_load_actors()` after scene initialization, not in `__init__()`

**Files Modified**:
- `/project/fhliang/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/grasp_single_in_scene.py` (lines 643-659)

**Result**: `google_robot_pick_object` task now displays black table and gray walls correctly ✅

## 📋 TODO Tasks

### Environment Fixes for Demo Collection
- [ ] **Fix move near scene**: Add black table and gray walls to `MoveNearGoogleBakedTexInScene-v1` for visual consistency
- [ ] **Fix specific object scenes**: Add visual elements to individual object environments:
  - [ ] Apple picking task environments
  - [ ] Sponge picking task environments  
- [ ] **Fix drawer task environments**: All drawer-related tasks need proper scene and positioning configuration

## 🔧 Key Configuration Files

- **`demo_collection/rlds_jaco.py`**: Main data collection script with environment mappings
- **`demo_collection/jaco_modded.py`**: Controller sensitivity settings  
- **`demo_collection/demo_manual2.py`**: Modified manual control script with unified positioning
- **`ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/grasp_single_in_scene.py`**: Environment definitions

## 🎯 Current Status

**Working**: Coke can and random object demo collection with proper visual scenes and controller sensitivity ✅
**Priority**: Fix move near task, specific object tasks (apple, sponge), and drawer tasks
**Goal**: Complete data collection pipeline for all task types with consistent real-world visual matching