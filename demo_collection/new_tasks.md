#### Pick Tasks
```bash
# Collect 5 demonstrations of picking a coke can with distractors
# Language instruction: "pick coke can"
python simple.py google_robot_pick_coke_can_on_drawer 5

# Collect 3 demonstrations of picking a sponge with distractors
# Language instruction: "pick sponge"
python simple.py google_robot_pick_sponge_on_drawer 3

# Collect 10 demonstrations of picking an apple with distractors
# Language instruction: "pick apple"
python simple.py google_robot_pick_apple_on_drawer 10
```

#### Open Drawer Tasks
```bash
# Collect 5 demonstrations of opening any drawer with objects on top
# Language instruction: "open [top/middle/bottom] drawer" (randomly selected)
python simple.py google_robot_open_drawer_with_objects 5

# Collect 3 demonstrations of opening specifically the top drawer with objects
# Language instruction: "open top drawer"
python simple.py google_robot_open_top_drawer_with_objects 3

# Collect 4 demonstrations of opening the middle drawer with objects
# Language instruction: "open middle drawer"
python simple.py google_robot_open_middle_drawer_with_objects 4

# Collect 6 demonstrations of opening the bottom drawer with objects
# Language instruction: "open bottom drawer"
python simple.py google_robot_open_bottom_drawer_with_objects 6
```

#### Close Drawer Tasks
```bash
# Collect 5 demonstrations of closing any drawer with objects on top
# Language instruction: "close [top/middle/bottom] drawer" (randomly selected)
python simple.py google_robot_close_drawer_with_objects 5

# Collect 3 demonstrations of closing specifically the top drawer with objects
# Language instruction: "close top drawer"
python simple.py google_robot_close_top_drawer_with_objects 3

# Collect 4 demonstrations of closing the middle drawer with objects
# Language instruction: "close middle drawer"
python simple.py google_robot_close_middle_drawer_with_objects 4

# Collect 6 demonstrations of closing the bottom drawer with objects
# Language instruction: "close bottom drawer"
python simple.py google_robot_close_bottom_drawer_with_objects 6
```

#### Place in Drawer Tasks
```bash
# Collect 8 demonstrations of placing object into any drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open [top/middle/bottom] drawer" (randomly selected)
#   Phase 2: "place [object_name] into [top/middle/bottom] drawer"
python simple.py google_robot_place_in_closed_drawer_with_objects 8

# Collect 5 demonstrations of placing into top drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open top drawer"
#   Phase 2: "place [object_name] into top drawer"
python simple.py google_robot_place_in_closed_top_drawer_with_objects 5

# Collect 6 demonstrations of placing into middle drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open middle drawer"
#   Phase 2: "place [object_name] into middle drawer"
python simple.py google_robot_place_in_closed_middle_drawer_with_objects 6

# Collect 4 demonstrations of placing into bottom drawer with distractors
# Language instruction: Two phases:
#   Phase 1: "open bottom drawer"
#   Phase 2: "place [object_name] into bottom drawer"
python simple.py google_robot_place_in_closed_bottom_drawer_with_objects 4
```