## **Picking Tasks** (70 episodes total)

```bash
# Standing coke can pickup (upright grasp)
python rlds.py google_robot_pick_standing_coke_can 10

# Pick apple (specific object)
python rlds.py google_robot_pick_apple 10

# Pick sponge (specific object)
python rlds.py google_robot_pick_sponge 10
```

## **Drawer Tasks**

### Opening drawers (different levels) - 60 episodes
```bash
# Top drawer opening (highest reach)
python rlds.py google_robot_open_top_drawer 20

# Middle drawer opening (mid-level reach)
python rlds.py google_robot_open_middle_drawer 20

# Bottom drawer opening (low reach, challenging)
python rlds.py google_robot_open_bottom_drawer 20
```

### Closing drawers (different levels) - 60 episodes
```bash
# Top drawer closing
python rlds.py google_robot_close_top_drawer 20

# Middle drawer closing
python rlds.py google_robot_close_middle_drawer 20

# Bottom drawer closing
python rlds.py google_robot_close_bottom_drawer 20
```

## **Placement Tasks** - 40 episodes

```bash
# Place apple in closed top drawer (pick + place + open sequence)
python rlds.py google_robot_place_apple_in_closed_top_drawer 20

# General placement in closed drawer
python rlds.py google_robot_place_in_closed_drawer 20
```

## **Moving Tasks** - 20 episodes

```bash
# Move objects near target locations
python rlds.py google_robot_move_near 20
```