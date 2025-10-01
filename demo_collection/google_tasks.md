## **Multi-Object Scene Tasks** - 250 episodes

### Multi-Object Picking Tasks - 150 episodes
```bash
# Pick opened coke can in multi-object scene (upright grasp)
python rlds.py google_robot_pick_coke_can_multi_object 50

# Pick apple in multi-object scene (specific object)
python rlds.py google_robot_pick_apple_multi_object 50

# Pick sponge in multi-object scene (specific object)
python rlds.py google_robot_pick_sponge_multi_object 50
```

### Multi-Object Drawer Tasks - 100 episodes
```bash
# Open top drawer in multi-object scene (highest reach)
python rlds.py google_robot_open_top_drawer_multi_object 50

# Close bottom drawer in multi-object scene (low reach, challenging)
python rlds.py google_robot_close_bottom_drawer_multi_object 50
```