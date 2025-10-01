import gymnasium as gym
import mani_skill2_real2sim.envs
import warnings

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    # New environments with distractor objects
    "google_robot_pick_coke_can_on_drawer",
    "google_robot_pick_sponge_on_drawer",
    "google_robot_pick_apple_on_drawer",
    "google_robot_open_drawer_with_objects",
    "google_robot_open_top_drawer_with_objects",
    "google_robot_open_middle_drawer_with_objects",
    "google_robot_open_bottom_drawer_with_objects",
    "google_robot_close_drawer_with_objects",
    "google_robot_close_top_drawer_with_objects",
    "google_robot_close_middle_drawer_with_objects",
    "google_robot_close_bottom_drawer_with_objects",
    "google_robot_place_in_closed_drawer_with_objects",
    "google_robot_place_in_closed_top_drawer_with_objects",
    "google_robot_place_in_closed_middle_drawer_with_objects",
    "google_robot_place_in_closed_bottom_drawer_with_objects",
    # Multi-object scene environments
    "google_robot_pick_coke_can_multi_object",
    "google_robot_pick_apple_multi_object",
    "google_robot_pick_sponge_multi_object",
    "google_robot_open_top_drawer_multi_object",
    "google_robot_close_bottom_drawer_multi_object",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0",
        {"model_ids": "baked_apple_v2"}
    ),
    # New environments with distractor objects
    "google_robot_pick_coke_can_on_drawer": ("PickCokeCanOnClosedDrawerInScene-v0", {}),
    "google_robot_pick_sponge_on_drawer": ("PickSpongeOnClosedDrawerInScene-v0", {}),
    "google_robot_pick_apple_on_drawer": ("PickAppleOnClosedDrawerInScene-v0", {}),
    "google_robot_open_drawer_with_objects": ("OpenDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_open_top_drawer_with_objects": ("OpenTopDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_open_middle_drawer_with_objects": ("OpenMiddleDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer_with_objects": ("OpenBottomDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_close_drawer_with_objects": ("CloseDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_close_top_drawer_with_objects": ("CloseTopDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_close_middle_drawer_with_objects": ("CloseMiddleDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer_with_objects": ("CloseBottomDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer_with_objects": ("PlaceIntoClosedDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer_with_objects": ("PlaceIntoClosedTopDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer_with_objects": ("PlaceIntoClosedMiddleDrawerWithObjectsCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer_with_objects": ("PlaceIntoClosedBottomDrawerWithObjectsCustomInScene-v0", {}),
    # Multi-object scene environments
    "google_robot_pick_coke_can_multi_object": ("MultiObjectGraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_apple_multi_object": ("MultiObjectGraspSingleAppleInScene-v0", {}),
    "google_robot_pick_sponge_multi_object": ("MultiObjectGraspSingleSpongeInScene-v0", {}),
    "google_robot_open_top_drawer_multi_object": ("MultiObjectOpenTopDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer_multi_object": ("MultiObjectCloseBottomDrawerCustomInScene-v0", {}),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
}


def make(task_name, **kwargs):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, env_kwargs = ENVIRONMENT_MAP[task_name]
    
    env_kwargs["obs_mode"] = "rgbd",
    env_kwargs["prepackaged_config"] = True

    for key, value in kwargs.items():
        if key in env_kwargs:
            warnings.warn(f"default value [{env_kwargs[key]}] for Key {key} changes to value [{value}]")
        env_kwargs[key] = value

    env = gym.make(env_name, **env_kwargs)
    return env
