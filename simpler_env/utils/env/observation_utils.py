def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        elif "jaco" in env.robot_uid or "ada" in env.robot_uid:
            # Jaco environments typically have 3rd_view_camera or base_camera
            if "3rd_view_camera" in obs["image"]:
                camera_name = "3rd_view_camera"
            elif "base_camera" in obs["image"]:
                camera_name = "base_camera"
            else:
                camera_name = "render_camera"  # fallback
        else:
            raise NotImplementedError(f"No default camera defined for robot: {env.robot_uid}")
    return obs["image"][camera_name]["rgb"]
