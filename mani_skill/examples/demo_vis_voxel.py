import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
import torch
import tqdm
import cv2
from mani_skill.utils.visualization.voxel_visualizer import visualise_voxel

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--voxel_size", type=int, default=150, help="The number of voxels per side")
    parser.add_argument("--video_path", type=str, default="output.mp4", help="The path to save the voxelization output video")
    parser.add_argument("--zoom_factor", type=float, default=1.0, help="Zoom factor of the camera when generating the output voxel visualizations")
    parser.add_argument(
        "--coord_bounds",
        nargs="*",
        type=float,
        default=[-7, -5, -0.5, 1, 5, 3],
        help="Whether or not to perform voxel segmentation estimations and include them in the results"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

def render_filtered_voxels(voxel_grid, zf=1.0):
    flood_id = 17
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    floor_map = (vis_voxel_grid[:, 9, ...] == flood_id)
    floor_map = torch.tensor(floor_map)
    floor_map = floor_map.unsqueeze(1).repeat(1, 11, 1, 1, 1)
    vis_voxel_grid[floor_map] = 0
    rotation_amount = 60
    rendered_img = visualise_voxel(vis_voxel_grid[0],
                                None,
                                None,
                                voxel_size=0.01,
                                zoom_factor=zf,
                                rotation_amount=np.deg2rad(rotation_amount))
    return rendered_img

def images_to_video(image_list, output_path, fps=30, is_color=True):
    # Ensure there are images in the list
    if not image_list:
        raise ValueError("The image list is empty.")

    # Get the size of the images
    height, width = image_list[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs, like 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

    # Write each image to the video
    for image in image_list:
        if is_color and len(image.shape) == 2:
            raise ValueError("Expected RGB images but received a grayscale image.")
        if not is_color and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    # if args.cam_width:
    #     sensor_configs["width"] = args.cam_width
    # if args.cam_height:
    #     sensor_configs["height"] = args.cam_height
    obs_mode_config = {"coord_bounds": [-7, -5, -0.5, 1, 5, 3], # TODO: add a list of coord bounds by parsing them
                    "voxel_size": args.voxel_size, 
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "segmentation": args.segmentation}
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="voxel",
        reward_mode="none",
        obs_mode_config=obs_mode_config,
        sensor_configs=sensor_configs,
    )

    # Step through the environment (with 100 steps) with random actions
    zf = args.zoom_factor # controlling camera zoom-ins
    obs, _ = env.reset()
    imgs = []
    for i in tqdm(range(100)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action))
        voxel_grid = obs["voxel_grid"]
        img = render_filtered_voxels(voxel_grid, zf)
        imgs.append(img)
        # env.render_human() # will render with a window if possible
    env.close()

    # Write a video showing voxel changes
    vid_path = args.video_path
    images_to_video(imgs, vid_path)


if __name__ == "__main__":
    main(parse_args())
