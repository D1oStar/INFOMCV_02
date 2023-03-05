import glm
import random
import numpy as np
from CameraConfig import CameraConfig

block_size = 1
cc = CameraConfig()


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = cc.voxel_pos_mp(block_size=block_size)
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cc.load_xml()
    return [cc.camera_position('cam1'), cc.camera_position('cam2'),
            cc.camera_position('cam3'), cc.camera_position('cam4')]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cc.load_xml()
    return [cc.rot('cam1'), cc.rot('cam2'), cc.rot('cam3'), cc.rot('cam4')]
