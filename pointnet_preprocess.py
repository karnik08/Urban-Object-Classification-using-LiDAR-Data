import os
import glob

import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from plyfile import PlyData, PlyElement
import open3d as o3d
import trimesh

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def createBPTriangleMesh(directory):
    
    for file in glob.glob(directory):
    
        pcd = o3d.io.read_point_cloud(file)

        if len(np.asarray(pcd.points)) < 250:
            continue

        pcd.estimate_normals()

        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radius = 1.5 * avg_distance

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius*2]))
        filename = '../data/test_mesh_objects/' + 'mesh_' + file.split('/')[-1]
        #to save to a file
#         o3d.io.write_triangle_mesh(filename, mesh)

        sampled = mesh.sample_points_poisson_disk(number_of_points=1000)
        filename = '../data/test_sampled_ply/' + 'sampled_' + file.split('/')[-1]
        #to save to a file
#         o3d.io.write_point_cloud(filename, sampled)

path='../data/test_ply_objects/*.ply'     
createBPTriangleMesh(path)

