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
        filename = '../data/test_mesh_objects/' +file.split('/')[-1].split('-')[0]+ '-mesh_' + file.split('/')[-1].split('-')[-1]
        #to save to a file
        o3d.io.write_triangle_mesh(filename, mesh)

        sampled = mesh.sample_points_poisson_disk(number_of_points=1000)
        filename = '../data/test_sampled_ply/' + file.split('/')[-1].split('-')[0]+ '-sampled_' + file.split('/')[-1].split('-')[-1]
        #to save to a file
        o3d.io.write_point_cloud(filename, sampled)

        
createBPTriangleMesh('../data/test_ply_objects/*.ply')
'''
# def createBPTriangleMesh(file):
    
#     pcd = o3d.io.read_point_cloud(file)

#     if len(np.asarray(pcd.points)) > 250:
# #         print('creating file')
#         pcd.estimate_normals()

#         distances = pcd.compute_nearest_neighbor_distance()
#         avg_distance = np.mean(distances)
#         radius = 1.5 * avg_distance

#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius*2]))
#         filename = '../data/test_mesh_objects/' + 'mesh_' + file.split('/')[-1]
#         #to save to a file
#         o3d.io.write_triangle_mesh(filename, mesh)
        
#         print('working on ',filename)

#         sampled = mesh.sample_points_poisson_disk(number_of_points=1000)
#         filename = '../data/test_sampled_ply/' + 'sampled_' + file.split('/')[-1]
#         #to save to a file
#         o3d.io.write_point_cloud(filename, sampled)
#     else:
#         pass
        
        

# ply_dir='../data/test_ply_objects/'     
# ply_dir = '/home/jupyter-seanandrewchen/shared/cusp-capstone/data/paris_lille/ply_objects/'
# input_path = os.path.join(ply_dir, '*.ply')
p1='../data/test_ply_objects/1000-302020600_700.ply'
p2='../data/test_ply_objects/1007-304020000_505.ply'
p3='../data/test_ply_objects/1010-303040204_555.ply'
p4='../data/test_ply_objects/1047-202050000_451.ply'
# createBPTriangleMesh(p1)
num_cores = mp.cpu_count()
pool = mp.Pool(num_cores)
print(num_cores)
start = time.time()
# # pool.map(createBPTriangleMesh, glob.glob(input_path))
pool.map(createBPTriangleMesh, [p1,p2,p3,p4])
end = time.time()
print(end - start)

# pool.close()
'''
