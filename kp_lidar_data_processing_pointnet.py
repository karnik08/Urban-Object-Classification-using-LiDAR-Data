#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # **DIRECTORY BASED**

# In[2]:



# def createPoissonTriangleMesh(directory):
#     count=0
#     for file in glob.glob(directory):
#         count+=1
#         pcd = o3d.io.read_point_cloud(file)
    
#         if len(np.asarray(pcd.points)) < 250:
#             continue
    
#         pcd.estimate_normals()

#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
#         filename = '../../data/paris_lille/mesh_objects_poisson/' + 'mesh_' + file.split('/')[-1]
# #         o3d.io.write_triangle_mesh(filename, mesh)

#         sampled = mesh.sample_points_poisson_disk(number_of_points=500)
#         filename = '../../data/paris_lille/sampled_poisson_poisson_ply/' + 'sampled_' + file.split('/')[-1]
# #         o3d.io.write_point_cloud(filename, sampled)
#     print(count)


# In[10]:


# count=0
def createBPTriangleMesh(directory):
    count=0
    for file in glob.glob(directory):
        if count==10:
            return
        count+=1
        print(file)
        pcd = o3d.io.read_point_cloud(file)

        if len(np.asarray(pcd.points)) < 250:
            continue

        pcd.estimate_normals()

        distances = pcd.compute_nearest_neighbor_distance()
        avg_distance = np.mean(distances)
        radius = 1.5 * avg_distance

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius*2]))
        filename = 'data/mesh_objects/' + 'mesh_' + file.split('/')[-1]
#         o3d.io.write_triangle_mesh(filename, mesh)

        sampled = mesh.sample_points_poisson_disk(number_of_points=1000)
        filename = 'data/sampled_ply/' + 'sampled_' + file.split('/')[-1]
#         o3d.io.write_point_cloud(filename, sampled)
#     print(count)


# In[14]:


# createPoissonTriangleMesh('../../data/paris_lille/ply_objects/*.ply')
# createPoissonTriangleMesh('paris_lille/ply_objects/*.ply')


# In[11]:


# createBPTriangleMesh('../../data/paris_lille/ply_objects/*.ply')
createBPTriangleMesh('data/ply_objects/*.ply')


# # **PARALLELIZED**

# In[ ]:


# def createBPTriangleMesh(file):
    
#     pcd = o3d.io.read_point_cloud(file)

#     if len(np.asarray(pcd.points)) > 250:
#         pcd.estimate_normals()

#         distances = pcd.compute_nearest_neighbor_distance()
#         avg_distance = np.mean(distances)
#         radius = 1.5 * avg_distance

#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius*2]))
#         filename = '/home/jupyter-seanandrewchen/shared/cusp-capstone/data/paris_lille/mesh_objects/' + 'mesh_' + file.split('/')[-1]
# #         o3d.io.write_triangle_mesh(filename, mesh)

#         sampled = mesh.sample_points_poisson_disk(number_of_points=1000)
#         filename = '/home/jupyter-seanandrewchen/shared/cusp-capstone/data/paris_lille/sampled_ply/' + 'sampled_' + file.split('/')[-1]
# #         o3d.io.write_point_cloud(filename, sampled)
#     else:
#         pass


# # In[ ]:


# ply_dir = '/home/jupyter-seanandrewchen/shared/cusp-capstone/data/paris_lille/ply_objects/'
# input_path = os.path.join(ply_dir, '*.ply')

# num_cores = mp.cpu_count()
# pool = mp.Pool(num_cores)

# start = time.time()
# pool.map(createBPTriangleMesh, glob.glob(input_path))
# end = time.time()
# print(end - start)

# pool.close()

