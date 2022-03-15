import os
from os import listdir
import glob

import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from plyfile import PlyData, PlyElement
import open3d as o3d
from pyntcloud import PyntCloud

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing




def voxelize(path):
    for ply_file in glob.iglob(path):

        ply = PyntCloud.from_file(ply_file)
        voxelgrid_id = ply.add_structure('voxelgrid', n_x=32, n_y=32, n_z=32)
        voxelgrid = ply.structures[voxelgrid_id]
        binary_feature_vector = voxelgrid.get_feature_vector(mode='binary')

        file_name = '../data/test_npy_objects_method2/' + ply_file.split('/')[-1].split('.')[0] + '.npy'
        #to save to a file
#         np.save(file_name, binary_feature_vector)

        
path='../data/test_ply_objects/*.ply'
voxelize(path)