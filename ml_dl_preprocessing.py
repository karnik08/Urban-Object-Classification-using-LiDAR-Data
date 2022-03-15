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

def plyToData(ply_file):
    
    data = ply_file.elements[0].data

    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        data_np[:,i] = data_pd[name]

    return data_np, data_pd

idd=0
def createPlyFile(df):
    global idd
    data=df.copy()
#     check=0
    normalized_data = pd.DataFrame(preprocessing.minmax_scale(data[['x', 'y', 'z']]))
    normalized_data.columns = ['x', 'y', 'z']
    normalized_data = normalized_data.reset_index(drop=True)
    data = data[['reflectance', 'label', 'class']]
    data = data.reset_index(drop=True)
    data = pd.concat([normalized_data, data], axis=1)
    
    features = pd.DataFrame()
    for i in df.label.unique():
#         check+=1
#         if check==10:
# #             return
#             break
        df_ = df[df.label == i]
        
        dfml = data[data.label == i]

        if df_['class'].iloc[0] == 0 or df_['class'].iloc[0] == 100000000:
            continue
        else:
#             print(i)
#             print('dl',df_.shape)
#             print('ml',dfml.shape)
            file_name = "../data/test_ply_objects/" +str(idd)+ "-"+str(int(df_.iloc[0]['class'])) + "_" + str(int(df_.iloc[0].label)) + ".ply"
            data_np = df_[['x', 'y', 'z']].to_numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data_np)
#             print(pcd)
            #write to the file
#             o3d.io.write_point_cloud(file_name, pcd)
            
            
            ##ml
            dfml = dfml[['x', 'y', 'z']]
            dfml = dfml.to_numpy()

            covariance_matrix = np.cov(dfml.T)
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

            linearity = (eigen_values[0] - eigen_values[1]) / eigen_values[0]
            planarity = (eigen_values[1] - eigen_values[2]) / eigen_values[1]
            sphericity = eigen_values[2] / eigen_values[0]
            omnivariance = (eigen_values[0] * eigen_values[1] * eigen_values[2]) ** 1/3
            anisotropy = (eigen_values[0] - eigen_values[2]) / eigen_values[0]
            eigenentropy = (eigen_values[0] * np.log(eigen_values[0]) + eigen_values[1] * np.log(eigen_values[1]) + eigen_values[2] * np.log(eigen_values[2])) * -1
            sum_of_eigenvalues = eigen_values[0] + eigen_values[1] + eigen_values[2]
            change_curvature = eigen_values[2] / (eigen_values[0] + eigen_values[1] + eigen_values[2])
            temp=pd.DataFrame([[i,linearity,planarity,sphericity,omnivariance,anisotropy,eigenentropy,
                            sum_of_eigenvalues,change_curvature,idd]],
                             columns=['label', 'linearity', 'planarity', 'sphericity',
                                  'omnivariance', 'anisotropy', 'eigenentropy', 
                                  'sum_of_eigenvalues', 'change_curvature', 'id'])
            features=pd.concat([features,temp])
    
#             print(features)
        
            idd+=1
#     print(features)
    return features
def labelsToClasses(df, featureDF):
    
    label_to_class = df[['label', 'class']].drop_duplicates()
    featuresDF = featureDF.merge(label_to_class, on=['label'])
    
    return featuresDF

def addMedianReflectance(df, featureDF):
    
    reflectance = pd.DataFrame()
    
    for class_name in df['class'].unique():
        
        _df = df[df['class'] == class_name]
        value = _df['reflectance'].median()
        
        temp=pd.DataFrame([[class_name,value]],columns=['class', 'reflectance'])
#         reflectance = reflectance.append(pd.Series({'class' : class_name, 'reflectance' : value}), ignore_index=True)
        reflectance=pd.concat([reflectance,temp])
    
#     featureDF = featureDF.merge(reflectance, on=['class'])
    featureDF=pd.merge(featureDF, reflectance, on="class")
    featureDF = featureDF.drop(['label'], axis=1)

    return featureDF

object_classes = {
    '0' : 'Unclassified',
    '100000000' : 'Other',
    '200000000' : 'Surface',
    '201000000' : 'Other Surface',
    '202000000' : 'Ground',
    '202010000' : 'Other Ground',
    '202020000' : 'Road',
    '202030000' : 'Sidewalk',
    '202040000' : 'Curb',
    '202050000' : 'Island',
    '202060000' : 'Vegetation',
    '203000000' : 'Building',
    '300000000' : 'Object',
    '301000000' : 'Other Object',
    '302000000' : 'Static',
    '302010000' : 'Other Static',
    '302020000' : 'Punctual Object',
    '302020100' : 'Other Punctual Object',
    '302020200' : 'Post',
    '302020300' : 'Bollard',
    '302020400' : 'Floor Lamp',
    '302020500' : 'Traffic Light',
    '302020600' : 'Traffic Sign',
    '302020700' : 'Signboard',
    '302020800' : 'Mailbox',
    '302020900' : 'Trash Can',
    '302021000' : 'Meter',
    '302021100' : 'Bicycle Terminal',
    '302021200' : 'Bicycle Rack',
    '302021300' : 'Statue',
    '302030000' : 'Linear',
    '302030100' : 'Other Linear',
    '302030200' : 'Barrier',
    '302030300' : 'Roasting',
    '302030400' : 'Grid',
    '302030500' : 'Chain',
    '302030600' : 'Wire',
    '302030700' : 'Low Wall',
    '302040000' : 'Extended',
    '302040100' : 'Other Extended',
    '302040200' : 'Shelter',
    '302040300' : 'Kiosk',
    '302040400' : 'Scaffold',
    '302040500' : 'Bench',
    '302040600' : 'Distribution Box',
    '302040700' : 'Lighting Console',
    '302040800' : 'Windmill',
    '303000000' : 'Dynamic',
    '303010000' : 'Other Dynamic',
    '303020000' : 'Pedestrian',
    '303020100' : 'Other Pedestrian',
    '303020200' : 'Still Pedestrian',
    '303020300' : 'Walking Pedestrian',
    '303020400' : 'Running Pedestrian',
    '303020500' : 'Stroller Pedestrian',
    '303020600' : 'Holding Pedesterian',
    '303020700' : 'Leaning Pedestrian',
    '303020800' : 'Skater',
    '303020900' : 'Rollerskater',
    '303021000' : 'Wheelchair',
    '303030000' : '2 Wheelers',
    '303030100' : 'Other 2 Wheels',
    '303030200' : 'Bicycle',
    '303030201' : 'Other Bicycle',
    '303030202' : 'Mobile Bicycle',
    '303030203' : 'Stopped Bicycle',
    '303030204' : 'Parked Bicycle',
    '303030300' : 'Scooter',
    '303030301' : 'Other Scooter',
    '303030302' : 'Mobile Scooter',
    '303030303' : 'Stopped Scooter',
    '303030304' : 'Parked Scooter',
    '303030400' : 'Moped',
    '303030401' : 'Other Moped',
    '303030402' : 'Mobile Moped',
    '303030403' : 'Stopped Moped',
    '303030404' : 'Parked Moped',
    '303030500' : 'Motorbike',
    '303030501' : 'Other Motorbike',
    '303030502' : 'Mobile Motorbike',
    '303030503' : 'Stopped Motorbike',
    '303030504' : 'Parked Motorbike',
    '303040000' : '4+ Wheelers',
    '303040100' : 'Other 4+ Wheelers',
    '303040200' : 'Car',
    '303040201' : 'Other Car',
    '303040202' : 'Mobile Car',
    '303040203' : 'Stopped Car',
    '303040204' : 'Parked Car',
    '303040300' : 'Van',
    '303040301' : 'Other Van',
    '303040302' : 'Mobile Van',
    '303040303' : 'Stopped Van',
    '303040304' : 'Parked Van',
    '303040400' : 'Truck',
    '303040401' : 'Other Truck',
    '303040402' : 'Mobile Truck',
    '303040403' : 'Stopped Truck',
    '303040404' : 'Parked Truck',
    '303040500' : 'Bus',
    '303040501' : 'Other Bus',
    '303040502' : 'Mobile Bus',
    '303040503' : 'Stopped Bus',
    '303040504' : 'Parked Bus',
    '303050000' : 'Furniture',
    '303050100' : 'Other Furniture',
    '303050200' : 'Table',
    '303050300' : 'Chair',
    '303050400' : 'Stool',
    '303050500' : 'Trash Can',
    '303050600' : 'Waste',
    '304000000' : 'Natural',
    '304010000' : 'Other Natural',
    '304020000' : 'Tree',
    '304030000' : 'Bush',
    '304040000' : 'Potted Plant',
    '304050000' : 'Hedge'
}

ply0 = PlyData.read('../data/Paris.ply')
ply1 = PlyData.read('../data/Lille1.ply')
ply2 = PlyData.read('../data/Lille2.ply')


data_np0, data_pd0 = plyToData(ply0)
data_np1, data_pd1 = plyToData(ply1)
data_np2, data_pd2 = plyToData(ply2)

features0 = createPlyFile(data_pd0)
features1 = createPlyFile(data_pd1)
features2 = createPlyFile(data_pd2)

features0 = labelsToClasses(data_pd0, features0)
features1 = labelsToClasses(data_pd1, features1)
features2 = labelsToClasses(data_pd2, features2)

features0 = addMedianReflectance(data_pd0, features0)
features1 = addMedianReflectance(data_pd1, features1)
features2 = addMedianReflectance(data_pd2, features2)

features = pd.concat([features0, features1, features2])

features['class'] = features['class'].astype(str)
features['class'] = features['class'].map(object_classes)

features = features[features['class'] != 'Unclassified']
features = features[features['class'] != 'Other']

print(features.shape)
# features.to_csv("../data/test_geometric_features_normalized.csv", index=False)