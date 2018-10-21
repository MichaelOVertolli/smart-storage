###############################################################################
#Copyright (C) 2018 Science of Imagination Laboratory, Carleton University
#
#Written by Mike Cichonski and Michael O. Vertolli
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

import abc
from collections import namedtuple
from glob import glob
import json
import numpy as np
import os
from PIL import Image

###############################################################################
### PIPELINE FUNCTIONS
###############################################################################

def processjson(lbl_record,
                json_folder='./smartstore/json/'):
    '''process each json file

    data:           the data contained in the json file
    currentpath:    the root path of the database
    '''
    lblsets = {}
    json_files = get_json(json_folder)
    for jfile in json_files:
        with open(jfile) as jdata:
            data = json.load(jdata)

        file_name = data['name']
        lbls = data['objects']

        for i, f in enumerate(data['frames']):
            if not f or not f['polygon']:
                continue
            frame = Frame._make([file_name, i])
            try:
                flblset = lblsets[frame]
            except KeyError:
                flblset = {}
                lblsets[frame] = flblset
            for polygon in f['polygon']:
                old_id = polygon['object']
                lbl = lbls[old_id]['name']
                new_id = lbl_record.get_or_add(lbl)
                point_cluster = []
                for j, x in enumerate(polygon['x']):
                    # will handle rounding later... want more raw data
                    y = polygon['y'][j]
                    point_cluster.append((x,y))
                point_cluster.sort()
                lblset = LabelSet._make([frame, lbl, tuple(point_cluster)])
                flblset[lblset] = None
    return lblsets

###############################################################################
### DATA CLASSES
###############################################################################

Frame = namedtuple('Frame', 'name, id')
LabelSet = namedtuple('LabelSet', 'frame, label, points')
DepthSet = namedtuple('DepthSet', 'frame, depthmap')

###############################################################################
### REPOSITORY CLASSES
###############################################################################

class IDRecord(abc.ABC):
    class DNEException(Exception):
        pass
    
    def __init__(self, fname):
        self.fname = fname
        self.record = self.open_file(fname)

    @abc.abstractmethod
    def open_file(self, fname):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @abc.abstractmethod
    def get_count(self):
        pass

    @abc.abstractmethod
    def add(self, value):
        pass

    @abc.abstractmethod
    def get_id(self, value):
        pass

    def get_or_add(self, value):
        try:
            id_ = self.get_id(value)
        except self.DNEException:
            self.add(value)
            id_ = self.get_id(value)
        return id_


class PDictIDRecord(IDRecord):
    COUNT = '__count'
    def open_file(self, fname):
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                record = pickle.load(f)
        else:
            record = {self.COUNT:0}
        return record

    def update(self):
        with open(fname, 'wb') as f:
            pickle.dump(self.record, f)

    def get_count(self):
        return self.record[self.COUNT]

    def add(self, value):
        self.record[value] = self.get_count()
        self.record[self.COUNT] += 1

    def get_id(self, value):
        try:
            id_ = self.record[value]
        except KeyError:
            raise super().DNEException
        return id_

###############################################################################
### IMPORT TOOLS
###############################################################################

def get_json(json_folder):
    return glob(os.path.join(json_folder, '*'))


def get_intrinsics(loc):
    with open(os.path.join(os.getcwd(),'data',loc,'intrinsics.txt'),'r') as fid:
        values = [map(lambda y: float(y), x.split(' ')) for x in
                  ''.join(fid.read().decode('utf-8')).split('\n') if x!='']
    intrinsics = np.transpose(np.reshape(values),(3,3))
    return intrinsics 


def get_extrinsics(loc,frame_id):
    with open(os.path.listdir(os.path.join('data',loc,'extrinsics'))[-1],'r') as fid:
        values = [map(lambda y: float(y), x.split(' ')) for x in
                  ''.join(fid.read().decode('utf-8')).split('\n') if x!='']
    # get extrinsics for all frames:
    ext = np.transpose(np.reshape(values,(-1,3,4)),(1,2,0))
    # get extrinsics for current frame_id:
    extrinsics = [[ext[0][0][frame_id],ext[0][1][frame_id], \
                   ext[0][2][frame_id],ext[0][3][frame_id]],\
                  [ext[1][0][frame_id],ext[1][1][frame_id], \
                   ext[1][2][frame_id],ext[1][3][frame_id]],\
                  [ext[2][0][frame_id],ext[2][1][frame_id], \
                   ext[2][2][frame_id],ext[2][3][frame_id]]]
    return extrinsics


# frame_id must be an int
def get_depth(loc,frame_id):
    assert(frame_id is int)
    depth_dir  = os.path.join(os.getcwd(),'data',loc,'depth')
    depth_list = os.path.listdir(depth_path)
    for img in depth_list:
        file_num = "0"*(7-len(str(1+frame_id*5)))+str(1+frame_id*5)+"-"
        if file_num in img:
            depth_path = os.path.join(depth_dir,img)
            break
    with Image.open(depth_path,'r') as depth:
        depth_map = np.array(depth)
    bit_shift = [[(d >> 3) or (d << 16-3) for d in row] for row in depth_map]
    depth_map = [[float(d)/1000.0 for d in row] for row in bit_shift]
    return depth_map


def depth2camera(depth_map,K):
    [x,y] = np.meshgrid(range(1,641),range(1,481))
    xyzcamera = []
    l = np.multiply([[d - K[0][2] for d in row] for row in x], \
                    [[d / K[0][0] for d in row] for row in depth_map])
    xyzcamera.append(l)
    l = np.multiply([[d - K[1][2] for d in row] for row in y], \
                    [[d / K[1][1] for d in row] for row in depth_map])
    xyzcamera.append(l)
    xyzcamera.append(depth_map)
    depth_map = [[int(bool(n)) for n in row] for row in depth_map]
    xyzcamera.append(depth_map)
    return xyzcamera


def camera2world(xyz_camera,extrinsics):
    xyz = []
    for j in range(0,len(xyz_camera[0][0])):
        for i in range(0,len(xyz_camera[0])):
            if xyz_camera[3][i][j]:
                data = (xyz_camera[0][i][j],\
                        xyz_camera[1][i][j],\
                        xyz_camera[2][i][j])
                xyz.append(data)

    rt = [(float(t[0]),float(t[1]),float(t[2])) for t in extrinsics]
    xyz_world = np.matmul(rt,np.transpose(xyz))
    # xyz_camera[3] are the valid world coords corresponding to each [x,y]
    xyz_world = (xyz_world, xyz_camera[3])
    return xyz_world


# xyd is a (type='frame', loc, frame_id) tuple
def depth2world(xyd):
    assert(xyd[0]=='frame')
    intrinsics = get_intrinsics(xyd[1])        # get intrinsics for current location
    extrinsics = get_extrinsics(xyd[1],xyd[2]) # get extrinsics for current frame
    depth_map  = get_depth(xyd[1],xyd[2])      # get depth for current frame
    
    xyz_camera = depth2camera(depth_map,intrinsics)  # calculate camera coords
    xyz_world  = camera2world(xyz_camera,extrinsics) # calculate world coords

    # return a (type='world_matrix',loc,frame_id, xyz_world) tuple
    return ('world_matrix',xyd[1],xyd[2],xyz_world)

