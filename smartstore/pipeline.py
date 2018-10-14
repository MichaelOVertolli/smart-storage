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

from glob import glob
import json
import numpy as np
import os
from PIL import Image


def get_json():
    return glob('./smartstore/json/*')


def processJSON (json_files, currentpath, plot):
    '''process each json file

    data:           the data contained in the json file
    currentpath:    the root path of the database
    plot:           number of different views plotted per frame
    '''

    for jn, jfile in enumerate(json_files):
        with open(jfile) as jdata:
            data = json.load(jdata)


        # parse json data into variables
        loc = data['name']
        date = data['date']
        frames = data['frames']
        objects = data['objects']
        
        # store indices of the empty frames & objects
        empty_frames = []
        empty_objects = []

        for i, f in enumerate(frames):
            if not f or not f['polygon']:
                empty_frames.append(i)

        for i,o in enumerate(objects):
            if not o:
                empty_objects.append(i)

        # get camera intrinsics
        K = np.transpose(np.reshape(readvaluesfromtxt(os.path.join \
                         (currentpath,'data',name,'intrinsics.txt')),(3,3)))

        # get camera extrinsics
        ex_file = os.path.listdir(os.path.join('data',name,'extrinsics'))[-1] 
        extrinsicsC2W = np.transpose(np.reshape(readvaluesfromtext(os.path.join \
          (currentpath,'data',name,'extrinsics',ex_file)), \
          (-1,3,4)),(1,2,0))

        for i, f in enumerate(frames):
            if i in empty_frames:
                continue
            image_path = os.path.join(currentpath,"data",name,"image")
            depth_path = os.path.join(currentpath,"data",name,"depth")
            image_list = os.path.listdir(image_path)
            depth_list = os.path.listdir(depth_path)
            for img in image_list:
                file_num = "0"*(7-len(str(1+i*5)))+str(1+i*5)+"-"
                if file_num in img:
                    image = os.path.join(image_path,img)
                    break
            for img in depth_list:
                file_num = "0"*(7-len(str(1+i*5)))+str(1+i*5)+"-"
                if file_num in img:
                    depth = os.path.join(depth_path,img)
                    break

            background = Image.open(image,'r').convert('RGBA') 

            (width,height) = background.size

            # ---------------------------------------------------- #
            # create frame and fill with data
            current_frame = Frame(i,width,height)
            current_frame.loc = name
            current_frame.background = background
            current_frame.depthMap = depthread(depth)
            current_frame.intrinsics = K
            current_frame.extrinsics = getextrinsics(extrinsicsC2W,i)
            exceptions   = []
            conflicts    = []
            polygons     = {}
            for polygon in f['polygon']:
                ID = polygon['object']
                exists = False
                for o in allObjects:
                    if objects[ID]['name'] == o.name:
                        current_object = o
                        current_object.updateID(ID)
                        exists = True
                        break
                if not exists: # new object
                    current_object = Object1(ID,objects[ID]['name'])
                    allObjects.append(current_object)
                polygons[str(current_object.getName())] = []
                current_object.add_frame(name,current_frame)
                for j, x in enumerate(polygon['x']):
                    x = int(round(x))
                    y = int(round(polygon['y'][j]))
                    polygons[str(current_object.getName())].append((x,y))
                    if 0 < x <= width and 0 < y <= height:
                        if (y,x) in zip(current_frame.row,current_frame.col):
                            conflicts.append(str([(x,y),current_object.getName()]))
                        else:
                            current_frame.add_data(current_object,x,y)
                    else:
                        exceptions.append(str([(x,y),current_object.getName()]))
                current_frame.addObject(current_object)

            file_path = os.path.join('data',name)
            try:
                os.path.makedirs(file_path)
            except OSError:
                if not os.path.isdir(file_path):
                    raise
            
            try:
                os.path.makedirs(os.path.join(file_path,str(i)))
            except OSError:
                if not os.path.isdir(os.path.join(file_path,str(i))):
                    raise

            passed  = open(os.path.join(file_path,str(i),'passed.txt'),'w')
            dropped = open(os.path.join(file_path,str(i),'dropped.txt'),'w')

###############################################################################
### IMPORT TOOLS
###############################################################################

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

###############################################################################
### MAIN RUN PROCEDURE
###############################################################################

if __name__ == "__main__":

    # something like this:
    
    # iterate over all JSON files

    frames = # get frame IDs for current json file

    for i,frame in enumerate(frames):
        # insert stuff from processJSON() above that reads loc from json file
        xyd = ('frame',loc,i) # create tuple to pass into depth2world below
        xyz_world = depth2world(xyd)
        processJSON() # TODO: a new version of this function that only processes the labels
