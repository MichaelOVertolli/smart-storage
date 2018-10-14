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


def processJSON (json_files, currentpath, local, plot):
    '''process each json file

    data:   the data contained in the json file
    currentpath:    the root path of the database
    local:  True if database is located locally, False if on web
    plot:   number of different views plotted per frame
    '''

    for jn, jfile in enumerate(json_files):
        with open(jfile) as jdata:
            data = json.load(jdata)


        # parse json data into variables
        name = data['name']
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
                         (currentpath,'data',name,'intrinsics.txt'),local),(3,3)))

        # get camera extrinsics
        ex_file = os.path.listdir(os.path.join('data',name,'extrinsics'))[-1] 
        extrinsicsC2W = np.transpose(np.reshape(readvaluesfromtext(os.path.join \
          (currentpath,'data',name,'extrinsics',ex_file),local), \
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
            current_frame.depthMap = depthread(depth,local)
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

def readvaluesfromtxt(filename,local):
    fid = open(filename,'r')
    values = [map(lambda y: float(y), x.split(' ')) for x in
              ''.join(fid.read().decode('utf-8')).split('\n') if x!='']
    fid.close()
    return values

def getextrinsics(extrinsicsC2W,frame):
    extrinsics = [[extrinsicsC2W[0][0][frame],extrinsicsC2W[0][1][frame], \
                   extrinsicsC2W[0][2][frame],extrinsicsC2W[0][3][frame]],\
                  [extrinsicsC2W[1][0][frame],extrinsicsC2W[1][1][frame], \
                   extrinsicsC2W[1][2][frame],extrinsicsC2W[1][3][frame]],\
                  [extrinsicsC2W[2][0][frame],extrinsicsC2W[2][1][frame], \
                   extrinsicsC2W[2][2][frame],extrinsicsC2W[2][3][frame]]]
    return extrinsics

def depthread(filename,local):
    depth = Image.open(filename,'r')
    depthmap = np.array(depth)
    depth.close()
    bitshift = [[(d >> 3) or (d << 16-3) for d in row] for row in depthmap]
    depthmap = [[float(d)/1000.0 for d in row] for row in bitshift]
    return depthmap

def depth2xyzcamera(K,depth):
    [x,y] = np.meshgrid(range(1,641),range(1,481))
    xyzcamera = []
    l = np.multiply([[d - K[0][2] for d in row] for row in x], \
                    [[d / K[0][0] for d in row] for row in depth])
    xyzcamera.append(l)
    l = np.multiply([[d - K[1][2] for d in row] for row in y], \
                    [[d / K[1][1] for d in row] for row in depth])
    xyzcamera.append(l)
    xyzcamera.append(depth)
    depth = [[int(bool(n)) for n in row] for row in depth]
    xyzcamera.append(depth)
    return xyzcamera

def camera2XYZworld(xyzcamera,extrinsics):
    xyz = []
    for j in range(0,len(xyzcamera[0][0])):
        for i in range(0,len(xyzcamera[0])):
            if xyzcamera[3][i][j]:
                data = (xyzcamera[0][i][j],\
                        xyzcamera[1][i][j],\
                        xyzcamera[2][i][j])
                xyz.append(data)
    xyzworld = transformpointcloud(xyz, extrinsics)
    vals = (xyzworld, xyzcamera[3])
    return vals

def transformpointcloud(xyz, rt):
    rt_ = [(float(t[0]),float(t[1]),float(t[2])) for t in rt]
    return np.matmul(rt_, np.transpose(xyz))

###############################################################################
### MAIN RUN PROCEDURE
###############################################################################

if __name__ == "__main__":

    objects = {} # { ID : [name,colour,frame_inds,oldIDs] }
    frames  = {} # 
