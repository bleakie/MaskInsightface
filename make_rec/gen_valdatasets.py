# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:02:01 2018

@author: wxy
"""
import os
import numpy as np
import cv2
import pickle,itertools
import random
import argparse

root_path = '/home/sai/YANG/image/Face_Recognition/Test_datasets/add_blake/'
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--total-num', default=80000, type=int, help='numbers of paris')
parser.add_argument('--same-path', default=root_path+'black_crop', type=str, help='img root')
parser.add_argument('--out-name', default=root_path+'black_crop.bin', type=str, help='putput name')
args = parser.parse_args()



def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])
    
def glob_format(path,fmt_list = ('.jpg','.png','.bmp')):
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            fmt = os.path.splitext(item)[-1]
            if fmt not in fmt_list: continue
            fs.append(item)
    return fs
    

def procese_pair(pair):
    a1 =[] #np.zeros((2,112,112,3))
    #a1_flip=np.zeros((2,112,112,3))
    for i in range(2):
        img = cv2.imread(pair[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #aligned = np.transpose(img, (2,0,1))
        # imgg = cv2.imencode('.png', img) 
        data_encode = np.array( cv2.imencode('.jpg', img)[1]).tostring()
        a1[i] = data_encode
        #img_flip=do_flip(img.copy())
        #data_encode = np.array( cv2.imencode('.jpg', img_flip)[1]).tostring()
        #a1_flip[i]= data_encode
    return a1
#    return a1, a1_flip
#boshi original
def gen_pairlist(base_path):
    img_list = glob_format(base_path)
    random.shuffle(img_list)
    pair_list_True=[]
    pair_list_False =[]

    gentor = itertools.combinations( img_list, 2)
    for pair in gentor:
        if len(pair_list_True) >=total_num and len(pair_list_False) >=total_num :
            break
        if pair[0].split("/")[-2] == pair[1].split("/")[-2]:
            pair_list_True.append(pair)
        else:
            pair_list_False.append(pair)

    num = min(len(pair_list_False),len(pair_list_True))

    print('num1:', num)
    pair_list_True = random.sample(pair_list_True, num)
    pair_list_False = random.sample(pair_list_False, num) #1:3 true:false
    pair_list = pair_list_True + pair_list_False
    label_list =[True]*len(pair_list_True) +[False]*len(pair_list_False)
    return pair_list, label_list

def main(base_path, out_name):
    pair_list, label_list = gen_pairlist(base_path)
    img1_list =[] #np.zeros((num*2,112,112,3),dtype=np.uint8)
    #img2_list =[] #np.zeros((num*2,112,112,3),dtype=np.uint8)  
    index = list(range(len(pair_list)))
    random.shuffle(index)
    label_out=[]
    for i in index:
        pair =pair_list[i]
        img1_list.append(open(pair[0],"rb").read())
        img1_list.append(open(pair[1],"rb").read())
        label_out.append(label_list[i])
        
        #a1, a1_flip = procese_pair(pair)
        
        #img1_list+=a1
        #img2_list+=a1_flip
        #for k in range(2):
        #    img1_list.append(a1[k])
        #    img2_list.append(a1_flip[k])
            #img1_list[i*2+k] =a1[k]
            #img2_list[i*2+k] =a1_flip[k]
    out = (img1_list,label_out)

    with open(out_name,'wb') as f:
        pickle.dump(out,f)
    f.close()
    print ('done')

    # import csv
    # with open('../images/test/pairs_std.csv', 'w') as csvFile:
    #     csvwriter = csv.writer(csvFile)
    #     for cs in pair_list:
    #         csvwriter.writerow(cs)
    
if __name__ == "__main__":          
	total_num =args.total_num
	base_path = args.same_path
	out_name = args.out_name
	main(base_path, out_name)
    
    
    
    
    
    
    
