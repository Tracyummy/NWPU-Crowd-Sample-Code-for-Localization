# all function summerize into this script

import numpy as np
import math
import sys
import cv2
from scipy import spatial as ss


def read_pred_and_gt(pred_file = './eval/tiny_val_loc_0.8_0.3.txt', gt_file = './eval/val_gt_loc.txt'):
    """ 
        NWPU pred format:  (file_id, pred_num,  [x,y [,x,y] ... ] , )
        NWPU  gt  format:  (file_id,  gtnum,  [x, y, sigma_s, sigma_l, level]  )
    """

    pred_data = {}
    with open(pred_file) as f:
        for line in f.readlines():
            
            line.strip().split(' ')

            # check each line must be size-2 tuple: pred_num should be corresponding to num of xy
            if len(line) <2 or len(line) % 2 !=0 or (len(line)-2)/2 != int(line[1]):
                flagError = True
                sys.exit(1)

            line_data = [int(i) for i in line]

            if line_data[1]==0:
                points = []
            else:
                assert line_data[1] > 0
                points = np.array(line_data[2:]).reshape(line_data[1],2)

            pred_data[line_data[0]] = {'num' : line_data[1] , 'points':  points }

    gt_data = {}
    with open(gt_file) as f:
        for line in f.readlines():
            
            line.strip().split(' ')

            line_data = [int(i) for i in line]
            # (file_id,  gtnum,  [(x, y, sigma_s,  sigma_l,  level), ...])

            if line_data[1]==0:
                gt_data[line_data[0]] = {'num': 0, 'points':[], 'sigma':[], 'level':[]}
            else:
                assert line_data[1] > 0
                points = np.array(line_data[2:]).reshape((len(line_data - 2)//5,5))
                gt_data[line_data[0]] = {'num' : line_data[1] , 'points':  points }
    
    return pred_data, gt_data




def evaluate(pred_file = './eval/tiny_val_loc_0.8_0.3.txt', gt_file = './eval/val_gt_loc.txt'):
    
    pred_data, gt_data = read_pred_and_gt(pred_file, gt_file)

    id_std = [i for i in range(3110,3610,1)]
    for i_sample in id_std:
        if gt_data[i_sample]['num'] !=0 and pred_data[i_sample]['num'] !=0:
            pass
            

def compute_metrics(dist_matrix,  match_matrix,  pred_p.shape[0],  gt_p.shape[0],  sigma_l,  level):
    pass


if __name__ == '__main__':
    # main()
    # read_pred_and_gt()
    a = np.array([1,1,2,3,4,4,4,5])
    print(np.where(a[1:] != a[:-1])[0] + 1)
