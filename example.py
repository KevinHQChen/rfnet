# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:20
# @Author  : xylon
import cv2
import torch
import random
import argparse
import numpy as np

import glob
import pickle

from utils.common_utils import gct
from utils.eval_utils import nearest_neighbor_distance_ratio_match
from model.rf_des import HardNetNeiMask
from model.rf_det_so import RFDetSO
from model.rf_net_so import RFNetSO
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--imgpath", default=None, type=str)  # image path
    parser.add_argument("--resume", default=None, type=str)  # model path
    args = parser.parse_args()

    print(f"{gct()} : start time")

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    print(f"{gct()} : model init")
    det = RFDetSO(
        cfg.TRAIN.score_com_strength,
        cfg.TRAIN.scale_com_strength,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        cfg.TRAIN.TOPK,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
        cfg.MODEL.KSIZE,
        cfg.MODEL.padding,
        cfg.MODEL.dilation,
        cfg.MODEL.scale_list,
    )
    des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
    model = RFNetSO(
        det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
    )

    print(f"{gct()} : to device")
    device = torch.device("cuda")
    model = model.to(device)
    resume = args.resume
    print(f"{gct()} : in {resume}")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])

    i = 0
    filelist = glob.iglob(r'../00/image_0/*.png')
    for infile in sorted(filelist):
        i = i + 1
        print(infile)
        kp, des, img = model.detectAndCompute(infile, device, (376, 1241))
        with open('dest/'+str(i-1)+'.txt', 'wb') as f:
            pickle.dump((kp, des), f)
