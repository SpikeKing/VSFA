#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/12


# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

"""
from math import floor

import xlsxwriter

"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
import os
import random

import time
from argparse import ArgumentParser
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import skvideo.io
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms, models

from root_dir import DATASETS_DIR
from utils.project_utils import traverse_dir_files
from utils.vqa_utils import unify_size, init_vid


def get_frame(file_name, idx, h, w):
    cap = cv2.VideoCapture(file_name)  # crashes here
    # print("opened capture {}".format(mp.current_process()))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print('[Info] 帧异常: {}'.format(idx))
        frame = None

    cap.release()
    return idx, frame


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class VideoDatasetWithOpenCV(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, video_names, name_info_dict, n_max=25, is_mp=True):
        super(VideoDatasetWithOpenCV, self).__init__()

        assert len(video_names) == len(name_info_dict.keys())
        self.is_mp = is_mp
        if self.is_mp:
            print('[Info] 提取特征: 多进程')
        else:
            print('[Info] 提取特征: 单进程')

        self.video_names = video_names
        self.name_info_dict = name_info_dict
        self.n_max = n_max

    def __len__(self):
        return len(self.video_names)

    def get_vid_names(self):
        return self.video_names

    @staticmethod
    def get_transformed_video(path, num_max):
        s_time = time.time()

        cap, n_frame, h, w = init_vid(path)
        print('[Info] 帧数: {}, h: {}, w: {}'.format(n_frame, h, w))

        h, w = unify_size(h, w, ms=512)

        # idx_list = []
        # for idx in range(n_frames):
        #     idx_list.append(idx)

        max_gap = num_max  # 最大帧数
        idx_list = []
        if n_frame > max_gap:
            v_gap = float(n_frame) / float(max_gap)  # 只使用100帧
            for gap_idx in range(max_gap):
                idx = int(floor(gap_idx * v_gap))
                idx_list.append(idx)
        else:
            for gap_idx in range(n_frame):
                idx_list.append(gap_idx)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # transformed_video = torch.zeros([n_frames, 3, h, w])

        print('[Info] 特征帧数: {}'.format(len(idx_list)))
        tensor_list = []
        for idx in idx_list:
            # print('[Info] 帧idx: {}'.format(idx))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                frame = cv2.resize(frame, (w, h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = transform(frame)
            except Exception as e:
                print('[Info] 异常帧: {}'.format(idx))
                continue
            if idx % 100 == 0:
                print('[Info] 已处理: {}'.format(idx))

            tensor_list.append(frame_tensor)

        transformed_video = torch.stack(tensor_list, dim=0)  # 合并tensor

        cap.release()
        elapsed_time = time.time() - s_time

        print('[Info] vid_shape: {}, time: {}, 视频: {}'.format(transformed_video.shape, elapsed_time, path))

        return transformed_video

    @staticmethod
    def get_transformed_video_mp(path, num_max):
        s_time = time.time()
        # n_prc = mp.cpu_count()
        n_prc = 10
        cap, n_frame, h, w = init_vid(path)

        h, w = unify_size(h, w, ms=512)
        pool = Pool(n_prc)

        max_gap = num_max  # 最大帧数
        idx_list = []
        if n_frame > max_gap:
            v_gap = float(n_frame) / float(max_gap)  # 只使用100帧
            for gap_idx in range(max_gap):
                idx = int(floor(gap_idx * v_gap))
                idx_list.append(idx)
        else:
            for gap_idx in range(n_frame):
                idx_list.append(gap_idx)

        prc_list = []
        # for idx in range(n_frame):
        for idx in idx_list:
            prc_list.append(pool.apply_async(get_frame, args=(path, idx, h, w)))

        pool.close()
        pool.join()

        res_dict = {}
        for r_prc in prc_list:
            idx, frame = r_prc.get()
            res_dict[idx] = frame

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_list = []
        for i in idx_list:
            try:
                frame = res_dict[i]
                frame = Image.fromarray(frame)
                frame_tensor = transform(frame)
            except Exception:
                print('[Info] 帧异常: {}'.format(i))
                continue

            tensor_list.append(frame_tensor)

        transformed_video = torch.stack(tensor_list, dim=0)  # 合并tensor

        elapsed_time = time.time() - s_time
        print('[Info] vid_shape: {}, time: {}, 视频: {}'.format(transformed_video.shape, elapsed_time, path))

        return transformed_video

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        print('[Info] 解码视频 {} 开始!'.format(video_name))

        score, path = self.name_info_dict[video_name]

        if self.is_mp:
            transformed_video = self.get_transformed_video_mp(path, self.n_max)
        else:
            transformed_video = self.get_transformed_video(path, self.n_max)  # 单进程

        sample = {
            'video': transformed_video,
            'name': video_name,
            'score': score
        }

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    print('[Info] 提取特征开始!')
    s_time = time.time()
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()

    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output = torch.cat((output1, output2), 1).squeeze()

    elapsed_time = time.time() - s_time
    print('[Info] 特征: {}, 耗时: {}'.format(output.shape, elapsed_time))
    return output


def get_vqc_mat_info(vid_dir):
    """
    数据集信息
    """
    print('[Info] 初始化数据集!')
    s_time = time.time()
    # data_path = os.path.join(DATASETS_DIR, 'live_vqc')
    data_info_path = os.path.join(vid_dir, 'data.mat')
    data_info = loadmat(data_info_path)  # index, ref_ids

    video_list_info = data_info['video_list']
    mos_list_info = data_info['mos']

    video_names, scores = [], []
    for vid_info in video_list_info:
        video_names.append(vid_info[0][0])
    for mos in mos_list_info:
        scores.append(mos[0])

    name_score_dict = dict()
    for vid_name, score in zip(video_names, scores):
        name_score_dict[vid_name] = score

    name_info_dict = dict()
    paths_list, names_list = traverse_dir_files(vid_dir, ext='.mp4')

    score_list = []
    for name, path in zip(names_list, paths_list):
        score = name_score_dict[name]
        score_list.append(score)
        name_info_dict[name] = (score, path)

    max_score = np.max(np.asarray(score_list))
    print('[Info] max_score: {}'.format(max_score))
    elapsed_time = time.time() - s_time
    print('[Info] 数据量: {}, 耗时: {}'.format(len(name_info_dict.keys()), elapsed_time))

    return video_names, name_info_dict


def get_processed_vids(feature_dir):
    """
    已处理的视频
    """
    paths_list, names_list = traverse_dir_files(feature_dir, ext='.npy')
    feature_set = set()
    for name in names_list:
        xx = name.split('_')[0]
        feature_set.add(xx)
    feature_list = list(feature_set)
    print('[Info] 已处理视频: {}'.format(feature_list))
    return feature_list


def main():
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--max_frames', type=int, help='max frames from videos')

    args = parser.parse_args()

    args.seed = 19920517
    args.database = "LIVE-VQC"
    args.frame_batch_size = 64
    args.disable_gpu = False

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = '/home/ldq/Downloads/KoNViD-1k/'  # videos dir
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = '/media/ldq/Research/Data/CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/'
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        # n_max = 25  # 最大处理帧数
        n_max = args.max_frames  # 最大处理帧数
        print('[Info] LIVE-VQC 最大帧数: {}'.format(n_max))
        videos_dir = os.path.join(DATASETS_DIR, 'live_vqc')  # 视频文件夹
        features_dir = "CNN_features_LIVE-VQC-{}/".format(n_max)  # CNN特征
        datainfo = 'data/LIVE-VQC.mat'  # 数据信息
        test_excel = "live_vqc_{}.xlsx".format(n_max)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print('[Info] device: {}'.format(device))

    if args.database == 'LIVE-VQC':

        wk = xlsxwriter.Workbook(test_excel)
        ws = wk.add_worksheet()
        titles = [u"视频", u"帧数", u"总耗时", u"解码时间", "特征时间"]
        for i, t in enumerate(titles):
            ws.write(0, i, t)

        vid_names, ni_dict = get_vqc_mat_info(videos_dir)
        dataset = VideoDatasetWithOpenCV(vid_names, ni_dict, n_max)
        vid_names = dataset.get_vid_names()

        feature_list = get_processed_vids(features_dir)

        n_rows = 1
        for i in range(len(dataset)):
            print('[Info]' + '-' * 50)
            name = vid_names[i]
            if name in feature_list:
                print('[Info] 已处理: {}'.format(name))
                continue

            s_time = time.time()
            current_data = dataset[i]
            e_time1 = time.time()
            time_decode = e_time1 - s_time
            print('[Info] 解码时间: {}'.format(time_decode))

            current_video = current_data['video']
            current_score = current_data['score']
            current_name = current_data['name']

            features = get_features(current_video, args.frame_batch_size, device)
            e_time2 = time.time()
            time_feature = e_time2 - e_time1
            print('[Info] 特征时间: {}'.format(time_feature))

            elapsed_time = time.time() - s_time
            print('[Info] 视频: {}, 总耗时: {}'.format(current_name, elapsed_time))

            ws.write(n_rows, 1, name)
            ws.write(n_rows, 2, round(elapsed_time, 4))
            ws.write(n_rows, 3, round(time_decode, 4))
            ws.write(n_rows, 4, round(time_feature, 4))
            n_rows += 1

            np.save(features_dir + str(current_name) + '_resnet-50_res5c', features.to('cpu').numpy())
            np.save(features_dir + str(current_name) + '_score', current_score)

        wk.close()
        print('[Info] LIVE-VQC完成!')
    else:
        Info = h5py.File(datainfo, 'r')
        vid_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                     range(len(Info['video_names'][0, :]))]
        scores = Info['scores'][0, :]
        video_format = Info['video_format'][()].tobytes()[::2].decode()
        width = int(Info['width'][0])
        height = int(Info['height'][0])
        dataset = VideoDataset(videos_dir, vid_names, scores, video_format, width, height)

        for i in range(len(dataset)):
            current_data = dataset[i]
            current_video = current_data['video']
            current_score = current_data['score']
            print('Video {}: length {}'.format(i, current_video.shape[0]))
            features = get_features(current_video, args.frame_batch_size, device)
            np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
            np.save(features_dir + str(i) + '_score', current_score)


if __name__ == "__main__":
    main()
    # get_vqc_mat_info(vid_dir=os.path.join(DATASETS_DIR, 'live_vqc'))
