#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/23
"""
import os
import numpy as np

from root_dir import RESULTS_DIR


def get_avg_and_std(data):
    arr = np.array(data)
    v_avg = np.average(arr)
    v_std = np.std(arr)
    return np.round(v_avg, 4), np.round(v_std, 4)


def main():
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-25-VSFA.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-25-VSFA-bi.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-50-VSFA.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-50-VSFA-bi.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-100-VSFA.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-100-VSFA-bi.npz')
    # data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-300-VSFA.npz')
    data_path = os.path.join(RESULTS_DIR, 'test', 'live-vqc-5cross-300-VSFA-bi.npz')

    data_npz = np.load(data_path)
    loss_list = data_npz['loss_list']
    avg_loss, std_loss = get_avg_and_std(loss_list)
    print("[Info] loss: {}, avg: {}, var: {}".format(loss_list, avg_loss, std_loss))

    srocc_list = data_npz['srocc_list']
    avg_srocc, std_srocc = get_avg_and_std(srocc_list)
    print("[Info] srocc: {}, avg: {}, var: {}".format(srocc_list, avg_srocc, std_srocc))

    krocc_list = data_npz['krocc_list']
    avg_krocc, std_krocc = get_avg_and_std(krocc_list)
    print("[Info] krocc: {}, avg: {}, var: {}".format(krocc_list, avg_krocc, std_krocc))

    plcc_list = data_npz['plcc_list']
    avg_plcc, std_plcc = get_avg_and_std(plcc_list)
    print("[Info] plcc: {}, avg: {}, var: {}".format(plcc_list, avg_plcc, std_plcc))

    rmse_list = data_npz['rmse_list']
    avg_rmse, std_rmse = get_avg_and_std(rmse_list)
    print("[Info] rmse: {}, avg: {}, var: {}".format(rmse_list, avg_rmse, std_rmse))


if __name__ == "__main__":
    main()
