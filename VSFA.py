"""Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8
#
# tensorboard --logdir=logs --port=6006
# CUDA_VISIBLE_DEVICES=1 python VSFA.py --database=KoNViD-1k --exp_id=0

import copy
from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime


class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))

        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy')
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #

        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFA(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32, is_bi=False):
        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.is_bi = is_bi
        if not is_bi:
            print('[Info] GRU 普通模式')
            self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
            self.q = nn.Linear(hidden_size, 1)
        else:
            print('[Info] GRU 双向模式')
            self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True, bidirectional=True)
            self.q = nn.Linear(hidden_size * 2, 1)

    def forward(self, input, input_length):
        input = self.ann(input)  # dimension reduction
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):  #
            qi = q[i, :np.int(input_length[i].numpy())]
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        if not self.is_bi:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        else:
            h0 = torch.zeros(1 * 2, batch_size, self.hidden_size, device=device)
        return h0


def get_livevqc_index(feature_dir):
    from utils.project_utils import traverse_dir_files
    paths_list, names_list = traverse_dir_files(feature_dir, ext='.npy')

    res_names = set()
    for name in names_list:
        res_names.add(name.split('_')[0])

    res_names = sorted(list(res_names))
    return res_names


def train_dataset(args, device, features_dir, train_index, val_index, test_index, max_len, scale, n_cross=0):
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    if args.model == 'VSFA':
        model = VSFA().to(device)  #
    elif args.model == 'VSFA-bi':
        model = VSFA(is_bi=True).to(device)  #

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-EXP{}-{}.pt'.format(args.model, args.database, args.exp_id, n_cross)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-EXP{}-{}'.format(args.model, args.database, args.exp_id, n_cross)

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.database, args.model,
                                       args.lr, args.batch_size, args.epochs,
                                       datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    criterion = nn.L1Loss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_val_criterion = -1  # SROCC min

    no_best, max_best = 0, 50  # 没有算法进展
    for epoch in range(args.epochs):
        # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)

        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        # Test
        if args.test_ratio > 0 and not args.notest_during_training:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.item()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, length.float())
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        if not args.disable_visualization:  # record training curves
            writer.add_scalar("loss/train", train_loss, epoch)  #
            writer.add_scalar("loss/val", val_loss, epoch)  #
            writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
            writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
            writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
            writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
            if args.test_ratio > 0 and not args.notest_during_training:
                writer.add_scalar("loss/test", test_loss, epoch)  #
                writer.add_scalar("SROCC/test", SROCC, epoch)  #
                writer.add_scalar("KROCC/test", KROCC, epoch)  #
                writer.add_scalar("PLCC/test", PLCC, epoch)  #
                writer.add_scalar("RMSE/test", RMSE, epoch)  #

        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print()
            print('-' * 50)
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0 and not args.notest_during_training:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC

            no_best = 0
            print('[Info] 算法持续优化! {}'.format(no_best))
            print('[Info] 最优结果 best_val_criterion: {}'.format(best_val_criterion))
        else:
            no_best += 1
            print('[Info] 算法没有进展! {}'.format(no_best))
            if no_best > max_best:
                break

    # Test
    if args.test_ratio > 0:
        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))

    if args.test_ratio > 0:
        return test_loss, SROCC, KROCC, PLCC, RMSE
    else:
        return []


def split_index(index, nc=5):
    """
    拆分index
    """
    n_sample = len(index)
    gap = int(np.ceil(n_sample / nc))
    idxes_list = []

    for s in range(0, n_sample, gap):
        e = s + gap
        if e > n_sample:
            e = n_sample
        s_index = index[s:e]
        idxes_list.append(s_index)

    return idxes_list


def main():
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='CVD2014', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA), or VSFA-bi')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--num_frame', type=int, default=25, help='num frame of features')
    args = parser.parse_args()

    # 测试参数
    # args.database = "LIVE-VQC"
    # args.num_frame = 25
    # args.model = "VSFA"
    # ---------------------------------------- #

    args.decay_interval = int(args.epochs / 10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    n_feature_frame = int(args.num_frame)
    print('[Info] 视频处理帧数: {}'.format(n_feature_frame))

    if args.database == 'KoNViD-1k':
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        features_dir = 'CNN_features_LIVE-VQC-{}/'.format(n_feature_frame)
        datainfo = None

    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print('[Info] 模型名称: {}'.format(args.model))

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    if args.database == "LIVE-VQC":
        print('[Info] 特征文件夹: {}'.format(features_dir))
        index = get_livevqc_index(features_dir)
        ref_ids = copy.deepcopy(index)
        random.shuffle(index)
        max_len = n_feature_frame
        scale = 100
    else:
        Info = h5py.File(datainfo, 'r')  # index, ref_ids
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
        ref_ids = Info['ref_ids'][0, :]  #
        max_len = int(Info['max_len'][0])
        scale = Info['scores'][0, :].max()  # label normalization factor

    if args.database == "LIVE-VQC":
        save_result_file = "live-vqc-5cross-{}-{}.npz".format(args.num_frame, args.model)
        print('[Info] 训练存储信息: {}'.format(save_result_file))
        nc = 5  # 交叉验证
        idxes_list = split_index(index, nc)
        loss_list, srocc_list, krocc_list, plcc_list, rmse_list = [], [], [], [], []
        for i in range(nc):
            print('-' * 100)
            test_index = idxes_list[i % nc]
            val_index = idxes_list[(i + 1) % nc]

            train_index = []
            for j in range((i + 2), (i + nc)):
                train_index += idxes_list[j % nc]
            print('[Info] 训练: {}, 测试: {}, 验证: {}'.format(len(train_index), len(test_index), len(val_index)))

            test_loss, SROCC, KROCC, PLCC, RMSE = train_dataset(args=args, device=device, features_dir=features_dir,
                                                                train_index=train_index, val_index=val_index,
                                                                test_index=test_index,
                                                                max_len=max_len, scale=scale, n_cross=i)

            print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(test_loss, SROCC, KROCC, PLCC, RMSE))

            loss_list.append(np.round(test_loss, 4))
            srocc_list.append(np.round(SROCC, 4))
            krocc_list.append(np.round(KROCC, 4))
            plcc_list.append(np.round(PLCC, 4))
            rmse_list.append(np.round(RMSE, 4))
            print('[Info] 第 {} 次 交叉验证完成!'.format(i))
            print('-' * 100)
            print()

        np.savez(save_result_file,
                 loss_list=np.asarray(loss_list),
                 srocc_list=np.asarray(srocc_list),
                 krocc_list=np.asarray(krocc_list),
                 plcc_list=np.asarray(plcc_list),
                 rmse_list=np.asarray(rmse_list))


    else:
        trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
        testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]

        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)

        train_dataset(args=args, device=device, features_dir=features_dir,
                      train_index=train_index, val_index=val_index, test_index=test_index,
                      max_len=max_len, scale=scale)


if __name__ == "__main__":
    main()
