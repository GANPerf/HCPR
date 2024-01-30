import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss1
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pandas as pd
from models.method import TuningBase
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from model import *
from moco import *
from models.randaugment import RandAugmentMC
#import wandb
#torch.cuda.set_device(0)
'''
wandb.init(project='yy',
           name='Ours',
           entity='yangyangshu0520'

           )

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 20,
  "batch_size": 16
}
'''

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):

    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)
class TransformTrain(object):
    def __init__(self,resize_size=256, crop_size=224, mean=imagenet_mean, std=imagenet_std):
        self.strong = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.ori = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip()])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        return [ self.normalize(self.strong(x)),self.normalize(self.strong(x)), self.normalize(self.ori(x))]


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    transform_train = TransformTrain()
    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train()) #image_train()
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["target"] = ImageList_idx(txt_tar, transform=transform_train) #image_train()
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(rationale, fine_tune, model, loader, fea_bank, socre_bank, netF, netB, netC, args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    label_bank = torch.randn(num_sample)  # .cuda()
    pred_bank = torch.randn(num_sample)
    nu=[]
    # s=[]
    # var_all=[]

    logits = []
    features = []

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            indx = data[2]
            paths = data[3]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            """if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)"""

            # if args.singular:
            # _, ss, _ = torch.svd(fea)
            # s10=ss[:10]/ss[0]
            # s.append(s10)
            if fine_tune == 0:
                outputs = netC(fea)
            else:
                fea = model.inference(inputs)
                outputs = netC(fea)

            softmax_out = nn.Softmax()(outputs)
            nu.append(torch.mean(torch.svd(softmax_out)[1]))
            output_f_norm = F.normalize(fea)
            # fea_bank[indx] = output_f_norm.detach().clone().cpu()
            label_bank[indx] = labels.float().detach().clone()  # .cpu()
            pred_bank[indx] = outputs.max(-1)[1].float().detach().clone().cpu()
            if start_test:
                all_input = inputs.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_idx = indx.int()
                all_path = paths
                all_fea = output_f_norm.cpu()
                start_test = False
            else:
                all_input = torch.cat((all_input, inputs.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, indx.int()), 0)
                all_path = np.concatenate((all_path, paths), axis=0)
                all_fea = torch.cat((all_fea, output_f_norm.cpu()), 0)

    probs = F.softmax(all_output, dim=1)
    rand_idxs = torch.randperm(len(all_fea)).cuda()
    banks = {
        "features": all_fea[rand_idxs][: 16384].cuda(),
        "probs": probs[rand_idxs][: 16384].cuda(),
        "ptr": 0,
    }
    _, predict = torch.max(all_output, 1)
    # for confidence
    prob = torch.softmax(all_output.detach(), dim=-1)
    confidence, _ = torch.max(prob, dim=-1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )

    dataframe = pd.DataFrame(
        {'image': all_path, 'real label': all_label,
         'predict_label': predict, 'confidence': confidence})
    dataframe.to_csv(str(args.name) + '.csv', index=False)

    _, socre_bank_ = torch.max(socre_bank, 1)
    distance = fea_bank.cpu() @ fea_bank.cpu().T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=4)
    score_near = socre_bank_[idx_near[:, :]].float().cpu()  # N x 4

    """acc1 = (score_near.mean(
        dim=-1) == score_near[:, 0]).sum().float() / score_near.shape[0]"""
    acc1 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == pred_bank)
    ).sum().float() / score_near.shape[0]
    acc2 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == label_bank)
    ).sum().float() / score_near.shape[0]

    """if True:
        nu_mean=sum(nu)/len(nu)"""

    # s10_avg=torch.stack(s).mean(0)
    # print('nuclear mean: {:.2f}'.format(nu_mean))

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        if True:
            return aacc, acc, banks  # , acc1, acc2#, nu_mean, s10_avg

    else:
        return accuracy * 100, mean_ent


def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight

def create_model(arch, args):
    model = Resnet(arch, args)

    model = model.cuda()
    return model

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank.cuda())
        _, idxs = distances.sort()
        idxs = idxs[:, : args.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1).cuda()
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


def refine_predictions(
        features,
        probs,
        banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()


    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def train_target(args):
    loss_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    loss_weight.data.fill_(0.1)
    alpha_ours = 0.1

    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    momentum_netF = network.ResBase(res_name=args.net).cuda()

    momentum_netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    momentum_netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir_src + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath, map_location='cuda:0'))
    modelpath = args.output_dir_src + "/source_B.pt"
    netB.load_state_dict(torch.load(modelpath, map_location='cuda:0'))
    modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(modelpath, map_location='cuda:0'))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{"params": v, "lr": args.lr * 1}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1


    for paramback, param1 in zip(netF.parameters(), momentum_netF.parameters()):
        param1.data.copy_(paramback.data)
    for paramback, param1 in zip(netB.parameters(), momentum_netB.parameters()):
        param1.data.copy_(paramback.data)
    for paramback, param1 in zip(netC.parameters(), momentum_netC.parameters()):
        param1.data.copy_(paramback.data)



    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, 12).cuda()

    '''
    moco_model = AdaMoCo(src_modelF=netF, src_modelB=netB, src_modelC=netC,
                         momentum_modelF=momentum_netF, momentum_modelB=momentum_netB, momentum_modelC=momentum_netC,
                         features_length=256, num_classes=args.class_num, dataset_length=num_sample,
                         temporal_length=5).cuda()
    '''
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[2]
            # labels = data[1]
            #inputs = inputs.cuda()
            output = netB(netF(inputs[2].cuda()))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)

            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    count = 0
    iter_fm = 0

    netF.train()
    netB.train()
    netC.train()
    #moco_model.train()
    acc_log = 0

    real_max_iter = max_iter

    rationale = []
    fine_tune = 0
    model = nn.Sequential(netF, netB)
    if args.dset == "VISDA-C":
        acc, accc, banks = cal_acc(rationale, fine_tune, model,
                            dset_loaders["test"],
                            fea_bank,
                            score_bank,
                            netF,
                            netB,
                            netC,
                            args,
                            flag=True,
                            )

    while iter_num < real_max_iter:
        try:
            inputs_test, target_label, tar_idx, path = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, target_label, tar_idx, path = next(iter_test)

        if target_label.size(0) == 1:
            continue

        #*******************
        if iter_num % (interval_iter * 10) == 0 and args.cls_par > 0 and count <= 7:
            netF.eval()
            netB.eval()
            count += 1
            #if iter_num/interval_iter >= (args.max_epoch-1)*10:
            if count > 7:
                rationale = obtain_a(dset_loaders['test'], netF, netB, netC, args)
                mem_label, retrain_input_all_w, retrain_input_all_s1,retrain_input_all_s2, retrain_feat_all, retrain_fea_all, retrain_idx_all, retrain_pseudo_all, retrain_label_all, retrain_output_all = obtain_label(dset_loaders['target'], netF, netB, netC, args, rationale, iter_num, interval_iter)
                #mem_label = mem_label.cuda()
                mem_label = torch.from_numpy(mem_label).cuda()
                a = torch.randperm(retrain_input_all_s1.size(0))

            else:
                retrain_input_all_w = []
                retrain_input_all_s1 = []
                retrain_input_all_s2 = []
                retrain_pseudo_all = []
                retrain_feat_all = []
                a = []
                rationale = []
                retrain_fea_all = fea_bank
                retrain_output_all = score_bank
            netF.train()
            netB.train()
        #************************

        if count > 7:
            # shuffle数据
            #a = torch.randperm(retrain_input_all_s1.size(0))
            retrain_input_all_s1 = retrain_input_all_s1[a, :, :, :]
            retrain_input_all_s2 = retrain_input_all_s2[a, :, :, :]
            retrain_input_all_w = retrain_input_all_w[a, :, :, :]

            retrain_pseudo_all = retrain_pseudo_all[a].to(torch.int64)

            retrain_feat_all = retrain_feat_all[a, :]

            #epoch_iter = math.ceil(retrain_input_all_w.size(0) / args.batch_size_fm)

            if iter_fm <= math.ceil(retrain_input_all_s1.size(0) / args.batch_size_fm) - 1:
                # inputs_ft_0 = retrain_input_all_0[i*args.batch_size:(i+1)*args.batch_size,:,:,:].cuda()
                # inputs_ft_1 = retrain_input_all_1[i * args.batch_size:(i + 1) * args.batch_size, :, :, :].cuda()
                inputs_ft_w = retrain_input_all_w[iter_fm * args.batch_size_fm:(iter_fm + 1) * args.batch_size_fm, :, :, :].cuda()
                feat_ft = retrain_feat_all[i*args.batch_size:(i+1)*args.batch_size,:].cuda()
                pred_ft = retrain_pseudo_all[iter_fm * args.batch_size_fm:(iter_fm + 1) * args.batch_size_fm].cuda()
                iter_fm += 1
            else:
                # inputs_ft_0 = retrain_input_all_0[i * args.batch_size:, :, :, :].cuda()
                # inputs_ft_1 = retrain_input_all_1[i * args.batch_size:, :, :, :].cuda()
                inputs_ft_w = retrain_input_all_w[iter_fm * args.batch_size_fm:, :, :, :].cuda()
                feat_ft = retrain_feat_all[i * args.batch_size:, :].cuda()
                pred_ft = retrain_pseudo_all[iter_fm * args.batch_size_fm:].cuda()
                iter_fm = 0

            if inputs_ft_w.size(0) == 1 or inputs_ft_w.size(0) == 0:
                iter_fm = 0
                continue
            output = netC(netB(netF(inputs_ft_w.cuda())))
            #softmax_out = nn.Softmax(dim=1)(output)

            Lx = F.cross_entropy(output, pred_ft, reduction='mean')

        #inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test[2].cuda()))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # output_re = softmax_out.unsqueeze(1)

        '''
        #**************shot
        if args.cls_par > 0:  #0.3
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()


        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss1.Entropy(softmax_out))  #Lent
            entropy_loss *= 0.5
            if args.gent:     #Ldiv
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par #1.0
            classifier_loss += im_loss
        #*******************
        '''
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C


            distance_a = output_f_ @ retrain_fea_all.T  # batch*256, select_num_sample*256->batch*select_num_sample
            _, idx_near_a = torch.topk(distance_a, dim=-1, largest=True, k=args.K + 1)
            idx_near_a = idx_near_a[:, 1:]  # batch x K
            score_near_a = retrain_output_all[idx_near_a]  # batch x K x C  这里的K是topk的k，C是类的个数12

            vis = torch.reshape(score_near_a, ((score_near_a.size(0)*score_near_a.size(1)), score_near_a.size(2)))
            vis = vis.float().cpu().numpy()
            dataframe = pd.DataFrame(vis)
            dataframe.to_csv('softmax_output' + '.csv', index=False)

            #****
            _, idx_far = torch.topk(distance, dim=-1, largest=False, k=2 + 1)
            idx_far = idx_far[:, 1:]  # batch x K
            score_far = score_bank[idx_far]  # batch x K x C
            #***

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        # nn
        softmax_out_un_far = softmax_out.unsqueeze(1).expand(
            -1, 2, -1
        )  # batch x K x C
        '''
        feats_w, logits_w = moco_model(inputs_test[2].cuda(), cls_only=True)

        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, banks)
        #probs_w = F.softmax(logits_w, dim=1)
        #pseudo_labels_w = probs_w.max(1)[1]

        _, logits_q, logits_ctr, keys = moco_model(inputs_test[0].cuda(), inputs_test[1].cuda())


        loss_ctr = contrastive_loss(
            logits_ins=logits_ctr,
            pseudo_labels=moco_model.mem_labels[tar_idx],
            mem_labels=moco_model.mem_labels[moco_model.idxs]
        )

        # update key features and corresponding pseudo labels
        if iter_num % len(dset_loaders["target"]) == 0:
            epoch = iter_num // len(dset_loaders["target"])
            moco_model.update_memory(epoch, tar_idx.cuda(), keys, pseudo_labels_w, target_label.cuda())
        '''
        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product


        loss_a = torch.mean(
            (F.kl_div(softmax_out_un, score_near_a.cuda(), reduction="none").sum(-1)).sum(1)
        )  # Equal to dot product

        pseudo_label = torch.softmax(outputs_test.detach() / args.T, dim=-1)  # logits_u_w
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold_fm).float()  # greater and equal（大于等于）

        logits_u_s = netC(netB(netF(inputs_test[0].cuda())))
        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

        score_far = torch.reshape(score_far,(score_far.size(0)*score_far.size(1), score_far.size(2)))
        softmax_out_un_far = torch.reshape(softmax_out_un_far,(softmax_out_un_far.size(0)*softmax_out_un_far.size(1),softmax_out_un_far.size(2)))
        dot_neg_far = softmax_out_un_far @ score_far.T
        dot_neg_far = (dot_neg_far.cuda()).sum(-1)  # batch*K
        neg_pred_far = torch.mean(dot_neg_far)


        mask = torch.ones((inputs_test[2].shape[0], inputs_test[2].shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)

        gap = max_iter // 6
        if iter_num % gap == 0:
            alpha_ours = alpha_ours * 0.1

        if count > 7:
            total_loss = Lx + args.lambda_u * Lu #+ L_a
        else:
            total_loss = loss_a + alpha * neg_pred #+ loss_ctr #+ neg_pred_far * alpha #+ neg_pred * alpha

        #update_labels(banks, tar_idx, feats_w, logits_w)

        #if iter_num % 30 == 0:
            #wandb.log({"loss_a": loss_a})
            #wandb.log({"loss_neg_pred": neg_pred})
            #wandb.log({"loss_ctr": loss_ctr})
            #wandb.watch(moco_model)

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_c.step()

        fine_tune = 0
        model = nn.Sequential(netF, netB)


        if iter_num % (interval_iter*10) == 0 or iter_num == max_iter: # or iter_num / interval_iter >= (args.max_epoch - 1) * 10:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == "VISDA-C":
                acc, accc, banks = cal_acc(rationale, fine_tune, model,
                    dset_loaders["test"],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                    "Task: {}, Iter:{}/{};  Acc on target: {:.2f}".format(
                        args.name, iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            netF.train()
            netB.train()
            netC.train()
    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args, rationale, iter_num, interval_iter):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            idx = data[2]
            paths = data[3]
            feat = netF(inputs[2].cuda())
            feas = netB(feat)
            outputs = netC(feas)
            if start_test:
                all_input_w = inputs[2].float().cpu()
                all_input_s1 = inputs[0].float().cpu()
                all_input_s2 = inputs[1].float().cpu()
                all_feat = feat.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_idx = idx.int()
                all_path = paths
                start_test = False
            else:
                all_input_w = torch.cat((all_input_w, inputs[2].float().cpu()),0)
                all_input_s1 = torch.cat((all_input_s1, inputs[0].float().cpu()), 0)
                all_input_s2 = torch.cat((all_input_s2, inputs[1].float().cpu()), 0)
                all_feat = torch.cat((all_feat, feat.float().cpu()),0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, idx.int()), 0)
                all_path = np.concatenate((all_path, paths), axis=0)

    all_output = nn.Softmax(dim=1)(all_output)
    #if iter_num/interval_iter >= (args.max_epoch-1)*10:
    retrain_input_all_w, retrain_input_all_s1,retrain_input_all_s2, retrain_feat_all, retrain_fea_all, retrain_output_all, retrain_idx_all, retrain_pseudo_all, retrain_label_all = select_sort_num(args, rationale, netF, netB, netC, all_feat, all_fea, all_output, all_label, all_idx, all_input_w, all_input_s1, all_input_s2)
    #else:
        #retrain_input_all, retrain_feat_all, retrain_fea_all, retrain_output_all, retrain_idx_all, retrain_pseudo_all, retrain_label_all = [],[],[],[],[],[],[]

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    #all_output[:,33] = all_output[:,33]*2
    confidence, predict = torch.max(all_output, 1)

    '''
    #********************for semi-supervised
    con_np = confidence.float().cpu().numpy()
    con_index_np = np.argwhere(con_np >= 0.95)
    con_index = torch.from_numpy(con_index_np.squeeze())
    retrain_input_all_w = all_input_w[con_index, :,:,:]
    retrain_input_all_s1 = all_input_s1[con_index,:,:,:]
    retrain_input_all_s2 = all_input_s2[con_index, :, :, :]
    retrain_feat_all = all_feat[con_index,:]
    retrain_fea_all = all_fea[con_index, :]
    retrain_output_all = all_output[con_index,:]
    retrain_idx_all = all_idx[con_index]
    retrain_pseudo_all = predict[con_index]
    retrain_label_all = all_label[con_index]
    
    accuracy_con = torch.sum(torch.squeeze(predict[con_index]).float() == all_label[con_index]).item() / float(
        all_label[con_index].size()[0])

    print("the selected number: %.4f" % (float(all_label[con_index].size()[0])))
    print("the selected acc: %.4f" % (accuracy_con * 100))
    #*******************
    '''

    #************************
    #class_id = np.where(predict == 33)
    #class_id = class_id[0]
    #print(class_id)
    #*************************

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])


    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)  # all_fea
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        #retrain_fea_all = torch.cat((retrain_fea_all, torch.ones(retrain_fea_all.size(0), 1)), 1) #all_fea
        #retrain_fea_all = (retrain_fea_all.t() / torch.norm(retrain_fea_all, p=2, dim=1)).t()

    #retrain_fea_all = retrain_fea_all.float().cpu().numpy() #N*feat_dim,4365*257
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()  #all_output
    for i in range(2):
        #if i == 0:
            #initc = aff.transpose().dot(retrain_fea_all)
            #initc = initc / (1e-8 + aff.sum(axis=0)[:,None]) #mean feat for all probility class  class_num*feat_dim,65*257
        #else:
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # mean feat for all probility class  class_num*feat_dim,65*257
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]          #0-class_num-1, 0-64

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    '''
    featc = torch.from_numpy(initc[:,0:256])
    all_a_predict, all_a_label, all_a_path, all_a_idx = change_half_label(args, netF, netB, netC, predict, all_feat, all_input, all_output, all_label, all_path, featc, rationale, all_idx)
    for i in range(all_a_predict.size(0)):
        all_a_predict[all_a_idx[i]] = all_a_predict[i]
    '''

    return predict.astype('int'), retrain_input_all_w, retrain_input_all_s1,retrain_input_all_s2, retrain_feat_all, retrain_fea_all, retrain_idx_all, retrain_pseudo_all, retrain_label_all, retrain_output_all


def select_sort_num(args, rationale, netF, netB, netC, all_feat, all_fea, all_output, all_label, all_idx, all_input_w, all_input_s1, all_input_s2):
    rationale = rationale.float().cpu().numpy()
    all_feat = all_feat.float().cpu().numpy()  # (4365,2048)
    feat_select = np.ones((args.sort_num * all_output.size(0), 2048))  # (5*4365, 2048)
    a_select = np.ones((args.sort_num * all_output.size(0), 2048))  # (5*4365, 2048)

    sorted_logits, sort_index = torch.sort(all_output, descending=True, dim=-1)  # 按行排序
    psuedo_select = sort_index[:, 0:args.sort_num]

    model = nn.Sequential(netF, netB, netC)
    target_layers = [netF.layer4[-1]]
    i = 0
    print("calculate more reliable samples!")
    for j in tqdm(range(all_output.size(0))):  # 4365
        input_tensor = all_input_w[j, :, :, :]
        input_tensor = torch.unsqueeze(input_tensor, 0)

        for k in range(args.sort_num):
            target_category = [int(psuedo_select[j, k])]  # None
            target_category = np.array(target_category)

            methods = \
                {"gradcam": GradCAM,
                 "scorecam": ScoreCAM,
                 "gradcam++": GradCAMPlusPlus,
                 "ablationcam": AblationCAM,
                 "xgradcam": XGradCAM,
                 "eigencam": EigenCAM,
                 "eigengradcam": EigenGradCAM,
                 "layercam": LayerCAM,
                 "fullgrad": FullGrad}
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 1  # 64/4

                grayscale_cam, weights = cam(input_tensor=input_tensor,
                                             target_category=target_category,
                                             aug_smooth=args.aug_smooth,
                                             eigen_smooth=args.eigen_smooth)  # grayscale_cam:batch*224*224; weights有+有-
                weights = weights[0, :]
                a_select[i, :] = all_feat[j, :] * np.maximum(weights, 0)
                i = i + 1

    a_select = torch.from_numpy(a_select)
    a_select = F.normalize(a_select, p=2, dim=1)  # 按行归一化
    a_select = a_select.float().cpu().numpy()
    psuedo_select = psuedo_select.int().cpu().numpy()
    psuedo_select_reshape = psuedo_select.reshape((1, args.sort_num * all_output.size(0)))
    psuedo_select_reshape = np.squeeze(psuedo_select_reshape)
    score = np.ones(args.sort_num * all_output.size(0))  # 5*4365
    for i in range(len(score)):  # 5*4365
        score[i] = np.linalg.norm(a_select[i, :] - rationale[psuedo_select_reshape[i], :])

    all_label_5 = np.ones(args.sort_num * all_output.size(0))  # 5*4365
    for i in range(len(all_label_5)):  # 5*4365
        all_label_5[i] = all_label[int(i / 5)]

    rank = np.ones(args.sort_num * all_output.size(0))  # 5*4365

    for i in range(all_output.size(1)):  # 65
        id = np.where(psuedo_select_reshape == i)
        sub_score = score[id]
        rank_sub_score = sub_score[np.argsort(sub_score)]  # 从小到大排序，找到要查找的分数占该类的位置
        for j in range(len(sub_score)):
            for k in range(len(rank_sub_score)):
                if sub_score[j] == rank_sub_score[k]:
                    rank[id[0][j]] = k

    start = True
    rank = torch.from_numpy(rank)
    psuedo_select = torch.from_numpy(psuedo_select)
    all_feat = torch.from_numpy(all_feat)

    for i in tqdm(range(all_output.size(0))):  # 4365
        sub_rank = rank[i * args.sort_num:(i + 1) * args.sort_num]
        sub_score = score[i * args.sort_num:(i + 1) * args.sort_num]
        b, b_idx = torch.sort(sub_rank)
        if b[0] < args.tau1 and b[1] > args.tau2:
            # if sub_score[b_idx[0]] <= sub_score[b_idx[1]]:
            # min_r = b_idx[0]
            # else:
            # min_r = b_idx[1]
            if start:
                retrain_pseudo_all = psuedo_select[
                    i, torch.argmin(sub_rank)].int().cpu()  # tensor(num)->tensor([num])  torch.argmin(sub_rank)
                retrain_pseudo_all = np.expand_dims(retrain_pseudo_all, 0)
                retrain_pseudo_all = torch.tensor(retrain_pseudo_all)

                retrain_input_all_w = all_input_w[i, :, :, :].float().cpu()
                retrain_input_all_w = torch.unsqueeze(retrain_input_all_w, 0)

                retrain_input_all_s1 = all_input_s1[i, :, :, :].float().cpu()
                retrain_input_all_s1 = torch.unsqueeze(retrain_input_all_s1, 0)

                retrain_input_all_s2 = all_input_s2[i, :, :, :].float().cpu()
                retrain_input_all_s2 = torch.unsqueeze(retrain_input_all_s2, 0)

                #retrain_input_all = all_input[i, :, :, :].float().cpu()
                #retrain_input_all = torch.unsqueeze(retrain_input_all, 0)


                retrain_feat_all = all_feat[i, :].float().cpu()
                retrain_feat_all = torch.unsqueeze(retrain_feat_all, 0)

                retrain_fea_all = all_fea[i, :].float().cpu()
                retrain_fea_all = torch.unsqueeze(retrain_fea_all, 0)

                retrain_output_all = all_output[i, :].float().cpu()
                retrain_output_all = torch.unsqueeze(retrain_output_all, 0)

                retrain_idx_all = all_idx[i].int()
                retrain_idx_all = np.expand_dims(retrain_idx_all, 0)
                retrain_idx_all = torch.tensor(retrain_idx_all)

                retrain_label_all = all_label[i].float().cpu()
                retrain_label_all = np.expand_dims(retrain_label_all, 0)
                retrain_label_all = torch.tensor(retrain_label_all)

                #retrain_path_all = all_path[i]
                #retrain_path_all = np.expand_dims(retrain_path_all, 0)
                start = False
            else:
                temp = psuedo_select[i, torch.argmin(sub_rank)]  # tensor(num)->tensor([num])
                temp = np.expand_dims(temp, 0)
                temp = torch.tensor(temp)
                retrain_pseudo_all = torch.cat((retrain_pseudo_all, temp), 0)

                #retrain_input_all = torch.cat((retrain_input_all, torch.unsqueeze(all_input[i, :, :, :], 0)), 0)

                retrain_input_all_w = torch.cat((retrain_input_all_w, torch.unsqueeze(all_input_w[i, :, :, :], 0)), 0)
                retrain_input_all_s1 = torch.cat((retrain_input_all_s1, torch.unsqueeze(all_input_s1[i, :, :, :], 0)),
                                                 0)
                retrain_input_all_s2 = torch.cat((retrain_input_all_s2, torch.unsqueeze(all_input_s2[i, :, :, :], 0)),
                                                 0)

                retrain_feat_all = torch.cat((retrain_feat_all, torch.unsqueeze(all_feat[i, :], 0)), 0)

                retrain_fea_all = torch.cat((retrain_fea_all, torch.unsqueeze(all_fea[i, :], 0)), 0)

                retrain_output_all = torch.cat((retrain_output_all, torch.unsqueeze(all_output[i, :], 0)), 0)

                temp = all_idx[i]  # tensor(num)->tensor([num])
                temp = np.expand_dims(temp, 0)
                temp = torch.tensor(temp)
                retrain_idx_all = torch.cat((retrain_idx_all, temp), 0)

                temp = all_label[i]  # tensor(num)->tensor([num])
                temp = np.expand_dims(temp, 0)
                temp = torch.tensor(temp)
                retrain_label_all = torch.cat((retrain_label_all, temp), 0)

                #temp = np.expand_dims(all_path[i], 0)
                #retrain_path_all = np.concatenate((retrain_path_all, temp), 0)
    '''
    accuracy = torch.sum(torch.squeeze(retrain_pseudo_all).float() == retrain_label_all).item() / float(
        retrain_label_all.size()[0])


    print("the first time selected number: %.4f" % (retrain_label_all.size()[0]))
    print("the first time selected acc: %.4f" % (accuracy * 100))

    for i in tqdm(range(all_output.size(0))):  # 4365
        if all_idx[i] not in retrain_idx_all:
            sub_rank = rank[i * args.sort_num:(i + 1) * args.sort_num]

            rank_rank = torch.ones(sub_rank.size(0))
            predict_id = torch.tensor([0,1,2,3,4])
            temp = sub_rank[torch.argsort(sub_rank,descending=False)]  # 从小到大排序，找到要查找的分数占该类的位置
            for j in range(sub_rank.size(0)):
                for k in range(temp.size(0)):
                    if sub_rank[j] == temp[k]:
                        rank_rank[j] = k


            #b, b_idx = torch.sort(sub_rank)

            #temp = psuedo_select[i, args.sort_num-1-torch.argmin(torch.abs(torch.flip(rank_rank, dims=[0])+torch.flip(predict_id, dims=[0])))]  # tensor(num)->tensor([num])
            temp = psuedo_select[i, torch.argmin(rank_rank + predict_id)]  # tensor(num)->tensor([num])
            #temp = psuedo_select[i, b_idx[1]]

            temp = np.expand_dims(temp, 0)
            temp = torch.tensor(temp)
            retrain_pseudo_all = torch.cat((retrain_pseudo_all, temp), 0)

            retrain_input_all = torch.cat((retrain_input_all, torch.unsqueeze(all_input[i, :, :, :], 0)), 0)

            retrain_feat_all = torch.cat((retrain_feat_all, torch.unsqueeze(all_feat[i, :], 0)), 0)

            retrain_fea_all = torch.cat((retrain_fea_all, torch.unsqueeze(all_fea[i, :], 0)), 0)

            retrain_output_all = torch.cat((retrain_output_all, torch.unsqueeze(all_output[i, :], 0)), 0)

            temp = all_idx[i]  # tensor(num)->tensor([num])
            temp = np.expand_dims(temp, 0)
            temp = torch.tensor(temp)
            retrain_idx_all = torch.cat((retrain_idx_all, temp), 0)

            temp = all_label[i]  # tensor(num)->tensor([num])
            temp = np.expand_dims(temp, 0)
            temp = torch.tensor(temp)
            retrain_label_all = torch.cat((retrain_label_all, temp), 0)

            temp = np.expand_dims(all_path[i], 0)
            retrain_path_all = np.concatenate((retrain_path_all, temp), 0)
    '''

    accuracy = torch.sum(torch.squeeze(retrain_pseudo_all).float() == retrain_label_all).item() / float(
        retrain_label_all.size()[0])

    print("the selected number: %.4f" % (retrain_label_all.size()[0]))
    print("the selected acc: %.4f" % (accuracy * 100))

    dataframe = pd.DataFrame(
        {'real label': all_label_5,
         'predict_label': psuedo_select_reshape, 'score': score, 'rank': rank})
    dataframe.to_csv(str(args.name) + 'retrain_5plus' + '.csv', index=False)

    #dataframe = pd.DataFrame(
        #{'image': retrain_path_all, 'real label': retrain_label_all,
         #'predict_label': retrain_pseudo_all})
    #dataframe.to_csv(str(args.name) + 'retrain' + '.csv', index=False)

    return retrain_input_all_w, retrain_input_all_s1,retrain_input_all_s2, retrain_feat_all, retrain_fea_all, retrain_output_all, retrain_idx_all, retrain_pseudo_all, retrain_label_all


def obtain_a(loader, netF, netB, netC, args):
    print("obtain rationale !")
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feat = netF(inputs)
            feas = netB(feat)
            outputs = netC(feas)
            if start_test:
                all_input = inputs.float().cpu()
                all_fea = feas.float().cpu()
                all_feat = feat.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_input = torch.cat((all_input, inputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_feat = torch.cat((all_feat, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    K = all_output.size(1)

    all_feat = all_feat.float().cpu().numpy()
    inita = torch.ones(K, 2048)  # 65*2048
    for i in tqdm(range(K)):  # K is the num_class
        class_id = np.where(predict == i)
        class_id = class_id[0]
        all_input_class = all_input[class_id]
        all_feat_class = all_feat[class_id]
        all_weight_feat = torch.ones(len(class_id), 2048)

        # gradcam for testing set
        model = nn.Sequential(netF, netB, netC)
        target_layers = [netF.layer4[-1]]
        for j in range(len(class_id)):  # 子类的个数
            input_tensor = all_input_class[j, :, :, :]
            input_tensor = torch.unsqueeze(input_tensor, 0)

            target_category = [int(i)]  # None
            target_category = np.array(target_category)

            methods = \
                {"gradcam": GradCAM,
                 "scorecam": ScoreCAM,
                 "gradcam++": GradCAMPlusPlus,
                 "ablationcam": AblationCAM,
                 "xgradcam": XGradCAM,
                 "eigencam": EigenCAM,
                 "eigengradcam": EigenGradCAM,
                 "layercam": LayerCAM,
                 "fullgrad": FullGrad}
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 1  # 64/4

                grayscale_cam, weights = cam(input_tensor=input_tensor,
                                             target_category=target_category,
                                             aug_smooth=args.aug_smooth,
                                             eigen_smooth=args.eigen_smooth)  # grayscale_cam:batch*224*224; weights有+有-
                weights = weights[0, :]
                # print(i)
                weighted_feat = all_feat_class[j, :] * np.maximum(weights, 0)
                weighted_feat = torch.from_numpy(weighted_feat).cuda()
                all_weight_feat[j, :] = weighted_feat
        # center_class = netB(all_weight_feat.cuda())  #因为netB没有relu，所以有负数
        inita[i, :] = torch.sum(all_weight_feat, dim=0) / len(class_id)

    inita = F.normalize(inita, p=2, dim=1)  # 按行归一化

    return inita

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument('--epoch_ft', type=int, default=40, help="max iterations for fine-tuning")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--batch_size_fm", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="VISDA-C")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="weight/target/")
    parser.add_argument("--output_src", type=str, default="weight/source/")
    parser.add_argument("--tag", type=str, default="LPA")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=True)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")
    parser.add_argument('--num_neighbors', default=10, type=int)

    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold_fm', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--tau1', default=400, type=int)
    parser.add_argument('--tau2', default=800, type=int)


    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--sort_num', type=int, default=5)
    parser.add_argument('--queue_size', type=int, default=32, help='queue size for each class')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "VISDA-C":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = "./data/"
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src, args.da, args.dset, names[args.s][0].upper()
        )
        args.output_dir = osp.join(
            args.output,
            args.da,
            args.dset,
            names[args.s][0].upper() + names[args.t][0].upper(),
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
