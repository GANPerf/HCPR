import torch
from torch import nn
import torch.nn.functional as F

class AdaMoCo(nn.Module):
    def __init__(self, src_modelF, src_modelB,src_modelC,momentum_modelF, momentum_modelB,momentum_modelC, features_length, num_classes, dataset_length, temporal_length):
        super(AdaMoCo, self).__init__()

        self.m = 0.999

        self.first_update = True

        self.src_modelF = src_modelF
        self.src_modelB = src_modelB
        self.src_modelC = src_modelC
        self.momentum_modelF = momentum_modelF
        self.momentum_modelB = momentum_modelB
        self.momentum_modelC = momentum_modelC

        self.momentum_modelF.requires_grad_(False)
        self.momentum_modelB.requires_grad_(False)
        self.momentum_modelC.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07

        #queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer(
            "labels", torch.randint(0, num_classes, (self.K,))
        )
        self.register_buffer(
            "idxs", torch.randint(0, dataset_length, (self.K,))
        )
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length))
        )

        self.register_buffer(
            "real_labels", torch.randint(0, num_classes, (dataset_length,))
        )

        self.features = F.normalize(self.features, dim=0)

        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_modelF.parameters(), self.momentum_modelF.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.src_modelB.parameters(), self.momentum_modelB.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.src_modelC.parameters(), self.momentum_modelC.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels.long()
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
        # compute query features
        feats_q = self.src_modelB(self.src_modelF(im_q))
        logits_q = self.src_modelC(feats_q)

        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.momentum_modelB(self.momentum_modelF(im_k))
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k
