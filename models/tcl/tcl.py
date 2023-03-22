import matplotlib.pyplot as plt
import torch
import argparse
import copy
import torch.nn.functional as F
import tqdm
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from models.basic_template import TrainTask
from network import backbone_dict
from .tcl_wrapper import SimCLRWrapper
from utils.ops import convert_to_ddp, convert_to_cuda
from models import model_dict


@model_dict.register('tcl')
class TCL(TrainTask):

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        kwargs = {'encoder_type': encoder_type, 'in_dim': dim_in, 'fea_dim': opt.feat_dim, 'T': opt.temp,
                  'num_cluster': self.num_cluster, 'mixup_alpha': opt.mixup_alpha, 'num_samples': self.num_samples,
                  'scale1': opt.scale1, 'scale2': opt.scale2}
        if opt.arch == 'simclr':
            tcl = SimCLRWrapper(**kwargs)
        else:
            raise NotImplemented
        tcl.register_buffer('pseudo_labels', self.gt_labels.cpu())

        if opt.syncbn:
            tcl = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcl)

        params = list(tcl.parameters())
        optimizer = torch.optim.SGD(params=params, lr=opt.learning_rate, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
        tcl = convert_to_ddp(tcl)
        self.logger.modules = [tcl, optimizer]
        self.tcl = tcl
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--sep_gmm', action='store_true')

        parser.add_argument('--temp', type=float, help='temp for contrastive loss')
        parser.add_argument('--scale1', type=float)
        parser.add_argument('--scale2', type=float)

        parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='cls_loss_weight')
        parser.add_argument('--align_loss_weight', type=float, default=1.0, help='align_loss_weight')
        parser.add_argument('--ent_loss_weight', type=float, default=1.0, help='ent_loss_weight')
        parser.add_argument('--ne_loss_weight', type=float, default=1.0, help='ne_loss_weight')
        parser.add_argument('--mixup_alpha', type=float, default=1.0, help='cls_loss_weight')

        parser.add_argument('--arch', type=str, default='simclr', help='simclr')

        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        is_warmup = not (self.cur_epoch >= opt.warmup_epochs)
        self.tcl.module.warmup = is_warmup

        images, _ = inputs
        self.tcl.train()

        im_w, im_q, im_k = images

        # compute loss
        contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss = self.tcl(im_w, im_q, im_k, indices)

        # SGD
        self.optimizer.zero_grad()
        loss = contrastive_loss + \
               opt.cls_loss_weight * (cls_loss1 + cls_loss2) + \
               opt.ent_loss_weight * ent_loss + \
               opt.ne_loss_weight * ne_loss + \
               opt.align_loss_weight * align_loss
        loss.backward()
        self.optimizer.step()
        self.logger.msg([contrastive_loss, cls_loss1, cls_loss2, ent_loss, ne_loss, align_loss], n_iter)

    def extract_features(self, model, loader):
        opt = self.opt
        features = torch.zeros(len(loader.dataset), opt.feat_dim).cuda()
        all_labels = torch.zeros(len(loader.dataset)).cuda()
        cluster_labels = torch.zeros(len(loader.dataset), self.num_cluster).cuda()

        model.eval()
        encoder = model.module.encoder_q
        classifier = model.module.classifier_q
        projector = model.module.projector_q

        local_features = []
        local_labels = []
        local_cluster_labels = []
        for inputs in tqdm.tqdm(loader, disable=not self.verbose):
            images, labels = convert_to_cuda(inputs)
            local_labels.append(labels)
            x = encoder(images)
            local_cluster_labels.append(F.softmax(classifier(x), dim=1))
            local_features.append(F.normalize(projector(x), dim=1))
        local_features = torch.cat(local_features, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        local_cluster_labels = torch.cat(local_cluster_labels, dim=0)

        indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

        features.index_add_(0, indices, local_features)
        all_labels.index_add_(0, indices, local_labels.float())
        cluster_labels.index_add_(0, indices, local_cluster_labels.float())

        if dist.is_initialized():
            dist.all_reduce(features, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_labels, op=dist.ReduceOp.SUM)

            mask = torch.norm(features, dim=1) > 1.5
            all_labels[mask] = all_labels[mask] / dist.get_world_size()
            cluster_labels[mask] = cluster_labels[mask] / dist.get_world_size()
            features = F.normalize(features, dim=1)
        labels = all_labels.long()
        return features, cluster_labels, labels

    def hist(self, assignments, is_clean, labels, n_iter, sample_type='context_assignments_hist'):
        fig, ax = plt.subplots()
        ax.hist(assignments[is_clean, labels[is_clean]].cpu().numpy(), label='clean', bins=100, alpha=0.5)
        ax.hist(assignments[~is_clean, labels[~is_clean]].cpu().numpy(), label='noisy', bins=100, alpha=0.5)
        ax.legend()
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        self.logger.save_image(img, n_iter, sample_type=sample_type)
        plt.close()

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt
        assert not opt.whole_dataset

        self.logger.msg_str('Generating the psedo-labels')

        labels = self.gt_labels
        confidence, context_assignments, features, cluster_labels, = self.correct_labels(self.tcl, labels)
        self.tcl.module.confidences.copy_(confidence.float())

        self.evaluate(self.tcl, features, confidence, cluster_labels, labels, context_assignments, n_iter)

    def evaluate(self, model, features, confidence, cluster_labels, labels, context_assignments, n_iter):
        opt = self.opt
        clean_labels = torch.Tensor(
            self.create_dataset(opt.data_folder, opt.dataset, train=True, transform=None)[0].targets).cuda().long()

        is_clean = clean_labels.cpu().numpy() == labels.cpu().numpy()
        self.hist(context_assignments, is_clean, labels, n_iter)
        train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()
        test_features, test_cluster_labels, test_labels = self.extract_features(model, self.test_loader)
        test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()

        from utils.knn_monitor import knn_predict
        knn_labels = knn_predict(test_features, features, clean_labels,
                                 classes=self.num_cluster, knn_k=200, knn_t=0.1)[:, 0]
        self.logger.msg_str(torch.unique(torch.argmax(test_cluster_labels, dim=1), return_counts=True))

        knn_acc = (test_labels == knn_labels).float().mean()

        estimated_noise_ratio = (confidence > 0.5).float().mean().item()
        if opt.scale1 is None:
            self.tcl.module.scale1 = estimated_noise_ratio
        if opt.scale2 is None:
            self.tcl.module.scale2 = estimated_noise_ratio

        noise_accuracy = ((confidence > 0.5) == (clean_labels == labels)).float().mean()
        from sklearn.metrics import roc_auc_score
        context_noise_auc = roc_auc_score(is_clean, confidence.cpu().numpy())
        self.logger.msg([estimated_noise_ratio, noise_accuracy,
                         context_noise_auc, train_acc, test_acc, knn_acc], n_iter)

    def correct_labels(self, model, labels):
        opt = self.opt

        features, cluster_labels, _ = self.extract_features(model, self.memory_loader)
        confidence, context_assignments, centers = self.noise_detect(cluster_labels, labels, features)

        model.module.prototypes.copy_(centers)
        model.module.context_assignments.copy_(context_assignments.float())

        return confidence, context_assignments, features, cluster_labels

    def noise_detect(self, cluster_labels, labels, features):
        opt = self.opt

        centers = F.normalize(cluster_labels.T.mm(features), dim=1)
        context_assignments_logits = features.mm(centers.T) / opt.temp
        context_assignments = F.softmax(context_assignments_logits, dim=1)
        losses = - context_assignments[torch.arange(labels.size(0)), labels]
        losses = losses.cpu().numpy()[:, np.newaxis]
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        labels = labels.cpu().numpy()

        from sklearn.mixture import GaussianMixture
        confidence = np.zeros((losses.shape[0],))
        if opt.sep_gmm:
            for i in range(self.num_cluster):
                mask = labels == i
                c = losses[mask, :]
                gm = GaussianMixture(n_components=2, random_state=0).fit(c)
                pdf = gm.predict_proba(c)
                confidence[mask] = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        else:
            gm = GaussianMixture(n_components=2, random_state=0).fit(losses)
            pdf = gm.predict_proba(losses)
            confidence = (pdf / pdf.sum(1)[:, np.newaxis])[:, np.argmin(gm.means_)]
        confidence = torch.from_numpy(confidence).float().cuda()
        return confidence, context_assignments, centers

    def test(self, n_iter):
        pass

    def train_transform(self, normalize):
        '''
        simclr transform
        :param normalize:
        :return:
        '''
        import torchvision.transforms as transforms
        from utils import TwoCropTransform

        opt = self.opt
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.5),
            transforms.ToTensor(),
            normalize
        ]

        weak_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(opt.img_size,
                                  padding=int(opt.img_size * 0.125)),
            transforms.ToTensor(),
            normalize
        ]

        train_transform = transforms.Compose(train_transform)
        weak_transform = transforms.Compose(weak_transform)

        def ThreeCropTransform(img):
            return TwoCropTransform(train_transform)(img) + [weak_transform(img), ]

        return ThreeCropTransform
