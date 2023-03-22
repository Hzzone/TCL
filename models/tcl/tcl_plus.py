import torch
import argparse
import copy
import torch.nn as nn

from models.basic_template import TrainTask
from network import backbone_dict
from utils.ops import convert_to_ddp, convert_to_cuda, load_network
from models import model_dict


@model_dict.register('tcl_plus')
class TCL(TrainTask):

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        state_dict = torch.load(opt.checkpoint_path, map_location='cpu')
        model_state_dict = load_network(state_dict['tcl'])
        all_assignments = model_state_dict['context_assignments'].cuda()
        all_max_assignments = all_assignments.max(dim=1).values
        all_max_indices = all_assignments.max(dim=1).indices
        mask = (all_max_assignments > opt.high_th)
        self.gt_labels[mask] = all_max_indices[mask]

        msg = encoder.load_state_dict({k[len('encoder_q.'):]: v for k, v in model_state_dict.items() if 'encoder_q' in k})
        print(msg)
        from .tcl_wrapper import Wrapper
        classifier = Wrapper.create_classifier(dim_in, self.num_cluster)
        msg = classifier.load_state_dict(
            {k[len('classifier_q.'):]: v for k, v in model_state_dict.items() if 'classifier_q' in k})
        print(msg)
        encoder = nn.Sequential(encoder, classifier)
        # encoder_ema = nn.Sequential(encoder_type(), Wrapper.create_classifier(dim_in, self.num_cluster))
        encoder_ema = copy.deepcopy(encoder)
        self.q_params, self.k_params = list(encoder.parameters()), list(encoder_ema.parameters())
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        optimizer = torch.optim.SGD(params=encoder.parameters(),
                                    lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        encoder = convert_to_ddp(encoder)
        encoder_ema = encoder_ema.cuda()

        self.logger.modules = [encoder, encoder_ema, optimizer]
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--mixup_alpha', type=float, default=1.0)
        parser.add_argument('--startLabelCorrection', type=int, default=30)
        parser.add_argument('--high_th', type=float, default=0.5)
        parser.add_argument('--ema', type=float, default=0.)
        parser.add_argument('--checkpoint_path', type=int)
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        images, _ = inputs
        self.encoder.train()
        self.encoder_ema.train()

        labels = self.gt_labels[indices]

        criterionCE = torch.nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            from utils import _momentum_update
            _momentum_update(self.q_params, self.k_params, opt.ema)
            for m in self.encoder_ema.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
            predsNoDA = self.encoder_ema(images)

        from .tcl_wrapper import mixup
        im_mix, mix_randind, mix_lam = mixup(images, alpha=opt.mixup_alpha)

        preds_mix = self.encoder(im_mix)

        if self.cur_epoch <= opt.startLabelCorrection:
            loss_mix = mix_lam * criterionCE(preds_mix, labels) + (1 - mix_lam) * criterionCE(preds_mix,
                                                                                              labels[mix_randind])
        else:
            z1 = torch.argmax(predsNoDA, dim=1)
            z2 = z1[mix_randind]
            B = 0.2

            loss_x1 = mix_lam * (1 - B) * criterionCE(preds_mix, labels)
            loss_x1_pred = mix_lam * B * criterionCE(preds_mix, z1)
            loss_x2 = (1 - mix_lam) * (1 - B) * criterionCE(preds_mix, labels[mix_randind])
            loss_x2_pred = (1 - mix_lam) * B * criterionCE(preds_mix, z2)
            loss_mix = loss_x1 + loss_x1_pred + loss_x2 + loss_x2_pred
        loss_mix = loss_mix.mean()

        loss = loss_mix
        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.msg([loss_mix, ], n_iter)

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt
        self.encoder_ema.eval()
        self.encoder.eval()
        encoder = self.encoder_ema
        # encoder = self.encoder

        clean_labels = torch.Tensor(
            self.create_dataset(opt.data_folder, opt.dataset, train=True, transform=None)[0].targets).cuda().long()
        from utils import extract_features
        cluster_labels, labels = extract_features(encoder, self.memory_loader)

        # evaluate
        train_acc = (torch.argmax(cluster_labels, dim=1) == clean_labels).float().mean()

        assert not opt.whole_dataset
        test_cluster_labels, test_labels = extract_features(encoder, self.test_loader)
        test_acc = (test_labels == torch.argmax(test_cluster_labels, dim=1)).float().mean()
        self.logger.msg_str(torch.unique(torch.argmax(test_cluster_labels, dim=1), return_counts=True))

        self.logger.msg_metric([train_acc, test_acc, ], n_iter)

    def test(self, n_iter):
        pass

    def train_transform(self, normalize):
        import torchvision.transforms as transforms

        opt = self.opt
        weak_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(opt.img_size,
                                  padding=int(opt.img_size * 0.125)),
            transforms.ToTensor(),
            normalize
        ]
        weak_transform = transforms.Compose(weak_transform)

        return weak_transform

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([lr, ], n_iter)
