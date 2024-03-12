import copy

import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time


def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def reset_l0(model):
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def reset_g(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_g1(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def custom_cross_entropy(preds, target):
    return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))


class FastFFTMetaSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False, lr_z=0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0, bnt=30, oht=1.5,
                 is_maml=1, confident_rate=0.05, cr_loop=8, aug_style=0, g_life=50, g_wp=5,
                 is_cbatch=0, reset_type=1, adv2=1.33, gen_y=0, lr_y=0.01, crg_loop=4, y_wp=5):
        super(FastFFTMetaSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.adv2 = adv2
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.ismaml = is_maml

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = int(synthesis_batch_size/cr_loop)
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        self.current_batch = None
        self.current_batch_iter = 0
        self.is_cbatch = is_cbatch
        self.reset_type = reset_type

        self.cr_loop = cr_loop
        self.confident_rate = confident_rate
        self.g_life = g_life
        self.g_wp = g_wp
        self.gen_y = gen_y
        self.lr_y = lr_y
        self.aug_style = aug_style
        self.crg_loop = crg_loop
        self.y_wp = y_wp
        self.bnt = bnt
        self.oht = oht

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g * self.iterations,
                                                   betas=[0.5, 0.999])

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

        self.aug3 = transforms.Compose([
            normalizer,
        ])

    def synthesize(self, targets=None):
        start = time.time()
        self.current_batch = None
        self.current_batch_iter = 0
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        if (self.ep % self.g_life == 0) and self.reset_l0 and self.ep > 0:
            if self.reset_type == 0:
                reset_l0(self.generator)
            elif self.reset_type == 1:
                reset_g(self.generator)
            elif self.reset_type == 2:
                reset_g1(self.generator)
                reset_g1(self.generator)

        self.ep += 1
        bi_list = []
        for __iter in range(self.crg_loop):
            best_inputs = None
            z1 = torch.randn(size=(self.synthesis_batch_size*self.cr_loop, self.nz), device=self.device).requires_grad_()
            z2 = torch.randn(size=(self.synthesis_batch_size * self.cr_loop, self.nz), device=self.device).requires_grad_()
            if self.gen_y == 0:
                targets, ys = self.generate_ys(cr=0.0)
                for cr_iter in range(1, self.cr_loop):
                    cr = cr_iter * self.confident_rate
                    tmp_target, tmp_ys = self.generate_ys(cr=cr)
                    targets = torch.cat((targets, tmp_target))
                    ys = torch.cat((ys, tmp_ys))
            elif self.gen_y == 1:
                targets, yf, yl, cr_vec = self.generate_lys(cr=0)
                for cr_iter in range(1, self.cr_loop):
                    cr = cr_iter * self.confident_rate
                    tmp_target, tmp_yf, tmp_yl, tmp_cr_vec = self.generate_lys(cr=cr)
                    targets = torch.cat((targets, tmp_target))
                    yf = torch.cat((yf, tmp_yf))
                    yl = torch.cat((yl, tmp_yl))
                    cr_vec = torch.cat((cr_vec, tmp_cr_vec))

                yl = yl.requires_grad_(True)
                ys = yf + torch.softmax(yl, dim=1)*cr_vec

            ys = ys.to(self.device)
            targets = targets.to(self.device)

            fast_generator = self.generator.clone()

            if self.gen_y == 0:
                optimizer = torch.optim.Adam([
                    {'params': fast_generator.parameters()},
                    {'params': [z1], 'lr': self.lr_z},
                    {'params': [z2], 'lr': self.lr_z},
                ], lr=self.lr_g, betas=[0.5, 0.999])
            elif self.gen_y == 1:
                optimizer = torch.optim.Adam([
                    {'params': fast_generator.parameters()},
                    {'params': [z1], 'lr': self.lr_z},
                    {'params': [z2], 'lr': self.lr_z},
                    {'params': [yl], 'lr': self.lr_y},
                ], lr=self.lr_g, betas=[0.5, 0.999])

            for it in range(self.iterations):
                inputs = fast_generator(z1=z1, z2=z2, targets=targets)
                if self.aug_style == 0:
                    inputs_aug = self.aug(inputs)
                elif self.aug_style == 1:
                    inputs_aug = self.aug1(inputs)
                elif self.aug_style == 2:
                    inputs_aug = self.aug2(inputs)
                elif self.aug_style == 3:
                    inputs_aug = self.aug3(inputs)

                #############################################
                # Inversion Loss
                #############################################
                t_out = self.teacher(inputs_aug)

                if self.gen_y == 1:
                    ys = yf + torch.softmax(yl, dim=1) * cr_vec
                    loss_y = kldiv(ys, t_out.detach(), reduction='none').sum(1).mean()

                loss_bn = sum([h.r_feature for h in self.hooks])
                loss_oh = custom_cross_entropy(t_out, ys.detach())

                # print("%s - %s - \n%s" % (it, loss_oh, t_out.max(1)[1]))

                if self.adv > 0 and (self.ep >= self.ep_start):
                    s_out = self.student(inputs_aug)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                        1) * mask).mean()  # decision adversarial distillation
                else:
                    loss_adv = loss_oh.new_zeros(1)

                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

                if loss_oh.item() < best_oh:
                    best_oh = loss_oh

                print("%s - bn %s - oh %s - adv %s" % (it, (loss_bn*self.bn).data, (self.oh*loss_oh).data, (self.adv * loss_adv).data))

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data

                optimizer.zero_grad()
                if self.gen_y == 1 and it > self.y_wp:
                    loss_y.backward(retain_graph=True)
                loss.backward()

                if self.ismaml:
                    if it == 0:
                        self.meta_optimizer.zero_grad()
                    fomaml_grad(self.generator, fast_generator)
                    if it == (self.iterations - 1):
                        self.meta_optimizer.step()

                optimizer.step()

            if self.bn_mmt != 0:
                for h in self.hooks:
                    h.update_mmt()

            # REPTILE meta gradient
            if not self.ismaml:
                self.meta_optimizer.zero_grad()
                reptile_grad(self.generator, fast_generator)
                self.meta_optimizer.step()

            self.student.train()
            self.prev_z = (z1, z2, targets)
            end = time.time()

            if self.is_cbatch == 1:
                if self.current_batch is None:
                    self.current_batch = best_inputs
                else:
                    self.current_batch = torch.cat((self.current_batch, best_inputs))

            bi_list.append(best_inputs)
            if (self.ep % self.g_life > self.g_wp or self.ep // self.g_life > 0) \
                    and (self.bn * loss_bn).item() < self.bnt\
                    and (self.oh * loss_oh).item() < self.oht:
                self.data_pool.add(best_inputs)

                dst = self.data_pool.get_dataset(transform=self.transform)
                if self.init_dataset is not None:
                    init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
                    dst = torch.utils.data.ConcatDataset([dst, init_dst])
                if self.distributed:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
                else:
                    train_sampler = None
                loader = torch.utils.data.DataLoader(
                    dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
                    num_workers=4, pin_memory=True, sampler=train_sampler)
                self.data_iter = DataIter(loader)
        return {"synthetic": bi_list}, end - start, best_cost, best_oh

    def sample(self):
        if self.is_cbatch == 1:
            if self.current_batch.shape[0]//self.sample_batch_size > self.current_batch_iter:
                data = self.current_batch[self.current_batch_iter*self.sample_batch_size
                                          :(self.current_batch_iter+1)*self.sample_batch_size]
                self.current_batch_iter += 1
                return data
            if self.current_batch.shape[0]%self.sample_batch_size > 0:
                print("err")
                data = self.current_batch[self.current_batch_iter * self.sample_batch_size:]
                self.current_batch_iter += 1
                return data
        return self.data_iter.next()

    def generate_ys(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, 100))
            target = torch.cat((tmp_label, target))

        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        return target, ys

    def generate_lys(self, cr=0.0, value=3):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, 100))
            target = torch.cat((tmp_label, target))

        yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
        yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        yf = yf.to(device=self.device)

        yl = torch.ones(self.synthesis_batch_size, self.num_classes)*(-value)
        yl.scatter_(1, target.data.unsqueeze(1), value)
        yl = yl.to(device=self.device)

        cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

        return target, yf, yl, cr_vec


    def generate_lys_v2(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, 100))
            target = torch.cat((tmp_label, target))

        yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
        yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        yf = yf.to(device=self.device)

        yl = torch.zeros(size=(self.synthesis_batch_size, self.num_classes), device=self.device)
        cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

        return target, yf, yl, cr_vec