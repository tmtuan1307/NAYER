import argparse
from math import gamma
import os
import random
import shutil
import warnings

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transformstransforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', default='nldf', choices=['nldf'])
parser.add_argument('--adv', default=1.33, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=10, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0.5, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/', type=str)
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

parser.add_argument('--bn_mmt', default=0.9, type=float,
                    help='momentum when fitting batchnorm statistics')

# CDF
parser.add_argument('--log_tag', default='ti-r34-test')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup', default=1, type=int, metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--g_steps', default=5, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--g_life', default=10, type=int,
                    help='meta gradient: is maml or reptile')
parser.add_argument('--g_loops', default=1, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--gwp_loops', default=1, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--lr_g', default=4e-3, type=float, help='initial learning rate for generator')
parser.add_argument('--le_emb_size', default=1000, type=int, metavar='N', help='which epoch to start kd')
parser.add_argument('--bnt', default=20, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--oht', default=3.0, type=float,
                    help='momentum when fitting batchnorm statistics')

# Basic
parser.add_argument('--data_root', default='/datasets/')
parser.add_argument('--teacher', default='resnet34')
parser.add_argument('--student', default='resnet18')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'])
parser.add_argument('--lr', default=0.2, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--eta_min', default=2e-4, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--T', default=20, type=float)

parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_acc1 = 0
time_cost = 0


def main():
    args = parser.parse_args()

    args.save_dir = args.save_dir + args.log_tag
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-' + args.log_tag
    log_name = 'R%d-%s-%s-%s%s' % (args.rank, args.dataset, args.teacher, args.student, args.log_tag) \
        if args.multiprocessing_distributed else '%s-%s-%s' % (args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'
                                                                    % (args.method, args.dataset, args.teacher,
                                                                       args.student, args.log_tag))
    if args.rank <= 0:
        for k, v in datafree.utils.flatten_dict(vars(args)).items():  # print args
            args.logger.info("%s: %s" % (k, v))

    name_project = 'datafree-%s/log-%s-%s-%s%s.txt' % (
    args.method, args.dataset, args.teacher, args.student, args.log_tag)
    wandb.init(project="FastDFKD",
               name=name_project,
               tags="t1",
               config=args.__dict__)

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    print("get_dataset")
    args.num_classes = num_classes
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    # pretrain = torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset, args.teacher),
    #                                    map_location='cpu')['state_dict']
    if args.dataset != 'imagenet':
        teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth' % (args.dataset, args.teacher),
                                           map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)

    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method == 'nayer':
        le_name = "label_embedding/" + args.dataset + "_le.pickle"
        with open(le_name, "rb") as label_file:
            label_emb = pickle.load(label_file)
            label_emb = label_emb.to(args.gpu).float()

        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            generator = datafree.models.generator.NLGenerator(ngf=64, img_size=32, nc=3, nl=num_classes,
                                                             label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                             sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops)
        elif args.dataset == 'tiny_imagenet':
            generator = datafree.models.generator.NLGenerator(ngf=64, img_size=64, nc=3, nl=num_classes,
                                                             label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                             sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 64, 64), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops)
        elif args.dataset == 'imagenet':
            generator = datafree.models.generator.NLDeepGenerator(ngf=64, img_size=224, nc=3, nl=num_classes,
                                                                  label_emb=label_emb, le_emb_size=args.le_emb_size,
                                                                  sbz=args.synthesis_batch_size)
            generator = prepare_model(generator)
            synthesizer = datafree.synthesis.NAYER(teacher, student, generator,
                                                  num_classes=num_classes, img_size=(3, 244, 244), init_dataset=args.cmi_init,
                                                  save_dir=args.save_dir, device=args.gpu, transform=ori_dataset.transform,
                                                  normalizer=args.normalizer,
                                                  synthesis_batch_size=args.synthesis_batch_size,
                                                  sample_batch_size=args.batch_size,
                                                  g_steps=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, adv=args.adv,
                                                  bn=args.bn, oh=args.oh, bn_mmt=args.bn_mmt, dataset=args.dataset,
                                                  g_life=args.g_life, g_loops=args.g_loops, gwp_loops=args.gwp_loops)
    else:
        raise NotImplementedError

    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int((args.epochs - args.warmup)), eta_min=args.eta_min)

    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        tm = time.time()
        args.current_epoch = epoch

        vis_results, cost, loss_synthesizer, loss_oh = synthesizer.synthesize()  # g_steps
        # if vis_results is not None:
        #     for vis_name, vis_image in vis_results.items():
        #         str_epoch = str(epoch)
        #         if epoch < 100:
        #             str_epoch = "0" + str_epoch
        #         if epoch < 10:
        #             str_epoch = "0" + str_epoch
        #         for save_iter in range(len(vis_image)):
        #             datafree.utils.save_image_batch(vis_image[save_iter], 'checkpoints/datafree-%s/%s%s%s-%s.png'
        #                                             % (args.method, vis_name, args.log_tag, str_epoch, save_iter))
        time_cost += cost
        if epoch >= args.warmup:
            del vis_results
            # del vis_image
            # del vis_name
            train(synthesizer, [student, teacher], criterion, optimizer, args)  # kd_steps
        tm = time.time() - tm


        student.eval()
        if epoch >= args.warmup:
            eval_results = evaluator(student, device=args.gpu)
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        else:
            acc1 = acc5 = val_loss = 0

        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f} '
                         'SL:{sl:.4f} OH:{oh:.4f} Cost:{co:.4f} Time:{tm:.4f}'.format(current_epoch=args.current_epoch, acc1=acc1,
                                                                        acc5=acc5, loss=val_loss,
                                                                        lr=optimizer.param_groups[0]['lr'],
                                                                        sl=loss_synthesizer, oh=loss_oh,
                                                                        co=cost, tm=tm))
        wandb.log({"Acc1": acc1, "Acc5": acc5, "VLoss": val_loss, "lr": optimizer.param_groups[0]['lr'],
                   "SLoss": loss_synthesizer, "OHLoss": loss_oh})

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s-%s.pth' % (args.method, args.dataset, args.teacher, args.student,
                                                                  args.log_tag)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)

        if epoch >= args.warmup:
            scheduler.step()

    if args.rank <= 0:
        args.logger.info("Best: %.4f" % best_acc1)
        args.logger.info("Generation Cost: %1.3f" % (time_cost / 3600.))


# do the distillation
def train(synthesizer, model, criterion, optimizer, args):
    global time_cost
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1, 5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        if args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
            images, cost = synthesizer.sample()
            time_cost += cost
        else:
            images = synthesizer.sample()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info(
                '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1,
                        train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
        elif args.print_freq > 0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, '
                             'train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                             .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps,
                                     train_acc1=train_acc1,
                                     train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()
