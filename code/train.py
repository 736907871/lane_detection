import torch, os, datetime
import numpy as np
import cv2

from model.model import parsingNet
from data.dataloader import get_train_loader
from data.constant import tusimple_row_anchor

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    """
    results
        {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    loss_dict
        'name': ['cls_loss', 'relation_loss', 'aux_loss', 'relation_dis'],
        'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), torch.nn.CrossEntropyLoss(), ParsingRelationDis()],
        'weight': [1.0, cfg.sim_loss_w, 1.0, cfg.shp_loss_w],  # sim_loss_w: 1  shp_loss_w: 0
        'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('seg_out', 'seg_label'), ('cls_out',)]
    """
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)
        # print('----loss---name:{}---loss_cur:{}'.format(loss_dict['name'][i], loss_cur))

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux):
    net.train()
    # print('gen progress_bar... start')
    progress_bar = dist_tqdm(data_loader)
    # progress_bar = dist_tqdm(train_loader)
    # print('gen progress_bar... finished')
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
    #     print('---b_idx:{}==data_label:{}---'.format(b_idx, type(data_label)))
    #     for ele in data_label[1:]:
    #         print(ele[0])
        # print('---b_idx:{}==data_label:{}---'.format(b_idx, type(data_label)))
        # colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255))
        # for i_ in range(32):
        #     img = data_label[0][i_]
        #     img = (img*255).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
        #     lines = data_label[1][i_]
        #     label = data_label[2][i_].cpu().numpy().astype(np.uint8)
        #     label[label > 0] = 255
        #     for j in range(56):
        #         y = tusimple_row_anchor[j]
        #         for p in range(4):
        #             x = lines[j][p]
        #             color = colors[p]
        #             # print('---img.shape---', img.shape, img.dtype)
        #             # cv2.circle(img=img, center=(int(x * 8), int(y)), radius=2, color=color, thickness=2)
        #             cv2.imwrite('./tmp.jpg', img)
        #             img = cv2.imread('./tmp.jpg')
        #             cv2.circle(img, (int(x * 8), int(y)), 2, color, thickness=2)
        #     cv2.imshow('img', img)
        #     print(img.dtype, np.min(img), np.max(img))
        #     cv2.imshow('label', label)
        #     cv2.waitKey(0)

        t_data_1 = time.time()
        reset_metrics(metric_dict)  # 'op': [MultiLabelAcc(), AccTopk(cfg.griding_num, 2), AccTopk(cfg.griding_num, 3), Metric_mIoU(cfg.num_lanes+1)],
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)  # {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
        # for key in results.keys():
        #     # print('\nkey:{}---v.shape:{}'.format(key, results[key].shape))
        #     if key == 'cls_label':
        #         print(results[key])
        """
        key:cls_out---v.shape:torch.Size([32, 101, 56, 4])
        key:cls_label---v.shape:torch.Size([32, 56, 4])
        key:seg_out---v.shape:torch.Size([32, 5, 36, 100])
        key:seg_label---v.shape:torch.Size([32, 36, 100])
        """
        # print('---1---')
        """loss_dict
             'name': ['cls_loss', 'relation_loss', 'aux_loss', 'relation_dis'],
            'op': [SoftmaxFocalLoss(2), ParsingRelationLoss(), torch.nn.CrossEntropyLoss(), ParsingRelationDis()],
            'weight': [1.0, cfg.sim_loss_w, 1.0, cfg.shp_loss_w],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('seg_out', 'seg_label'), ('cls_out',)]
        """
        loss = calc_loss(loss_dict, results, logger, global_step)
        # print('---2---')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('---3---')
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)

    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(work_dir)

    print('start training ......')
    for epoch in range(resume_epoch, cfg.epoch):
        # print('\n\n\ntrain-params:{}\n\n\n'.format([net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_aux]))
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_aux)

        if epoch % 50 == 0:
            save_model(net, optimizer, epoch, work_dir, distributed)
    logger.close()
