import os
import math
import argparse
import random
import logging
import sys
sys.path.append('.')


import torch
import torch.multiprocessing as mp
import numpy as np

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import cv2

def load_gt_for_metrics(gt_path, lq_path, output_shape):
    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    lq_img = cv2.imread(lq_path, cv2.IMREAD_COLOR)
    if gt_img is None:
        raise FileNotFoundError('Cannot read GT image: {}'.format(gt_path))
    if lq_img is None:
        raise FileNotFoundError('Cannot read degraded image: {}'.format(lq_path))
    if gt_img.shape != lq_img.shape:
        raise ValueError('Paired source images have different shapes: LQ={}, GT={}'.format(
            lq_img.shape, gt_img.shape))
    if gt_img.shape != output_shape:
        gt_img = cv2.resize(gt_img, (output_shape[1], output_shape[0]))
    return gt_img


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/test.yml',
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    #### distributed training settings

    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')
    opt['train']['distill'] = False
    opt['train']['ewc'] = False
    if not torch.cuda.is_available():
        opt['gpu_ids'] = None


    #### mkdir and loggers
    os.makedirs(opt['path']['results_root'], exist_ok=True)
    test_img_dir = os.path.join(opt['path']['results_root'], 'test_images')
    os.makedirs(test_img_dir, exist_ok=True)
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create val dataloader
    val_opt = opt['datasets'].get('val')
    if val_opt is None:
        raise ValueError('Missing datasets.val in option file.')
    val_set = create_dataset(opt, val_opt)
    val_loader = create_dataloader(val_set, val_opt, opt, None)
    logger.info('Number of val images in [{:s}]: {:d}'.format(val_opt['name'], len(val_set)))

    #### create model

    model = create_model(opt)

    psnr_values = []
    ssim_values = []
    group_scores = {}
    logger.info('Start evaluation.')
    for idx, val_data in enumerate(val_loader, 1):
        img_name = val_data['LQ_path'][0]
        gt_name = val_data['GT_path'][0]
        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals()
        en_img = np.clip(visuals['rlt'], 0, 255).astype(np.uint8)
        gt_img = load_gt_for_metrics(gt_name, img_name, en_img.shape)

        real_name = os.path.basename(img_name)
        cv2.imwrite(os.path.join(test_img_dir, real_name), en_img)

        psnr_inst = util.calculate_psnr(en_img, gt_img)
        ssim_inst = util.calculate_ssim(en_img, gt_img)
        if not (math.isinf(psnr_inst) or math.isnan(psnr_inst)):
            psnr_values.append(psnr_inst)
            ssim_values.append(ssim_inst)
            group_name = os.path.basename(os.path.dirname(img_name))
            group_scores.setdefault(group_name, []).append(psnr_inst)

        logger.info('[{:d}/{:d}] {} PSNR: {:.4f} SSIM: {:.4f}'.format(
            idx, len(val_loader), real_name, psnr_inst, ssim_inst))

    if psnr_values:
        logger.info('# Evaluation # Average PSNR: {:.4f}, Average SSIM: {:.4f}'.format(
            sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)))
        for group_name, values in sorted(group_scores.items()):
            logger.info('# Group {} # PSNR: {:.4f} ({:d} images)'.format(
                group_name, sum(values) / len(values), len(values)))
    else:
        logger.warning('No valid PSNR values were computed.')
    logger.info('Saved restored images to: {}'.format(test_img_dir))
    logger.info('End of evaluation.')

if __name__ == '__main__':
    main()
