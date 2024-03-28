import argparse
import template

parser = argparse.ArgumentParser()

# Basic Training Control
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gpu_ids', default=0, type=list)
# LR Scheduler
parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
parser.add_argument('--lr_decay_steps', default=20, type=int)
parser.add_argument('--lr_decay_rate', default=0.5, type=float)
parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

# Restart Control
parser.add_argument('--load_best', action='store_true')
parser.add_argument('--load_dir', default=None, type=str)
parser.add_argument('--load_ver', default=None, type=str)
parser.add_argument('--load_v_num', default=None, type=int)

# Training Info
parser.add_argument('--exp', default='debug_test', type=str)
parser.add_argument('--dataset', default='supervision_dataset', type=str)
parser.add_argument('--data_dir', default=None, type=str)
parser.add_argument('--train_dir', default='/data/xiaoyi/Fundus/STARE/train/image_512_png', type=str)
parser.add_argument('--val_dir', default='/data/xiaoyi/Fundus/STARE/test/image_512_png', type=str)
parser.add_argument('--test_dir', default='ref/data', type=str)
parser.add_argument('--img_size', default=(512, 512), type=tuple)
parser.add_argument('--model_name', default='u_net', type=str)
parser.add_argument('--loss', default='ce', type=str)
parser.add_argument('--metrics', default=['f1', 'iou', 'auroc', 'accuracy', 'precision', 'specificity', 'recall', 'mcc',
                                          'skeletal_similarity', "calscore", 'corinf'], type=list)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--no_augment', action='store_true')
parser.add_argument('--log_dir', default='lightning_logs', type=str)

# Model Hyperparameters
parser.add_argument('--hid', default=64, type=int)
parser.add_argument('--block_num', default=8, type=int)
parser.add_argument('--in_channel', default=3, type=int)
parser.add_argument('--layer_num', default=5, type=int)

# Other
parser.add_argument('--aug_prob', default=0.5, type=float)
