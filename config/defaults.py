from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "weights/outside15k-resnet50-outsideNet"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.root_dataset = "./data/"
_C.DATA.list_train = "./data/training.odgt"
_C.DATA.list_val = "./data/validation.odgt"
_C.DATA.list_stats = "./data/training.odgt"
_C.DATA.num_class = 24
# multi_scale train/test, size of short edge (int or tuple)
_C.DATA.img_sizes = (300, 400, 500, 700, 800)
# maximum input image size of long edge
_C.DATA.img_max_size = 1000
# maxmimum downsampling rate of the network
_C.DATA.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATA.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATA.random_flip = True
# randomly crop images if the size exceeds the maximum size
_C.DATA.random_crop = True
# info file
_C.DATA.class_info = "./data/outside15k.json"
# dump information to output file
_C.DATA.dump_model = ""


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50"
# architecture of net_decoder
_C.MODEL.arch_decoder = "outsideNet"
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# utilization of spatial mask
_C.MODEL.spatial_mask = False
# align corners during upsampling
_C.MODEL.align_corners = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 1
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continuing from checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer
_C.TRAIN.optim = "SGD"
# initial learning rate for encoder
_C.TRAIN.lr_encoder = 0.02
# initial learning rate for decoder
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16
# frequency to display
_C.TRAIN.disp_iter = 20
# use validation during training
_C.TRAIN.eval = False
# step size for validation calculation
_C.TRAIN.eval_step = 5
# best score from previous Training
_C.TRAIN.best_score = 0
# optimizer data from previous training
_C.TRAIN.optim_data = ""

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# currently only supports 1
_C.EVAL.batch_size = 1
# output visualization during validation
_C.EVAL.visualize = False
# the checkpoint to evaluate on
_C.EVAL.checkpoint = "epoch_20.pth"
# use multi scale for evaluation
_C.EVAL.multi_scale = False

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
# use multi scale for testing
_C.TEST.multi_scale = False
