# DATA
dataset='Tusimple'
# data_root = None
# data_root = '../data/tusimple/test_set_481MOV'
data_root = '../data/tusimple/train_set'

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

# log_path = None
log_path = '../logtest'



# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
# test_model = None
# test_model = '../log/20210329_184937_lr_4e-04_b_32/ep090.pth'
# test_model = '../log/20210406_141731_lr_4e-04_b_32/ep950.pth'
# test_work_dir = '../log/20210406_141731_lr_4e-04_b_32/ep950'
test_model = '../weights/tusimple_18.pth'
test_work_dir = '../weights/tusimple_18'



num_lanes = 4