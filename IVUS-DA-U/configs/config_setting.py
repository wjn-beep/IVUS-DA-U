from datetime import datetime
from utils import *


class setting_config:
    """
    Configuration for GH/ZZ/XW IVUS segmentation.
    """

    network = 'vmunet'

    model_config = {
        'num_classes': 4,
        'input_channels': 1,
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'ivus'
    data_path = './data/ivus'

    criterion = CeDiceLoss(num_classes=4, loss_weight=[0.4, 0.6])

    pretrained_path = './pre_trained/'
    num_classes = 4
    input_size_h = 512
    input_size_w = 512
    input_channels = 1

    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'

    batch_size = 8
    epochs = 200

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'

    train_transformer = SegmentationCompose([
        myResize(input_size_h, input_size_w),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myNormalize(datasets, train=True, data_root=data_path),
        myToTensor(),
    ])

    test_transformer = SegmentationCompose([
        myResize(input_size_h, input_size_w),
        myNormalize(datasets, train=False, data_root=data_path),
        myToTensor(),
    ])

    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    sch = 'CosineAnnealingLR'
    T_max = 100
    eta_min = 1e-5
    last_epoch = -1
