import math
import paddle.optimizer.lr as lr_sched
import paddle.optimizer as optim


def build_lr_scheduler(cfg, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg['decay_list']:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg['decay_rate']
        return cur_decay

    def warm_up_lr_lbmd(cur_epoch):
        num_epoch = 5
        cur_decay = 1
        init_lr = 0.00001
        base_lr = cfg['learning_rate']
        cur_decay = init_lr + (base_lr - init_lr) * \
            (1 - math.cos(math.pi * cur_epoch / num_epoch)) / 2
        return cur_decay

    lr_scheduler = lr_sched.LambdaDecay(
        cfg['learning_rate'], lr_lbmd, last_epoch=last_epoch)
    warmup_lr_scheduler = None
    if cfg['warmup']:
        warmup_lr_scheduler = lr_sched.LambdaDecay(
            1.0, warm_up_lr_lbmd, last_epoch=last_epoch)
    return lr_scheduler, warmup_lr_scheduler


def build_optimizer(cfg_optimizer, lr_scheduler, warmup_lr_scheduler, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{
        'params': biases,
        'weight_decay': 0.0
    }, {
        'params': weights,
        'weight_decay': cfg_optimizer['weight_decay']
    }]

    if cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(
            learning_rate=lr_scheduler,
            parameters=parameters,
            weight_decay=0.0001)
        warm_up_optimizer = None
        if warmup_lr_scheduler:
            warm_up_optimizer = optim.Adam(
                learning_rate=warmup_lr_scheduler,
                parameters=parameters,
                weight_decay=0.0001)

    elif cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    else:
        raise NotImplementedError(
            "%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer, warm_up_optimizer
