import codecs

import yaml

from paddle3d.slim.quant import QAT
from paddle3d.utils.checkpoint import load_pretrained_model


def build_slim_model(cfg, slim_cfg_path):
    """ Slim the model and update the cfg params
    """
    with codecs.open(slim_cfg_path, 'r', 'utf-8') as f:
        slim_dic = yaml.load(f, Loader=yaml.FullLoader)

    slim_type = slim_dic['slim_type']
    if slim_type == "QAT":
        # create QAT
        quant_config = slim_dic["slim_config"]['quant_config']
        slim = QAT(quant_config=quant_config)

        # load pretrained weight befor qat
        load_pretrained_model(cfg.model, slim_dic["pretrain_weights"])

        # slim the model
        slim(cfg.model)

        # add a new filed 'slim' in cfg for export
        cfg.slim = {'slim_type': slim_type, 'slim_cls': slim}
    else:
        raise ValueError("slim method `{}` is not supported yet")

    # update finetune params
    update_dic(cfg.dic, slim_dic.get("finetune_config", {}))

    return cfg


def update_dic(dic, another_dic):
    """Recursive update dic by another_dic
    """
    for k, _ in another_dic.items():
        if (k in dic and isinstance(dic[k], dict)) and isinstance(
                another_dic[k], dict):
            update_dic(dic[k], another_dic[k])
        else:
            dic[k] = another_dic[k]
    return dic
