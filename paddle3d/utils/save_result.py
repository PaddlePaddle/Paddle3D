import os
import json
import yaml
import copy
from collections import OrderedDict
from paddle3d.utils.logger import logger
import paddle


def update_train_results(model_save_root,
                         pdx_model_name,
                         prefix,
                         metric_info,
                         done_flag=False,
                         last_num=5,
                         ema=False):
    if paddle.distributed.get_rank() != 0:
        return
    assert last_num >= 1
    train_results_path = os.path.join(model_save_root, "train_result.json")
    save_model_tag = ["pdparams", "pdopt"]
    save_inference_tag = [
        "inference_config", "pdmodel", "pdiparams", "pdiparams.info"
    ]
    if ema:
        save_model_tag.append("pdema")
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = pdx_model_name
        train_results["label_dict"] = ""
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if prefix == "best_model":
        train_results["models"]["best"]["mAP"] = metric_info["mAP"]
        train_results["models"]["best"]["NDS"] = metric_info["NDS"]

        for tag in save_model_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, f"model.{tag}")
        for tag in save_inference_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"].copy()
        train_results["models"][f"last_{1}"]["mAP"] = metric_info["mAP"]
        train_results["models"][f"last_{1}"]["NDS"] = metric_info["NDS"]
        for tag in save_model_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, f"model.{tag}")
        for tag in save_inference_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, dict_data.items())


def setup_orderdict():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def dump_infer_config(pdx_cfg, path):
    setup_orderdict()
    infer_cfg = OrderedDict()
    inference_config = pdx_cfg['inference_config']
    pdx_model_name = pdx_cfg['pdx_model_name']
    infer_cfg["Global"] = {"model_name": pdx_model_name}
    if "transforms" in inference_config:
        transforms = inference_config["transforms"]
    else:
        logger.error("This config does not support dump transform config!")

    # TODO: Configuration required config for high-performance inference.
    transforms_pipelines = []
    for func in transforms:
        ordered_func = OrderedDict()
        ordered_func['type'] = func['type']
        for k in func:
            if k == 'type':
                continue
            ordered_func[k] = func[k]
        transforms_pipelines.append(ordered_func)

    infer_cfg["PreProcess"] = {
        "transform_ops":
        [infer_preprocess for infer_preprocess in transforms_pipelines]
    }

    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(infer_cfg, f)
    logger.info("Export inference config file to {}".format(os.path.join(path)))
