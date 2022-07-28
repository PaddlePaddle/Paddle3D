# paddle3d.apis.Config

  配置类方法，用于解析配置文件(yaml格式)，提取文件中指定的组件并实例化成对应的Paddle3D对象

## \_\_init\_\_

  * **参数**

    * path: 配置文件路径

    * learning_rate: 更新的学习率参数，可以不指定

    * batch_size: 更新的batch_size，可以不指定

    * iters: 更新的训练步数，可以不指定

    * epochs: 更新的训练轮次，可以不指定

    *注意：使用一个 batch 数据对模型进行一次参数更新的过程称之为一步，iters 即为训练过程中的训练步数。完整遍历一次数据对模型进行训练的过程称之为一次迭代，epochs 即为训练过程中的训练迭代次数。一个epoch包含多个iter。*

  * **异常值**

    * ValueError: 未指定配置文件路径时抛出该异常

    * FileNotFoundError: 指定文件不存在时抛出该异常

    * RuntimeError: 指定文件不是 yaml 格式时抛出该异常

## update

  更新配置类的特定超参

  * **参数**

    * learning_rate: 更新的学习率参数，可以不指定

    * batch_size: 更新的batch_size，可以不指定

    * iters: 更新的训练步数，可以不指定

    * epochs: 更新的训练轮次，可以不指定

## to_dict

  将配置类中的组件信息转成字典形式并返回

## batch_size

  单卡batch_size大小

## iters

  训练步数，与epochs互斥，当指定iters时，epochs不生效

## epochs

  训练轮次

## lr_scheduler

  调度器对象

## optimizer

  优化器对象

## model

  模型对象

## train_dataset

  训练数据集

## val_dataset

  评估数据集
