# paddle3d.apis.Trainer

  训练器对象，支持在指定的数据集上训练和评估模型

## \_\_init\_\_

  * **参数**

      * model: 待训练或者评估的模型

      * iters: 更新的训练步数，可以不指定，与epochs互斥，当指定iters时，epochs不生效

      * epochs: 更新的训练轮次，可以不指定

      * optimizer: 训练所用的优化器

      * train_dataset: 训练数据集

      * val_dataset: 评估数据集，可以不指定

      * resume: 是否从检查点中恢复到上一次训练状态

      * checkpoint: 检查点参数，用于保存训练过程中的模型参数和训练状态，该参数可以是：

        *  `dict` 类型，指定构建默认 [Checkpoint](./checkpoint.md) 类对象的参数。

        * 继承了 [paddle3d.apis.CheckpointABC](./checkpoint.md) 的类对象

      * scheduler: 调度器参数，用于决定训练过程中的调度行为，该参数可以是：

        *  `dict` 类型，指定构建默认 [Scheduler](./scheduler.md) 类对象的参数。

        * 继承了 [paddle3d.apis.SchedulerABC](./scheduler.md) 的类对象

      * dataloader_fn: 数据加载器参数，用于构建数据加载器，该参数可以是：

        *  `dict` 类型，指定构建默认 Dataloader 类对象的参数，如 `batch_size` / `drop_last` / `shuffle` 。

        * 继承了 `paddle3d.apis.CheckpointABC` 的类对象

      *注意：使用一个 batch 数据对模型进行一次参数更新的过程称之为一步，iters 即为训练过程中的训练步数。完整遍历一次数据对模型进行训练的过程称之为一次迭代，epochs 即为训练过程中的训练迭代次数。一个epoch包含多个iter。*

  * **异常值**

    * RuntimeError: 当指定的Checkpoint存在数据且未设置 `resume` 时，此时数据存在被覆写的隐患，因此将抛出该异常

## train

  执行训练流程的接口

## evaluate

  执行评估流程的接口

  * **异常值**

    * RuntimeError: 初始化时如果未指定评估数据集，则抛出该异常
