# paddle3d.apis.CheckpointABC

  检查点抽象基类，定义检查点应该实现的方法

## have

  检查点中是否保存了指定tag的信息

  * **参数**

    * tag: 数据tag

## get

  获取检查点中的指定信息

  * **参数**

    * tag: 数据tag

## push

  保存一组模型参数和优化器参数到检查点中

  * **参数**

    * params_dict: 待保存的模型参数

    * opt_dict: 待保存的优化器参数

    * kwargs: 其余参数，和各个继承类实现有关

## pop

  删除检查点队列中最先保存的数据

  * **参数**

    * kwargs: 其余参数，和各个继承类实现有关

## empty

  检查点是否为空

## record

  记录一组训练信息到检查点中

  * **参数**

    * key: 训练信息标签

    * value: 训练信息内容

## meta

  检查点的元数据

## metafile

  检查点保存元数据的文件路径

## rootdir

  检查点的根路径

<br>

# paddle3d.apis.Checkpoint

  检查点类方法，支持保存模型和优化器参数，以及训练过程中的状态信息，继承自抽象基类CheckpointABC

## push

  保存一组模型参数和优化器参数到检查点中

  * **参数**

    * params_dict: 待保存的模型参数

    * opt_dict: 待保存的优化器参数

    * tag: 参数的标签，可以不填写

    * enqueue: 保存的参数是否放入队列中，队列中的参数在超过限制时会被自动删除，默认为True

    * verbose: 是否打印详细日志

## pop

  删除检查点队列中最先保存的数据

  * **参数**

    * verbose: 是否打印详细日志

## rwlock

  读写锁，用于保护多进程场景下不会读写检查点不会造成数据冲突
