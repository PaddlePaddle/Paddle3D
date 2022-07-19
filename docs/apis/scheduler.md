# paddle3d.apis.SchedulerABC

  调度器抽象基类，定义调度器应该实现的方法

## step

  通知调度器对象步进一次，并返回当前步的调度状态 `SchedulerStatus`

<br>

# paddle3d.apis.Scheduler

  调度器类，继承自SchedulerABC，用于决定Trainer训练过程中的调度行为，包括：

  * 是否打印日志

  * 是否保存检查点

  * 是否执行评估操作

## \_\_init\_\_

  * **参数**

      * save_interval: 保存检查点的间隔步数

      * log_interval: 打印日志的间隔步数

      * do_eval: 是否在保存检查点时启动评估

<br>

# paddle3d.apis.SchedulerStatus

  namedtuple对象，包含了调度状态

## do_eval

  是否执行评估操作

## do_log

  是否打印日志

## save_checkpoint

  是否保存检查点
