# M2 接口

控制器不调用 `mj_step`。物理只由 `FeedingScene.step_physics` 前进。

M3 使用 `control_step` 作为唯一正式步进入口；底层 compute/step/observe 供内部测试，不是完整调用循环。

可执行示例：`python -m feedingrobot.scripts.m2_control_example --pose 0`（构型可选 0、1、2）。示例包含初始化、保持、运动、显式停止、故障后禁止步进以及 reset；不需要补写力矩提交或传感更新。`hold_driver=True` 只用于固定头部的 M2 静态口沿夹具，默认仍执行场景头部驱动。

```python
from feedingrobot.controllers.cartesian_impedance import control_step

controller.guard = guard
controller.reset(state, tau_applied, phase)
controller.set_command(twist_world, command_time, valid_until, phase)
next_state, info = control_step(controller, guard)
# 每个物理 tick 调用一次；命令按场景时钟续期。
# info["apply"] 为 False 时未执行物理步，调用方结束本回合。
```

该入口负责实际力矩提交、步后测量、滤波/延迟更新和保护核验。STOP/RECOVER 请求立即进入 STOPPING，清空正常命令，关闭任务及零空间力矩，以独立 `stop.damping_nm_s_per_rad` 制动。重复停止不重置计时；STOPPING/STOPPED 拒绝新运动命令，只能 reset 重启。显式停止记录 stop_request，不伪造首故障。

`control_step(controller, guard, phase="STOP")` 和 `RECOVER` 在上电接管期间也于当前控制区间生效。启动制动力矩仍从实际已施加值按原变化率爬升；真实的持续饱和等后续故障仍记录，不能因已请求停止而隐藏。

M2 的接触保护在 controller.reset 时绑定当前模型的 geom ID、名称及 body 归属。未命名机械臂 geom 由实际 link body 识别；未知几何、错误 ID/名称组合以及未授权工具—机械臂接触均拒绝。M1 原有分组接口保持不变。

非有限原始/补偿力样本不进入滤波和延迟队列，立即锁存 ABORTED；后续调用不推进物理时钟。合法滤波/延迟预热只标记 valid=false，不属于故障。estimated_kinematics 用逐子体历史速度差分计算惯性补偿，与 exact_model 分开输出；首样本 estimated_valid=false。

`twist_world` 是世界系六维速度，前三维平移（m/s），后三维角速度（rad/s）。命令时间用场景回合时钟，不用墙钟，也不用可能落后一步的缓存。`command_time <= now < valid_until`，有效期最长 0.1 s。NaN 或 Inf 会取消命令并进入 `ABORTED`。相同时间戳只接受内容完全一致的重传，不能改 twist、阶段或截止时间。`now == valid_until` 时命令已经过期，当前控制区间直接走停止分支。

保护力是步后样本 `S_{k+1}` 的未滤波补偿力。`compute` 只用上一份已缓存的测量，不在同一次计算里再推进滤波器。

阶段 `SELECT` / `TRANSPORT` / `WAIT_READY` 用 FREE 档，`ACQUIRE` 用取餐档，`APPROACH` / `TRANSFER` / `RETRACT` 用 MOUTH 档。新目标从当前刚度和阻尼开始，0.2 s 线性插到目标；切回原档位也是一次新过渡，不会跳回原档增益。过渡期间速度和偏差取起点限制与目标限制中更严的一组，过渡结束后才用目标档的完整限制。力与接触规则仍按当前阶段立即生效。

关节速度约束作用在加速度整形、受阻抑制和工作空间裁剪之后的最终候选上。`run` 和零命令都用同一个标量 beta 缩放全部六个通道，然后再积分。远离限位的零命令仍按现有加速度上限减速，不会被改成硬停止。饱和暂停、禁止推进、上电和停止不开始积分。beta 为 0 的区间不积分任务位移，下一步从已经执行的参考速度加速，而不是从投影前的历史加速。校正量单独记录。

reset 后力矩从给定的 `tau_applied`（场景复位后为 0）按变化率爬升，最多 1 s。这段时间不积分任务 twist。步后 observe 发现的故障从下一控制区间起改为 bias 加关节阻尼，不再跟踪旧参考。`rho <= 0.01` 在当次 `compute` 内锁存 `reason="singularity"`，并在同一区间进入这套受控停止：取消命令、重新锚定一次、清空任务历史、刚度和阻尼有效值为 0、`tau_task` 与 `tau_null` 为 0、`tau_raw = bias - d_stop·dq`，其中 d_stop 由独立停车标定冻结。停止优先于上电和档位过渡。相位日志保持触发时的阶段。`0.01 < rho < 0.05` 只平滑缩小命令，不紧急停止。非有限雅可比或失败的 SVD 进入 `ABORTED`，不用 `rho=0` 假装通过停止。停止计时从该故障区间的起点开始。rho 恢复后不会自动清除故障；只有 reset 清除。复位后若仍然奇异，下一次 compute 会再次锁存。已经锁存的其他故障原因保持不变。`ABORTED` 表示非有限状态或 MuJoCo warning，调用方应停止仿真。

`info` 中的 `ft_*` 字段是步后样本上的补偿结果，其 sample_tick/sample_time 对应 next_state。控制计算只使用上一份已可用缓存。保护使用未延迟的 `compensated_wrench_tcp`。`delivered_wrench_tcp` 是低通并延迟后的观测，不能当作即时保护。

步进前的 `finite=false`、已有 MuJoCo warning 或控制所需状态/原始 wrench 非有限值，在任何 `mj_forward` 前拒绝：锁存 ABORTED、取消命令、`apply=false`，当前及后续调用不推进 tick/time。`compute` 与 `control_step` 使用同一预检。未知 phase 在修改状态前抛出 `ValueError`；None 保持原义。完整验收与 full-v3 证据格式见 `docs/m2_acceptance.md`。
