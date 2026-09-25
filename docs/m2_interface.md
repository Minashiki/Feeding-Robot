# M2 接口

控制器不调用 `mj_step`。物理只由 `FeedingScene.step_physics` 前进。

```text
controller.reset(state, tau_applied, phase)
controller.set_command(twist_world, command_time, valid_until, phase)
tau_arm, info = controller.compute(state, dt)
next_state = scene.step_physics(tau_arm)
guard.observe(next_state, info["phase_k"], info)
```

`twist_world` 是世界系六维速度，前三维平移（m/s），后三维角速度（rad/s）。命令时间用场景回合时钟，不用墙钟，也不用可能落后一步的缓存。`command_time <= now < valid_until`，有效期最长 0.1 s。NaN 或 Inf 会取消命令并进入 `ABORTED`。相同时间戳只接受内容完全一致的重传，不能改 twist、阶段或截止时间。`now == valid_until` 时命令已经过期，当前控制区间直接走停止分支。

保护力是步后样本 `S_{k+1}` 的未滤波补偿力。`compute` 只用上一份已缓存的测量，不在同一次计算里再推进滤波器。

阶段 `SELECT` / `TRANSPORT` / `WAIT_READY` 用 FREE 档，`ACQUIRE` 用取餐档，`APPROACH` / `TRANSFER` / `RETRACT` 用 MOUTH 档。更严的速度和偏差限制在切换时立即生效，刚度在 0.2 s 内插值。

reset 后力矩从给定的 `tau_applied`（场景复位后为 0）按变化率爬升，最多 1 s。这段时间不积分任务 twist。故障后下一控制周期改为 bias 加关节阻尼，不再跟踪旧参考。`ABORTED` 表示非有限状态或 MuJoCo warning，调用方应停止仿真。

`info` 中的 `ft_*` 字段是步前样本上的补偿结果。保护使用未延迟的 `compensated_wrench_tcp`。`delivered_wrench_tcp` 是低通并延迟后的观测，不能当作即时保护。
