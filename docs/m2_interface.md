# M2 接口

控制器不调用 `mj_step`。物理只由 `FeedingScene.step_physics` 前进。

```text
controller.reset(state, tau_applied, phase)
controller.set_command(twist_world, command_time, valid_until, phase)
tau_arm, info = controller.compute(state, dt)
next_state = scene.step_physics(tau_arm)
guard.observe(next_state, info["phase_k"], info)
```

`twist_world` 是世界系六维速度，前三维平移（m/s），后三维角速度（rad/s）。`command_time` 不能晚于当前回合时间，未知阶段会被拒绝。命令默认每 50 ms 发布一次，有效期 100 ms。过期会锁存 `command_expired`，重复命令不会清除已有故障。

阶段 `SELECT` / `TRANSPORT` / `WAIT_READY` 用 FREE 档，`ACQUIRE` 用取餐档，`APPROACH` / `TRANSFER` / `RETRACT` 用 MOUTH 档。更严的速度和偏差限制在切换时立即生效，刚度在 0.2 s 内插值。

reset 后力矩从给定的 `tau_applied`（场景复位后为 0）按变化率爬升，最多 1 s。这段时间不积分任务 twist。故障后下一控制周期改为 bias 加关节阻尼，不再跟踪旧参考。`ABORTED` 表示非有限状态或 MuJoCo warning，调用方应停止仿真。

`info` 中的 `ft_*` 字段是步前样本上的补偿结果。保护使用未延迟的 `compensated_wrench_tcp`。`delivered_wrench_tcp` 是低通并延迟后的观测，不能当作即时保护。
