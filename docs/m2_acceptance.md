# M2 full-v3 验收

`validation/m2/spec.py` 定义完整 302 类矩阵，标定种子 0–2 共 2718 条轨迹，正式种子 3–8 共 5436 条轨迹。A/B/C、物理阈值及混合力关闭配置保持冻结。测试清单 `tests/m2_full_nodeids.json` 与正式反例清单分别核验，不能用测试总数代替覆盖。

## 模块与证据

- `provenance`：输入哈希、配置快照、初态、solver、执行身份与测试记录。
- `reference_checks`：解析任务目标、独立速度/约束/积分/增益重建。
- `physics_checks`：运动学、动力学、传感链与工具惯性补偿重建。
- `contact_checks`：独立接触许可、坐标转换、作用侧与关于 TCP 的净 wrench。
- `checks`：结构筛查及单例评分；`package`：包完整性、配对、趋势与唯一放行判定。
- `runner`：通过公开入口采集；`negatives`：真实副本变异；`legacy`：fixes-v6 只读适配。

每条轨迹的状态数组长度为 N+1，控制区间和步后 F/T 数组长度为 N。`t/tick` 来自真实状态；`p_ref_before/r_ref_before` 是进入控制区间前的参考，`p_ref/r_ref` 是本区间计算结果。`command_times` 保存发布时间和有效期，取消后两个值为 -1。`candidate_twist/beta/v_hist`、参考校正与约束原因支持独立重建。步前中止不增加控制区间或伪造物理样本。

接触 offsets 分隔每个步后样本，frame 的行是接触轴在世界系的方向；`contact_wrench_world` 保留世界系六维向量；完整 `contact_wrench_contact` 前三维是力、后三维是接触矩，世界向量由 frame 转置变换。该 wrench 作用于 geom2，geom1 取反。工具外部接触按所有权求和，并加上同刻施加外载；关于 TCP 的矩包括力矩臂。`tool_acceleration/arm_acceleration` 保存步后加速度，独立 Jacobian 差分及刚体惯性公式校验工具补偿。代数与同刻物理一致性使用 1e-7 容差，不扩大既有力/矩预算。

full-v2 及更早报告保留历史诊断意义，不能补写字段成为 full-v3 正式证据。

## 契约覆盖

| 契约 | 正常证据 | 独立检查与反例 |
|---|---|---|
| T01–T02 | 既有数学、动力学、零空间回归 | 解析/差分基准、动力学重建 |
| T03、T12–T14 | 公开入口、上电/运行/停车/reset | 步前异常不推进、阶段非法无副作用、首故障不覆盖 |
| T04–T05 | 保持、正反阶跃、两圈圆轨迹、姿态正弦 | 命令/积分重建及解析目标；丢动作、部分运动、伪造暂停 |
| T06–T07 | 工具静动态补偿、外力/矩、食物承载 | 原始 F/T、COM Jacobian/惯性重建、净载荷平衡 |
| T08–T11 | 三种弹簧、刚度对照、受阻释放、脉冲 | 力平衡、恢复窗口、峰值、冲量 |
| T15 | 同种子噪声/延迟配对 | 采集/交付时钟与因果重建 |
| T16 | 承载、触盘、口沿及退出 | 接触许可与六维变换；错误作用侧、力矩臂、漏接触 |
| T17、完整性 | 完整清单、回执、pytest 三阶段、正式反例 | 缺项、混合身份、遗漏哈希、布尔摘要与报告不一致 |

## 运行与恢复

所有动态运行使用 `python -m feedingrobot.scripts.m2_limited --output <资源记录目录> -- <命令>`；采用 feedingrobot 环境，单 worker，单 CPU、MemoryHigh=4 GiB、MemoryMax=6 GiB、SwapMax=0。systemd 不可用则停止，不绕开限额。

依次运行 `validate_m2 --scope full --calibrate --output <全新标定目录>`，只读核验 `verify_m2_report --scope calibration --run <标定目录> --check-current-workspace`，然后运行 `validate_m2 --scope full --calibration <标定目录> --output <全新正式目录>`。正式生成器运行新的 M1、fixes-v6、必需回归、正式轨迹及真实副本反例，再由 package 完成封包。最后使用 `verify_m2_report --scope full --run <正式目录> --check-current-workspace` 独立复核。

暂存核验与标定核验始终不放行 M3。完成记录最后原子发布；中断时保留目录和失败证据，不手改 incomplete、版本或 m3_ready。修复源码/配置后使用新 run_id 重新冻结；不拼接旧正式轨迹。当前没有已完成的 full-v3 全矩阵，保持 `m3_ready=false`。

## 初态布置补充

触盘案例先完成 M2 IK，再用同一种子的原有八个食物候选，按顺序选择第一个无非法接触的布置。原始 M1 reset 状态与最终 M2 初态分别保存；`food_placement_attempt` 记录选中序号，离线在固定最终机械臂姿态下重建候选合法性，不能跳候选或换种子。该修复处理 IK 后餐具柄与食物初始穿透，不改变模型、碰撞门槛或运动参数。

实施期间在回归的真实副本流程中已查看种子 3、4，种子 4 暴露初态穿透；后续全矩阵结果必须披露这一事实，不能将全部 3–8 种子称为从未查看过的保留集。

接触/穿透故障案例还必须在预定 tick 保留完整注入接触（ID、位置、距离、frame、零 wrench），首故障摘要不能替代该原始见证。删除见证触发 `contact_stimulus`。

`external` 记录区间开始时的目标施力；`external_body_com_before` 与 `applied_wrench_com` 分别记录施力前质心和本次实际施加的世界系质心 wrench。MuJoCo 在区间中保持质心力矩，步后核验使用步后质心换算到 TCP，不能直接沿用步前世界施力点。代数转换仍使用既定数值一致性容差。实际运动差分与工具惯性补偿的交叉检查沿用已有力/矩误差预算。
