# M2 第五轮修复方案：阶段驱动的增益验收与持续卸载验收

编写日期：2026-09-25。关联：[M2FixPlanV4.md](M2FixPlanV4.md)、[M2Plan.md](M2Plan.md)、[SimModelPlan.md](SimModelPlan.md)。

**目标：关闭 V4 审查确认的两处判定漏洞。保留已经达标的阻抗控制器、接触减速、慢卸载和快速释放场景，补齐阶段—增益对应关系以及释放后持续零载荷、零命令的独立验收。**

本文是待执行任务书。本次仅编写方案，不表示 V5 代码或验收已经完成。V1–V4 的有效约束继续适用；本轮不扩展完整 M2 的任务范围，不降低物理门槛，完整 M2 未完成时 `m3_ready=false`。

## 1. 当前基线与事实

### 1.1 项目和报告身份

- GitHub：[Minashiki/Feeding-Robot](https://github.com/Minashiki/Feeding-Robot)。地址来自本地 `origin`，不表示远端已同步当前工作区。
- 编写时 HEAD：`450f65a8fce245c68aca52b2581c8f6788225560`，提交说明 `M2 v1.2`。当前存在未提交修改及未跟踪的 V3/V4 校验模块，实施前保存工作区差异、完整输入集合及哈希，不能只绑定 HEAD。
- 本轮审查起点：[m2-fix-v4-r3/report.json](outputs/m2/acceptance/m2-fix-v4-r3/report.json)。保留原目录及其真实历史结论，不直接修改 schema 或 passed 标志升级成 V5。
- 对应 M1 回归：[m2-fix-v4-regression-r2/report.json](outputs/m1/acceptance/m2-fix-v4-regression-r2/report.json)。以 M2 包中的实际绑定为准，不能误用此前中断目录。
- 固定 M1 物理基线：[m1-fix-v3b/report.json](outputs/m1/acceptance/m1-fix-v3b/report.json)。

### 1.2 上轮已完成的独立测试

| 项目 | 实测结果 |
|---|---|
| M2 回归 | 187 passed，78.73 s |
| M1 完整回归 | 28 passed，84.71 s |
| 新 M1 报告及工作区、历史 v3b | 均通过 |
| r3 默认 `fixes-v4` 完整 CLI | 退出 0，无临时目录错误 |
| 默认 full 范围 | 退出 1，列出完整 M2 所有剩余缺项 |
| 物理记录 | 54 份，未发现非有限数值 |
| 收敛比较 | 630 条通过 |
| 快速释放九例最差线速度 | 0.01335474 m/s，低于 0.03 m/s |
| 同窗口最差额外 +X 位移 | 0.001482214 m，低于 0.005 m |

正式轨迹中的实际增益切换与释放后零外力也已单独核对，未发现本轮两种反例对应的实际异常。**本轮修复对象是验收器不能拒绝错误证据，不据此认定正式物理试验失败或要求重写控制器。**

## 2. 已确认漏洞与完成标准

### 2.1 V5-01：以自报档位验证自报增益

在正式包副本的 `cases/gear-s0-A.npz` 中保留 `phase` 的 ACQUIRE 切换，修改：

```text
K[:] = [300, 300, 300, 8, 8, 8]
D[:] = [49, 49, 49, 0.8, 0.8, 0.8]
transition_active[:] = false
active_gear[:] = FREE
target_gear[:] = FREE
```

重新计算该文件的证据哈希后，核心检查通过；**完整 `verify_m2_report --scope fixes-v4 --check-current-workspace` 也退出 0，fixes_passed=true、reasons=[]。**

根因：`controllers/v4fields.py::_check_gain_path()` 根据轨迹提供的 active/target/trans 选择期望值，没有从阶段变化重建应发生的档位过渡。物理 gear 案例虽然检查出现 ACQUIRE，却没有强制该阶段与实际 K/D 对应，因此“阶段切了、增益没切”仍能认证。

完成标准：正常物理案例的目标档位由固定阶段映射唯一决定；增益轨迹、过渡标志、active/target 和完成事件都必须与独立重建一致。反例即使修改多个相关自报字段并更新哈希，也必须被具体阶段/增益规则拒绝。

### 2.2 V5-02：只找到首次零载荷，未验证后续持续卸载

在 `cases/release_fast-s0-A.npz` 副本中，将 t∈[0.5,0.6) s 的 `external_wrench[:,0]` 改为 −3 N，更新证据哈希后，核心检查仍返回 `fixes_passed=true、evidence_valid=true、reasons=[]`。本反例上轮已验证到核心入口，**未声称已完成该副本的完整 CLI 测试**；本轮两层都必须覆盖。

根因：`controllers/v3spec.py::_score_fast_release()` 查找首次外力归零，随后 `_release_window()` 只核对响应及部分命令，没有约束释放后的载荷完整时序；当前寻找零命令、零载荷也只使用前三个线性分量。

完成标准：释放事件按六维命令及六维施加载荷重建，线性和旋转量分别按单位判断；归零后至试验结束持续零载荷，撤载请求后至试验结束持续零命令。一次重新加载、纯外力矩或纯角速度命令均不能通过。

## 3. 修改边界与实施顺序

### 3.1 不变项

1. 控制律、增益配置、正常速度上限、力矩幅值及 **2000 N·m/s** 变化率保持不变；不为验收器补漏重新标定控制器。
2. 保持 54 个物理案例：pause、gear、wrench、takeover、release、release_fast × seed 0/1/2 × A/B/C。单元/状态机和报告反例另外计数。
3. gear 的 2.0 s 时长、0.008 m/s 接近与 0.002 m/s 接触慢压；release/release_fast 的 2.5 s 时长和 −X 3 N 刺激保持。
4. 原 1.2 s 慢卸载仍是独立案例，不替代一个物理步内解除载荷的快速释放。
5. MuJoCo 3.14.0、NumPy 2.2.6、pytest 9.1.1 不变。普通复制、独占临时目录、源包只读、两阶段执行继续保留。
6. M1 主资产、配置、事件、20 ms 门槛不变；诊断高速参数不进入正常控制。勺—餐盘接触只按既定阶段与 geom 白名单允许；混合力控保持 `disabled`。
7. 不新增完整 T08–T17 试验，不以本轮通过替代正式种子 3–8 和完整 M2 矩阵。

### 3.2 实施步骤

| 步骤 | 实现内容 | 出口 |
|---|---|---|
| S0 | 记录输入身份，固化两个原始反例 | 未修版分别复现完整 CLI 误通过、核心误通过 |
| S1 | 阶段驱动的预期增益重建与字段交叉检查 | 假切档、错误目标、提前/延迟完成等反例拒绝 |
| S2 | 完整六维释放事件和持续零载荷/命令检查 | 重新加载、角向残留、窗口末端注入等反例拒绝 |
| S3 | 小型解析测试、现有实际轨迹正例、完整 CLI 反例 | 正例通过，错误因目标规则拒绝 |
| S4 | 冻结源码/规格，新 M1/M2 正式执行和独立复核 | V5 通过有完整证据；full 按真实缺项拒绝 |

修改主要集中在：

- `src/feedingrobot/controllers/v4fields.py`：增益轨迹独立重建、phase/gear/transition 对应关系。
- `src/feedingrobot/controllers/v3spec.py`：gear 完成事件、释放时序和持续条件、反例注册与错误码。
- `src/feedingrobot/controllers/acceptance.py`、两个 M2 CLI：V5 版本和执行范围。
- 现有报告/状态机测试及必要的 `tests/test_m2_v5.py`；记录缺少阶段初态时，仅补该字段和绑定。
- 标定/接口文档：说明新判定语义，记录本轮实测。

继续复用现有模块，不复制整个验收器到新版本文件。不建设通用回放框架；本轮的两个纯检查函数和必要调用接入即可完成目标。

## 4. V5-01 实现：由阶段独立重建增益路径

### 4.1 可信输入与输出的区别

重建器的输入是：已验证的 case 规格、初始化阶段、逐区间 phase、真实 dt、冻结增益及 0.2 s 过渡时间。K/D、active_gear、target_gear、transition_active 是**待核对的输出**，不得用它们决定是否需要过渡或何时完成。

固定阶段映射：

| phase | 目标档位 |
|---|---|
| SELECT、TRANSPORT、WAIT_READY | FREE |
| ACQUIRE | ACQUIRE |
| APPROACH、TRANSFER、RETRACT | MOUTH |
| RECOVER、STOP | STOP |

映射可复用当前 `guard.GEAR_OF` 的定义并绑定输入哈希，但校验不能调用控制器 `_update_gains()` 生成“独立期望”。未知阶段直接失败。正常 54 例仍不允许进入故障停止；STOP/RECOVER 与已有故障注入采用原专用规格，不借本轮要求修改停止控制律。

正常物理案例的初始化阶段来自固定案例规格，目前为 TRANSPORT；据此设置初始 FREE 的 K/D 和已完成档位。不能直接取轨迹第一行的 active_gear 为初始真值。如果以后需要中途截取轨迹，必须另定义受绑定的初始过渡状态，本轮不支持任意截段认证。

### 4.2 与实际控制区间对齐

`phase[k]` 表示第 k 控制区间采用的阶段；本区间执行 `_update_gains(dt[k])` 后产生 `K[k]/D[k]`。因此目标变化的第一行已经推进一个 dt，不是 alpha=0 的切换前行。

以 FREE→ACQUIRE 为例：

- dt=1 ms：第一行平移 Kp=299.25、Dp=48.93；到 0.2 s 完成时 Kp=150、Dp=35。
- dt=0.5 ms：第一行 Kp=299.625、Dp=48.965；同样按 0.2 s 完成。
- 完成的那一行必须是目标终值，transition_active=false，active_gear=target_gear=ACQUIRE。

数学时间由真实 dt 累加；最终一步吸收浮点误差，使用既定 1e-12 s 量级时间容差，不允许以一个完整物理步作为任意早完/晚完的容差。

### 4.3 简单的独立状态机

重建器内部维护 `expected_K/D、completed_gear、desired_gear、transition_start_K/D、elapsed、in_transition`，逐区间处理：

```text
初始化：由 case 的初始化阶段取得固定初始增益，过渡关闭。
每个区间 k：
  desired = 固定映射(phase[k])
  若 desired 与上一个期望目标不同：
    起点 = 重建器此前的期望 K/D
    elapsed = 0，启动新过渡
  若目标相同：保持原起点与 elapsed，不重新计时
  若处于过渡：
    elapsed += dt[k]
    alpha = min(elapsed / 0.2, 1)
    expected_K/D = (1-alpha)*起点 + alpha*固定目标增益
    达到终点：吸附到目标增益，完成档位改为 desired，关闭过渡
  否则：expected_K/D = 固定目标增益
  比较该区间全部增益、档位、过渡标志与重建结果
```

重建状态不因看到某行错误日志而重新锚定，也不把误差逐步吸收到新的期望中。实际前一行在已通过验证时应等于该期望；新过渡起点因此与“当前实际增益”一致。

中途返回原档位也启动新的过渡：FREE→MOUTH 运行 50 ms 后 Kp=250，切回 FREE 时下一步分别为 250.25（1 ms）、250.125（0.5 ms）。中途转第三档按同样规则处理。连续 transition_active=true 的区段可能包含多次目标改变，**不能只按布尔连续段分组后一直使用段首 target**。

阶段改变但映射档位相同，例如 TRANSPORT→WAIT_READY，不重启过渡。重复同阶段命令同样不重启。这里的 completed_gear 表示上一次完成的档位，不等同于当前阶段目标；过渡未完成时不能强制 active_gear 提前等于目标。

### 4.4 必须核对的内容

1. 每行 `target_gear == gear_of(phase)`；不要以自报 target 选择期望增益。
2. K/D 六通道分别匹配重建值，沿用 `atol=1e-9、rtol=1e-7`。不是仅检查非负、有限、最大跳变量。
3. active_gear 与重建的已完成档位一致；transition_active 与重建过渡状态一致。轨迹包含 elapsed/start 等诊断字段时，也核对而非直接相信。
4. 过渡结束后保持固定终值，不能最后一行才补上正确增益。过渡提前结束、无限保持 active、重启计时或目标中途错误都失败。
5. 从重建状态计算严格速度/参考偏差限制，或先证明轨迹档位状态逐行一致再复用它；不能让错误自报 FREE 顺便放宽 ACQUIRE 的限制。
6. gear 物理案例必须出现规定阶段切换及其真实增益完成事件，并覆盖结束后的有效样本。phase 被整体改成 TRANSPORT 时，因缺少必需切换失败，不能退化为另一个合法静态案例。

错误码建议区分 `phase_gear_mismatch、gain_path_mismatch、transition_state_mismatch、transition_incomplete`，记录 case、区间/tick、字段、期望与实际值。不只输出“比较不同”或泛化成 NaN。

### 4.5 与证据、收敛的接入

该检查必须由 `_check_physical()` 的核心路径调用，生成器和独立校验都执行。物理 gear 的完成事件由重建结果与实际 K/D 一致性确定；不能仅用 phase=ACQUIRE 且 trans=false 推定完成。

保留全部原 A/B、A/C 指标及 contact/switch 事件，可增加明确的 `gain_complete` 事件并按原 20 ms 门槛比较。比较条数由固定规格展开，不把上一轮的 630 当成硬编码成功条件。

### 4.6 检查函数的边界契约

扩展现有 `_check_gain_path` 的输入，显式传入 `phase` 和已验证的 case 初始阶段。dt 取固定 A/B/C 规格，并先证明记录时间轴与之相符；不能让被修改的时间轴决定更宽的过渡容差。当前案例等步长即可，不新增可变步长控制功能。

- 调用前：全部区间数组长度为 N，状态时间轴长度为 N+1；枚举、类型、形状、有限性和时间单调性已通过。N=0 或必要字段缺失直接失败，不使用默认 FREE 补齐。
- 调用中：只维护本次调用的重建状态，不修改 NPZ 数组，不读取全局控制器状态，不复用上一案例的 completed_gear。发现矛盾可以停止该案例的增益评分，但整个案例必须失败。
- 调用后：返回结构化错误及确有需要的重建完成事件。可以在函数内逐行比较，无须额外持久化一整份期望轨迹；错误中的期望值足以定位问题。
- 正常 gear 案例必须在结束前完成指定 ACQUIRE 过渡，并保留完整的完成后观测区间。解析测试可以在过渡中结束，以验证前缀是否正确；这种局部测试通过不能代替完整物理案例通过。
- 正常路径不应套用 STOP 的线性插值：当前 STOP 是立即零增益，仍由既有停止专用规则验收。若共享函数接收 STOP，必须按该明确规则处理，不能根据正常档位伪造一个 0.2 s STOP 过渡。

## 5. V5-02 实现：持续零载荷与持续零命令

### 5.1 六维量分别检查，不能混用单位

`external_wrench[k]` 是实际施加在区间 `[t_k,t_k+1)` 的测试外载，不是腕部补偿后的测量值。零载荷规则作用于这个施加量；不能要求接触力、食物重力或补偿残余也变成零。

固定数值容差如下，写在代码规格中，不从 NPZ 或元数据读取：

| 条件 | 判定 |
|---|---|
| 线外力为零 | `norm(external_wrench[k,:3]) ≤ 1e-6 N` |
| 外力矩为零 | `norm(external_wrench[k,3:]) ≤ 1e-6 N·m` |
| 线速度命令为零 | `norm(command[k,:3]) ≤ 1e-4 m/s` |
| 角速度命令为零 | `norm(command[k,3:]) ≤ 1e-4 rad/s` |

力和矩、线速度和角速度各自比较，不对六维混合单位向量使用同一个范数。命令零容差沿用当前线性检查的量级并补齐角向检查；这不是对实际 TCP 响应速度的新门槛。必需字段的 shape/dtype/finite 检查先执行，NaN 不得进入零判断。

### 5.2 事件重建与持久条件

延续 V4 的 blocked_first 独立重建；在其后查找实际下发的六维零命令确定 unload_start。快速案例要求它对应受阻识别后的下一个可用控制区间，不允许先等待机器人静止再移后评分起点。

从 unload_start 起，查找首个六维零外载区间，确定 force_zero。快速案例仍要求：

```text
0 ≤ t_force_zero - t_unload_start ≤ dt + 1e-9 s
```

增加两条完整时序约束：

```text
command[unload_k : N] 逐区间满足线/角命令均为零
external_wrench[force_zero_k : N] 逐区间满足线力/力矩均为零
```

检查一直到最后一个控制区间，不只到 0.5 s 峰值窗口或 1.0 s 稳定窗口。当前 release/release_fast 的剩余时段没有重新加载或推进计划，因此尾部数据也是验收范围。边界采用整数区间索引；force_zero 本行必须包含在检查中。不能用末尾状态帧代替最后一个施加输入区间。

**先确定首次事件，再验证其后的持续条件；发现重新加载即失败。** 不可向后寻找“最后一次归零”或“之后一直为零的更晚时刻”来跳过非法脉冲。也不能修改 unload_start 使脉冲落在评分窗口外。

### 5.3 慢卸载与快速释放共享后置条件

建议新增一个小函数，例如 `check_release_tail(command, external, unload_k, zero_k, ...)`，由慢、快释放两条路径共同调用，避免只修快速分支而保留同类遗漏。

- 快速分支：一个物理步内完成归零，其后持续为零。
- 慢分支：保留原 1.2 s ramp；从 unload_start 起零命令，从 ramp 完成的 force_zero 起零外力。不能把 ramp 期间的合法非零外力误判为重新加载。
- 其他族按各自刺激规格处理；例如 wrench 案例的已知载荷不是本函数的输入范围。

首次受阻、撤载前 −X 3 N 的真实前提、原方向、完整时长检查继续执行。新增判定不替代原受阻重建，也不能只看元数据中的 `unload=fast`。

### 5.4 评分与错误输出

只有刺激时序和持续条件正确，才能把速度、位移、稳定性视为该释放案例的有效成绩。失败轨迹可以输出诊断指标，但案例 passed 必须为 false，不能因响应很小而忽略重新加载。

原门槛不变：从 unload_start 和 force_zero 各自起算，0.5 s 内速度≤0.03 m/s、沿 +X 额外位移≤5 mm；稳定窗口线速度≤0.005 m/s、角速度≤0.05 rad/s；保留完整窗口、差分速度交叉检查和无旧任务追补要求。

错误码建议：`external_reloaded、external_torque_reloaded、command_resumed、angular_command_resumed`。输出首次违规区间/tick、物理时间、分量/范数、容差及对应释放事件。若载荷或命令不满足归零事件本身，则输出 `event_mismatch`/`unload_duration`；不要把非法重新加载说成普通响应速度不达标。

### 5.5 检查函数的边界契约

`check_release_tail` 接收已经通过结构检查的 `(N,6)` command/external、长度 N+1 的状态时间轴以及重建的两个整数索引。要求 `0 ≤ unload_k ≤ zero_k < N`；索引缺失、越界或尾部为空直接失败，不能依靠 `all(empty)==true` 放行。

实现可以先计算四个长度 N 的范数数组，再分别检查两个闭起点、开终点的切片：

```text
linear_command_norm[unload_k:N] <= command_linear_zero
angular_command_norm[unload_k:N] <= command_angular_zero
external_force_norm[zero_k:N]   <= external_force_zero
external_torque_norm[zero_k:N]  <= external_torque_zero
```

每种错误至少保留第一处违规，不用均值、累计冲量或低通后的值替代逐步判断。事件重建与尾部检查必须使用同一组固定零容差，避免首次事件按 `<`、后续按另一阈值判断。所有时间从已验证的 `t_state[index]` 得出。

尾部合法不代表响应窗口足够：另行要求 unload 和 force_zero 各自的 1.0 s 观察窗口完整。慢卸载在 ramp 结束后也必须满足既有窗口要求；若更严格的共同检查暴露现有记录不足，明确报 `sample_count`，先按原规格确认合法时序，不缩短稳定窗口或改用最终单点。

## 6. 必须新增的验收测试

### 6.1 增益解析测试与实际轨迹反例

解析测试直接验证重建器，不需要运行全物理。覆盖 A/B 两个 dt、三正常档位有向切换、50/150 ms 中断后返回源档或转第三档、同档位不同 phase、重复阶段命令、终点前后边界。

至少包含以下正例与反例：

| 输入 | 期望 |
|---|---|
| 正常 FREE→ACQUIRE | 首步和终值符合第 4.2 节，逐通道通过 |
| FREE→MOUTH 50 ms 后返回 FREE | 下一步 250.25/250.125，完成时间正确 |
| 中途变更目标而 trans 连续为 true | 依据 phase 重启一次新过渡，不能沿用第一段目标 |
| TRANSPORT→WAIT_READY、重复同阶段命令 | 不重启同档位过渡 |
| 原始反例：phase 到 ACQUIRE，但 K/D/active/target/trans 全程伪装 FREE | 具体 phase/gear 或增益路径错误，核心和完整 CLI 均拒绝 |
| 仅修改 target 与 phase 不一致 | `phase_gear_mismatch` |
| K/D 值看似合法，但提前完成、延迟完成、一直不完成或末尾才补终值 | 对应过渡或路径错误 |
| 全部相关字段都改为 FREE，连 phase 也改为 TRANSPORT | 因 gear 案例必需阶段/事件缺失而拒绝 |
| 错误 active/trans 试图放宽 ACQUIRE 限制 | 拒绝，不能通过自报档位扩大门槛 |
| 任一必需增益数组为空、缺行、NaN/Inf | 由既有结构/有限性规则拒绝 |

如果旧 F5 单元证据使用四通道 `[Kp,Kr,Dp,Dr]`，需与重建器的六维 K/D 有明确映射并验证预期长度；不能把空 diff 或单点数组当成完成整段过渡。

### 6.2 释放持续条件测试

| 反例/正例 | 必须结果 |
|---|---|
| 原始反例：release_fast-s0-A 在 [0.5,0.6) s 再施加 −X 3 N | `external_reloaded`，更新哈希后核心和完整 CLI 都拒绝 |
| 归零后的一个单独物理步重新加载 | 拒绝，不能依靠平均值或低通滤波消失 |
| 纯外力矩脉冲、线外力仍为零 | `external_torque_reloaded` |
| 仅 +Y/+Z 外力，X 分量为零 | 外力范数检查拒绝 |
| 重新加载在 0.5 s 响应窗后、1.0 s 窗后或最后一个控制区间 | 均拒绝，检查覆盖至 N−1 |
| 零线命令但非零角速度命令 | `angular_command_resumed` |
| 恢复线速度命令，但响应尚未超限 | `command_resumed`，不能等待速度违规 |
| 修改事件时间将脉冲“移出窗口” | 按输入重建事件，拒绝不一致，不采用编辑后的时间 |
| 慢卸载合法 ramp | ramp 期间不误报；归零后重新加载仍拒绝 |
| 零容差边界内、恰等于边界、明确超出边界 | 前两者通过，后者失败；力/矩和线/角分别测试 |

测试先验证完整合法包通过，再复制并单项破坏；数值/语义反例更新对应 SHA-256，避免仅靠旧哈希报错。完整 CLI 必须包含目标错误码，即使同时报告 summary_mismatch，也不能把后者当成目标漏洞已经修复的唯一证据。

### 6.3 执行方式与成本控制

保留独占 TemporaryDirectory、普通 copy2、一次一个副本、源包前后哈希及并发核验约束。小型解析测试用于快速迭代；正式反例基于一次完整生成的正例包，不为每个反例重新仿真。

核心验收入口只检查证据，最终 CLI 检查核心、两阶段执行记录和摘要；反例副本调用核心入口，不能递归重跑自身。两个原始反例必须各有完整 CLI 集成覆盖，证明最后放行路径同样拒绝；其余变体可由共享规则的直接测试加核心包测试覆盖。

固定反例 ID、错误码和排序。保留 V4 的 NaN 增益、暂停 5 m/s、100 rad、ABORTED、文件/哈希/时间/形状/旋转、慢 ramp 冒充快释放等已闭合反例。不得为减少运行时间删除旧必需节点或跳过正例确认。

### 6.4 两个原始漏洞的可重复操作流程

以下流程写入自动化测试或独立复现脚本。旧版本复现使用 r3 及与其绑定的源码；修复后验证使用新生成的 V5 合法包及其绑定源码。不要用旧 V4 包遇到 V5 schema 拒绝来证明语义漏洞已修好，也不要因源码已经变化而把工作区哈希失败当成目标反例成功。

1. 对合法源包先运行同版本的核心入口和完整 CLI，确认均通过。保存源包全部文件的 SHA-256 集合。
2. 在独占临时目录中普通复制完整包。每个反例从合法源包重新复制，不能在另一个失败副本上叠加。
3. 用 `np.load(path, allow_pickle=False)` 物化全部原数组并关闭文件。仅修改指定数组，再以 NPZ 格式写回副本原路径；保留其他字段、原 dtype 和 shape，不把 NPZ 替换成 JSON。
4. **反例 G**：修改 `cases/gear-s0-A.npz` 的 K/D、active/target/trans，数值完全使用第 2.1 节；断言原 phase 中确有 ACQUIRE，且修改后 phase 和其他数组逐项未变。
5. **反例 R**：修改另一个副本的 `cases/release_fast-s0-A.npz`。用 `t_state[:-1]` 选取 `[0.5,0.6)` s 区间，将 `external_wrench[mask,0]` 置为 −3.0；先断言 mask 非空且位于首次归零之后。其余五个分量和其他数组不变。按物理时间选取，避免将 A 的行号直接套到 B。
6. 只更新副本 `evidence_manifest.json` 中该案例的 SHA-256；先断言对应路径恰有一条。保留原 passed 标志、派生摘要、运行清单和其它清单成员，测试“保留通过摘要仍不能认证”。不得重写校验器或门槛。
7. 分别调用核心 `evaluate_evidence` 和 `python -m feedingrobot.scripts.verify_m2_report --scope fixes-v5 --run <副本> --check-current-workspace`。保存完整 stdout、stderr、退出码和核心结构化原因。V5 的 G 必须有阶段/增益错误，R 必须有 `external_reloaded`；CLI 均退出 1。若同时出现摘要不一致，可保留，但不能仅凭摘要或哈希错误验收。
8. 重新核对源包全部哈希不变，清理各自临时目录。失败记录保留在测试日志或新正式包的反例结果中，不写回源包。

| 对象 | 修复后核心结果 | 修复后完整 CLI | 不能替代的证据 |
|---|---|---|---|
| 合法 V5 包 | fixes_passed=true | fixes-v5 退出 0 | 只跑解析测试 |
| 反例 G，证据哈希正确 | fixes_passed=false，具体阶段/增益原因 | 退出 1，含目标原因 | schema 不符、旧工作区哈希失败 |
| 反例 R，证据哈希正确 | fixes_passed=false，external_reloaded | 退出 1，含目标原因 | 仅 summary_mismatch 或 file_hash |
| 合法 V5 包的 full 请求 | 完整范围不通过 | 退出 1，完整列出缺项 | 将 fixes-v5 通过改成 m3_ready=true |

增加一个“合法包 → G/R → 再次合法包”的顺序测试，确认检查器没有跨调用状态污染。并发验证沿用 V4 测试，不允许两个副本共用可写文件。

## 7. 报告、版本与失败语义

新增 `schema_version=m2-fix-v5`、报告 scope=`fixes_v5`、CLI `--scope fixes-v5`。这些是待实现接口，当前脚本只支持到 fixes-v4。复用现有验收模块，不因文件名含 v3/v4 而复制整个实现。

旧 V4 包只能作为历史/诊断输入，不能被 fixes-v5 降级认证。固定 schema 的案例规格必须包含阶段驱动增益和持续释放检查，不能由 manifest 自行省略。

- `evidence_valid` 表示记录结构、身份和派生摘要可核对；语义失败仍必须 `fixes_passed=false`。字段矛盾和无效事件需要明确分类，不从人类说明字符串猜测类别。
- `tests_passed` 要求必需测试节点及两阶段执行均真实通过；不能只看 pytest 返回码或测试文件名。
- `fixes_passed` 要求全部固定指标、两个新规则及旧回归通过。
- `scope_passed` 决定请求范围的退出码；本轮 `m3_ready=false、hybrid_force_status=disabled`。
- full 模式继续列出 T08、T09、T10 物理挡面、T11、T15、T16、完整 T17、种子 3–8 和完整矩阵等缺项，退出 1；这不替代 fixes-v5 的正反例验收。

comparisons 中可增加增益完成事件、释放尾部最大力/矩及命令残余摘要，所有新增量由原始数组复算。若不增加摘要字段，错误结果仍需携带足够信息定位至具体区间。

CLI 接入不能只给 argparse 增加 `fixes-v5`：当前 `verify_m2_report.main()` 的成功返回分支限定为 fixes-v4，必须同步接通 V5 的范围判定和成功返回。覆盖合法 fixes-v5 退出 0、两个非法副本退出 1、full 缺项退出 1，以及旧 scope 不被错误升级四条集成路径。若打印原因时将结构化对象转成字符串，必须保留可识别错误码，保证 CLI 也能核对拒绝原因。

`validate_m2` 同样存在只允许 fixes-v4 的入口分支；同步更新生成范围、schema/scope 常量、必需测试与反例集合、默认 M1 回归绑定。正式命令仍显式传入新 M1 路径，防止默认值遗留造成新包绑定旧回归。

生成器在真实记录、反例结果、输入前后哈希及 M1 绑定齐备后写最终报告；遇到失败保留证据和原因，不通过重写摘要修复结果。固定源码后执行新 M1/M2；若共同输入再变更，需要重新生成受影响回归。

## 8. 正式执行与交付

### 8.1 新目录

```text
outputs/m2/acceptance/m2-fix-v5/
outputs/m1/acceptance/m2-fix-v5-regression/
```

目录存在时拒绝覆盖，使用明确的新 run_id。保留旧 r3 及此前中断目录。输入/模型身份、54 个物理案例、单元证据、全部比较、两阶段执行清单、反例结果和标定说明沿用 V4 的绑定要求。

### 8.2 执行流程

先完成 S1/S2 及解析反例，再跑现有 M2 回归；最终冻结代码后运行：

```bash
conda run -n feedingrobot python -m pytest tests/test_m2_*.py -q

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m1_report \
  --run outputs/m1/acceptance/m1-fix-v3b

MUJOCO_GL=egl conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 \
  --config configs/m1_scene.json \
  --output outputs/m1/acceptance/m2-fix-v5-regression

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m1_report \
  --run outputs/m1/acceptance/m2-fix-v5-regression --check-current-workspace
```

以下为 V5 待实现的最终生成/核验契约：

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m2 \
  --scope fixes-v5 \
  --scene-config configs/m1_scene.json \
  --controller-config configs/m2_controller.json \
  --acceptance-config configs/m2_acceptance.json \
  --m1-baseline outputs/m1/acceptance/m1-fix-v3b \
  --m1-regression outputs/m1/acceptance/m2-fix-v5-regression \
  --output outputs/m2/acceptance/m2-fix-v5

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m2_report \
  --scope fixes-v5 --run outputs/m2/acceptance/m2-fix-v5 \
  --check-current-workspace

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m2_report \
  --run outputs/m2/acceptance/m2-fix-v5 --check-current-workspace
```

前两条仅在修复范围全部达标时退出 0；第三条在完整 M2 仍缺项时退出 1。正常 fixes-v5 核验不需要特殊 TMPDIR；已有并发/异常清理测试保留，无须重新引入硬链接优化。

## 9. 最终验收清单

- [ ] 两个原始反例在新核心及完整 CLI 中均被具体目标规则拒绝，合法包通过。
- [ ] 增益期望由初始化规格和 phase 重建；自报 active/target/trans 不能决定自己的合格门槛。
- [ ] 中断返回、第三档切换、重复阶段、同档位 phase、终点边界均按真实 dt 通过；错误轨迹逐行可定位。
- [ ] gear 正常接触切换和真实增益完成事件仍存在，K/D 未被改成更容易通过的新参数。
- [ ] 六维施加载荷和六维零命令分别按单位核对；首次归零后至最后区间无重新加载，无推迟评分或跳过脉冲。
- [ ] 慢 ramp 不被误判；快释放的一步撤载、速度、位移、稳定窗口及无追补门槛全部保持。
- [ ] 原 54 个物理案例与完整 A/B/C 指标、单元测试及旧反例覆盖保留，计数不是替代规格的通过条件。
- [ ] 新 M1 完整回归和独立校验、历史基线校验、工作区身份、两阶段证据和派生摘要一致。
- [ ] 正式目录未覆盖；源包未被篡改测试污染；CLI 退出状态与请求范围一致。
- [ ] `fixes_passed` 有完整依据；完整 M2 尚未完成时 `m3_ready=false`，混合模式 disabled。

执行 Agent 最终报告应给出两处判定修改、原始反例的新错误码和退出码、合法轨迹的增益/释放关键指标、新正式包和 M1 回归路径，以及完整 M2 剩余任务。满足以上清单后关闭 V5，继续推进 M2Plan 的正式任务验收。
