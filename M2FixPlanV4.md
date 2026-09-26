# M2 第四轮修复方案：完整字段验收、隔离临时副本与快速释放

编写日期：2026-09-25。关联：[M2FixPlanV3.md](M2FixPlanV3.md)、[M2Plan.md](M2Plan.md)、[SimModelPlan.md](SimModelPlan.md)。

**本轮目标：关闭 V3 审查仍能复现的验收漏检与运行故障，将已完成独立试验的快速撤载正式纳入证据链。保留现有阻抗控制器、接触减速和已通过的收敛方案。**

本文是待执行任务书；本次仅编写方案，不表示代码已修复或 V4 已验收。本文细化、补充 V3 的未闭合要求，不降低原物理门槛。完整 M2 尚缺项时，`m3_ready` 必须保持 false。

## 1. 起点与已确认事实

### 1.1 仓库与证据身份

- GitHub：[Minashiki/Feeding-Robot](https://github.com/Minashiki/Feeding-Robot)。地址取自本地 `origin`，不声称远端已包含本地实现。
- 编写时 HEAD：`450f65a8fce245c68aca52b2581c8f6788225560`，提交说明 `M2 v1.2`。V3 实现仍包含工作区修改和未跟踪文件，**不能把 HEAD 当成完整源码身份**。执行前保存差异和独立输入哈希。
- 固定物理基线：[m1-fix-v3b/report.json](outputs/m1/acceptance/m1-fix-v3b/report.json)。
- 上轮正式包：[m2-fix-v3-r2/report.json](outputs/m2/acceptance/m2-fix-v3-r2/report.json)。它是本轮审查起点，不能直接升级为 V4 认证。
- 上轮新 M1 回归：[m2-fix-v3-regression/report.json](outputs/m1/acceptance/m2-fix-v3-regression/report.json)。
- 更早的 `m2-fix-v3` 因比较行事件顺序不稳定未被采用；继续保留原目录，不覆盖、重命名或修补为正式结果。

### 1.2 上轮独立审查结果

| 检查 | 实测结果 | 本轮如何使用 |
|---|---|---|
| 现有 M2 测试 | 178 passed，60.80 s | 保留原测试语义，补充漏检反例 |
| M1 完整回归 | 28 passed，83.33 s | 修改完成后生成新的完整回归 |
| 新 M1 报告与当前工作区、历史 v3b 独立校验 | 均通过 | 历史基线保持只读；新源码需新的绑定 |
| r2 的收敛比较 | 516 条比较均通过 | 保留全指标比较，不退回仅比较位置误差 |
| 45 份物理轨迹数值扫描 | 未发现非有限数值 | 漏检反例不等于正式轨迹实际存在 NaN |
| 默认 V3 独立校验命令 | 创建硬链接报 `Invalid cross-device link`，退出 1 | 修复默认运行路径，不要求用户手工设置 TMPDIR |
| 改用隔离、可硬链接的临时目录后核验 r2 | 退出 0，fixes_passed=true | 证明原数据可复算，同时暴露环境依赖 |
| 含 NaN 增益的临时副本完整核验 | 退出 0，reasons=[] | 必须变为明确失败 |
| 默认 full 范围 | 退出 1，首先报告缺 T08 | 保留；不能解释成完整 M2 只剩 T08 |

新 gear 接触力峰值 A=1.550334 N、B=1.696535 N，差 0.146201 N，小于允许差 0.169653 N。0.008 m/s 接近、接触后 0.002 m/s 慢压、接触条件满足后切换 ACQUIRE、总时长 2.0 s 的方案符合允许的正常标定方向，本轮不要求恢复撞击更大的旧刺激。

### 1.3 已有快速撤载实测，不冒充正式认证

上轮审查在临时输出中，把受阻后的外力撤除时间改为一个物理步，执行 seed 0/1/2 × A/B/C 共九例。原测试断言通过，独立复算得到：

| 指标 | 九例最差值（取保守显示位数） | 门槛 |
|---|---:|---:|
| 释放后 0.5 s TCP 线速度峰值 | 0.01349 m/s | ≤0.03 m/s |
| 同窗口沿 +X 额外位移 | 1.490 mm | ≤5 mm |
| 最后稳定窗口线速度 | 0.001198 m/s | ≤0.005 m/s |
| 最后稳定窗口角速度 | 0.004608 rad/s | ≤0.05 rad/s |

seed 0 的 A/B 释放时间差约 1 ms，受阻首次时间相同；这些数据支持保留现有控制器。**临时试验尚未形成 V4 完整报告、完整收敛比较和 M1 绑定，必须重新正式运行，不能复制临时数组来代替新执行。**

## 2. 当前必须修正的三项问题

| 编号 | 已复现问题 | 根因与影响 | 完成定义 |
|---|---|---|---|
| V4-01 必需字段漏检 | 分别把 K 写入 NaN、暂停 v_ref 改成 5 m/s、q 改成 100 rad、mode 全改为 ABORTED，更新证据哈希后，核心校验均通过；NaN 副本完整 V3 校验也通过 | `_check_physical()` 的 needed 列表只证明字段存在；有限性只覆盖少量数组；关节范围、参考速度及正常终止没有完整独立验收 | 固定字段规格覆盖所有必需量；结构、数值及语义三层检查全部执行；四项反例均被完整 CLI 拒绝 |
| V4-02 临时副本依赖硬链接和共享目录 | 正式包复制到默认 `/tmp` 时出现跨设备硬链接错误 | `copy_function=os.link` 有挂载限制；固定 `m2-v3-adversarial` 目录还会被其他运行删除 | 普通独立复制、每次独占临时目录；默认环境和并发检查均可复现；不改源包 |
| V4-03 快速释放证据缺失 | 现有 release 在受阻后用 1.2 s 撤载，外力归零后才开始评分 | 该窗口不能证明解除受阻时的瞬态；慢卸载与快速移除约束不是同一测试 | 新增九个正式快速撤载案例，记录并重建受阻、开始撤载、首个零外力区间；覆盖整个释放瞬态和稳定窗口 |

其中 V4-01 是验收结论的阻塞问题；V4-02 是可复核性的阻塞问题；V4-03 是与规划目标的一致性缺口。不得仅以“控制器目前跑起来正常”关闭这三项。

## 3. 边界与实施顺序

### 3.1 保持不变

1. 依赖版本保持 MuJoCo 3.14.0、NumPy 2.2.6、pytest 9.1.1。
2. 控制链保持世界系 twist → 整形/参考积分 → `Jᵀ(Ke+Dė)+bias+零空间` → 力矩限幅/斜率 → `step_physics`。
3. 正常关节速度 ≤0.5 rad/s、预测速度上限 0.4 rad/s；实际力矩前四轴 ±87 N·m、后三轴 ±12 N·m；变化率 2000 N·m/s，含首步真实起点。
4. M1 主资产、配置、事件和 20 ms 收敛门槛不改；高速加速诊断参数不进入正常控制。勺—餐盘接触由 ACQUIRE 阶段及具体 geom 白名单允许。
5. 混合法向力保持 `disabled`。旧证据只读，任何重新执行使用新的唯一输出目录。
6. 不因验收字段缺失重构控制器；只补充真实日志，发现具体算法问题时才修改对应算法。

### 3.2 工作顺序

| 步骤 | 工作 | 必须检查的出口 |
|---|---|---|
| S0 | 固定当前输入身份和四个漏检反例 | 未修版反例成立，原包未改 |
| S1 | 临时副本改为普通复制和独占目录 | 默认目录、并发、异常清理、源包只读检查通过 |
| S2 | 字段规格、完整数值/形状检查与错误码 | 必需字段无遗漏；非法值在指标计算前拒绝 |
| S3 | 增益、参考、关节范围、运行状态及相应日志补齐 | 有限但错误的数据也拒绝；合法轨迹通过 |
| S4 | 快速撤载九例、全指标 A/B/C、现有 45 例回归 | 54 个物理案例全部按各自定义通过 |
| S5 | 正例上的单项篡改集、最终 M1/M2 包、独立 CLI | V4 修复范围通过；full 因真实缺项失败 |

主要修改文件：`controllers/v3spec.py`、`controllers/acceptance.py`、两个 M2 CLI、`tests/m2_evidence.py`、`tests/test_m2_v2.py`、报告反例测试及标定文档。可新增小型 `tests/test_m2_v4.py`。复用当前实现，不复制整份 1300 多行校验器到另一个版本文件；模块名可以暂时保留，公共 schema/规格版本必须明确。

## 4. V4-01：所有必需字段都必须真正验收

### 4.1 先建立字段检查表，再计算物理指标

使用一个简单的 `FIELD_SPECS` 字典规定 shape、dtype、是否允许空、有限性和语义检查入口。所有物理案例共用基础表，族专有字段追加。校验顺序固定为：

```text
文件身份/类型 → 必需字段集合 → 原始 dtype/精确 shape → 有限性 →
索引、枚举与时间 → 状态/控制语义 → 族指标 → A/B/C → 摘要一致性
```

不要先 `reshape()` 或 `astype(float/int/str)` 再检查格式，以免错误数组长度、数值字符串、布尔值和小数 tick 被静默修正。结构失败时记录 case/field 并停止该案例的指标计算，继续收集其他案例错误；不能返回空指标后又跳过必需比较。

### 4.2 固定 shape 与类型

设 N 为规格时长/dt 的整数，C 为该案例扁平接触记录数，nq/nv 来自固定模型：

| 字段 | shape | 检查要求 |
|---|---|---|
| t_state、tick | `(N+1,)` | 时间为有限实数；tick 为原始整数，连续；初始时间/tick 对应元数据 |
| q、dq | `(N+1,7)` | 有限实数、列顺序绑定真实 arm 索引 |
| tcp_pos、tcp_twist | `(N+1,3)`、`(N+1,6)` | 有限实数；twist 顺序固定为线速度、角速度 |
| tcp_rot | `(N+1,3,3)` | 有限且每帧为合法 SO(3) |
| tau_applied、tau_raw | `(N,7)` | 有限；只有实际施加力矩执行硬幅值/变化率验收，raw 可因限幅而超出幅值 |
| tau_before、qpos0 | `(7,)`、`(nq,)` | 有限，初始化身份相符；补记 qvel0 `(nv,)` 以核对配对初始速度 |
| command、v_ref、K、D | `(N,6)` | 有限实数；各自执行不同语义检查，不把原始命令当成已整形速度 |
| p_ref、r_ref | `(N,3)`、`(N,3,3)` | 有限；旋转合法；使用步前状态检查参考偏差 |
| wrench_raw、wrench_compensated、external_wrench | `(N,6)` | 有限；注明坐标系、作用方向、力矩参考点和步前/步后采样 |
| force_used | `(N,3)` | 有限，真实用于当前整形的上一可用补偿力 |
| phase、mode、execution、active_gear、target_gear | `(N,)` | 真正的字符串数组；取值必须属于固定枚举；禁止未知值默认 FREE/STOP |
| pause_applied、saturation_flags、blocked_now、transition_active | `(N,)` | 统一用 bool，或事先声明的严格 0/1 整数；禁止任意实数按 >0.5 转布尔 |
| warnings、guard_status、first_fault_tick | 按明确固定规格 | warnings 为逐步非负整数计数；guard 状态与首故障字段不能缺省成正常 |
| contact_offsets | `(N+1,)` | 原始整数；首项 0、末项 C、单调不减，范围均在 `[0,C]` |
| contact_force、contact_torque、contact_point | `(C,3)` | 有限；三个字段与 geom、distance 使用相同 C |
| contact_dist、contact_geom1/2、contact_frame | `(C,)`、`(C,)`、`(C,9)` | 有限数值、已知 geom 名、合法接触坐标架；不得缩短任何一个数组 |

无接触时允许 C=0，此时接触数组必须具有上述零行 shape，offsets 全为 0。除此以外，必需状态/控制数组不能为空。某字段若决定不用于验收，就不应列为必需证明量；不能保留关键字段名称却完全忽略其值。

旋转合法性采用 `max|RᵀR-I|≤1e-6、|det(R)-1|≤1e-6`；这是格式容差，不放宽姿态误差门槛。所有非有限量一律拒绝，不用 `nan_to_num`、删帧或填零恢复。未知额外字段不得覆盖核心字段或提供新的门槛。

### 4.3 增益：有限并不等于正确

1. 检查 K/D 全部有限、非负，平移和旋转三通道分别与固定配置匹配。
2. 静态档位期直接核对冻结的 FREE/ACQUIRE/MOUTH 增益；正常案例出现任意有限巨大增益也必须失败。
3. 档位切换从轨迹中的阶段变化及真实初始增益重建：新过渡从当前实际 K/D 开始，按真实 dt 累计，0.2 s 完成。中途返回原档位不跳变，重复同阶段命令不重新计时。
4. 逐通道核对 `gain(t)=(1-alpha)*start+alpha*target` 及终值，容差沿用摘要 `atol=1e-9、rtol=1e-7`。不能只比对相邻最大跳变量，常数错误增益同样要失败。
5. `transition_active`、起止事件与实际 K/D 互相一致；必要时补记 transition_elapsed 和真实起点，但校验期不能无条件相信这些派生值。

冻结配置快照必须与运行输入哈希绑定。不能修改快照中的增益或速度上限，再用这份快照认可同一包的异常数据。

### 4.4 参考速度：区分任务积分、约束校正和暂停

当前只有 v_ref 文件存在，不足以证明暂停没有推进。最小日志补充包括：实际送入整形器的暂停开关、execution 分支、整形前参考、普通任务积分增量、参考校正量/原因、最后 v_ref。记录值取自真正执行位置，不从测试计划时间猜测“应该暂停”。

必须验证：

- 线速度和角速度分别不超过当步生效的档位上限；过渡时保持更严格限制。FREE 为 0.05 m/s、0.3 rad/s；ACQUIRE 为 0.02、0.15；MOUTH 为 0.01、0.1。更严格的实际约束只能进一步收紧。
- `power_on/stop/prohibit` 或实际饱和暂停分支中，任务积分增量为 0，v_ref 为 0（范数容差 1e-9）；在此类分支写入 5 m/s 必须拒绝。
- 零上层命令不自动等于硬暂停：正常 `zero` 分支可以执行整形减速。验收依据实际分支，不强制所有零命令瞬间清历史速度。
- 平移参考变化分解为“任务积分 + 约束/重锚校正”，逐步核对；旋转用 SO(3) 组合核对，不能直接相加欧拉角。步前状态用于参考偏差，步后状态用于响应指标。
- 发生参考校正或重锚时，按现有控制语义核对 v_ref；不能把校正除以 dt 伪造高速运动，也不能在暂停期间积累待执行任务位移。
- 参考线/角速度限制不等同于实际 TCP 速度硬门槛。接管、扰动或释放的实际速度按对应物理案例判定，不能混用以制造误报。

若日志中的 mode 是控制步末更新值，而 execution 是控制步开始实际采用的分支，应明确保存 `mode_before/mode_after` 或同等字段；验收不能通过错一拍的标志判断“本步未执行 power_on”。只修日志时序，不为日志检查改变控制行为。

### 4.5 关节范围与正常结束

关节 q 范围从已冻结的 Panda 模型及 `ModelIndex` 的 arm joint 映射读取，可离线加载模型而不推进物理。不得使用报告自带的宽范围，不假设所有关节都是 ±π。逐帧要求 `q_min-1e-6 ≤ q ≤ q_max+1e-6 rad`；100 rad 必须得到 `joint_range`。模型身份/映射也进入输入哈希。dq≤0.5 rad/s、力矩幅值和 2000 N·m/s 变化率继续逐步检查。

正常五族及新增快速释放案例要求：

- 全时长完整执行，未出现 STOPPING/ABORTED 或首故障，warnings 为 0。
- 除 takeover 有真实 POWER_ON→RUNNING 过程外，其余正常案例采用实际 RUNNING；takeover 必须在规定窗口内完成接管。未知状态直接拒绝。
- 保存每步 guard 状态和最终状态，首故障存在与否采用明确字段/掩码；正常包全程 ABORTED 必须失败，不因轨迹足够长而通过。
- 已有 F3 等故障注入案例使用各自规定的预期故障、首 tick 和停止窗口；不能把正常案例的要求错误套到故障测试上，也不能让故障测试名称成为任意异常的豁免。

范围、增益或模式等语义失败需独立错误码，不依赖比较结果偶然变化而间接报错。对所有案例，字段校验与语义检查必须在生成器和独立校验器中采用同一固定规格，不能只写 pytest 断言。

## 5. V4-02：可移植、独占且不污染源包的反例副本

### 5.1 最小实现

用标准库 `tempfile.TemporaryDirectory(prefix="m2-v4-adversarial-")` 为每次调用创建独占根目录，默认服从正常临时目录选择；复制使用 `shutil.copytree(..., copy_function=shutil.copy2)`。不使用硬链接，不要求临时目录与 run 位于同一挂载点。

```python
with TemporaryDirectory(prefix="m2-v4-adversarial-") as work:
    for case in ADVERSARIAL_SPECS:
        with TemporaryDirectory(prefix="case-", dir=work) as case_work:
            dest = Path(case_work) / "run"
            shutil.copytree(source_run, dest, copy_function=shutil.copy2)
            # 确认未改副本的核心判定通过，再作一次指定破坏。
            # 执行固定规则复算，记录具体错误及证据身份。
```

一次只保留一个副本，避免 20 多个完整 NPZ 包同时占空间。日志结果保留在调用方结果对象或输出中，临时副本自动清理。只删除当前调用创建的目录，不按固定名称删除现有目录；源包、其他运行目录、用户指定的 TMPDIR 本身一律不删。

此方案只使用普通复制，不需要“先硬链接、跨设备失败再回退”的双路径。副本里的所有待修改文件都必须是独立文件；对源包外的符号链接按既有路径约束拒绝，不允许篡改测试跟随其写入其他位置。

### 5.2 验收方式

1. **默认路径：** 不设置特殊 TMPDIR，正式 `verify_m2_report --scope fixes-v4` 完整运行，正常退出 0。
2. **显式临时根：** 指定另一个可写临时根同样通过。若环境可提供跨挂载目录则实测；另以测试中禁止调用 `os.link` 的方式确认实现已无硬链接依赖。
3. **并发：** 两个进程同时核验同一只读包，使用同一个 TMPDIR，均成功、结果一致，互不删除文件。无需重复物理仿真。
4. **异常清理：** 在一个副本篡改步骤注入预期异常，只有该调用的临时目录被清理；另一调用继续完成，源包不变。
5. **源包只读：** 运行前后对源包完整文件集合及内容哈希比对；所有反例修改只能落在副本。读取带来的访问时间变化不作为写入证据。
6. **错误输出：** 临时空间不可写等真实 I/O 失败返回非零及明确 `io_error`，不能变成 fixes_passed，也不输出只含“物理失败”的误导结论。

## 6. V4-03：把快速解除受阻正式纳入验收

### 6.1 案例组织

保留原 45 个代表案例作为回归，其中 `release-s{seed}-{variant}` 明确标为慢卸载。新增 `release_fast-s{seed}-{variant}`，seed=0/1/2，variant=A/B/C，共九例。完整修复范围固定为 **54 个物理案例**，单元/状态机案例另外计数。

快速释放沿用当前受阻夹具：构型 0、世界系 +X 推进、−X 3 N 外力、2.5 s 总时长。A/C 的 dt=1 ms、N=2500；B 的 dt=0.5 ms、N=5000；状态分别为 N+1 帧。动力学、控制器和正常阶段保持一致，新增 family 不能因现有字符串分派漏接物理校验。

保留 seed 和构型的真实身份；九例不等于九个不同机械臂构型。保留 pause 族三构型要求。本轮仍不替代 M2Plan 的正式 2 s 物理挡面 T10、全部弹簧刚度或保留种子。

### 6.2 刺激与时间定义

1. 从与已有 release 相同的沉降初态开始，保存完整 qpos/qvel、控制参考、滤波器等初始身份。A/B/C 只改变规定 dt/solver，不各自修改刺激以迎合比较。
2. 从 t≥0.02 s 按既有 20 Hz 发布逻辑发送 +X 0.05 m/s 指令；t≥0.04 s 施加 −X 3 N。区分指令计划时间与实际生效时间，全部记录。
3. 用实际整形方向、力反馈和步前参考误差重建受阻：反向力 >0.3 N、沿推进方向滞后 >2 mm 连续 10 ms。若实现做了速度整形，使用真正进入该判断的方向，不以未整形命令替代。不得只信 sticky 的 `blocked` 标志。
4. 首次受阻被控制回路识别后，在下一个可用控制区间下发零命令，发起撤载。记录 `blocked_first、unload_start、force_zero` 三个事件及其 tick。
5. **快速撤载：从 unload_start 起最多一个本组物理步内外力变为零，随后保持零。** 不允许保留 1.2 s ramp，不提前降低 3 N，不等待弹性误差自行耗散后再宣告开始。
6. 卸载开始后不再发推进命令，不重置控制器、不人为重新锚定参考、不写 qpos/qvel、不暂停仿真、不跳过已产生的速度峰值。现有控制器正常约束校正仍执行并如实记录。

事件以施加于 `[t_k,t_k+1)` 的外力/命令为准。校验器从数组重建：受阻后命令首次归零为撤载请求边界，外力首次持续为零是 force_zero；核对与声明的 unload_start 相差不超过已定义的一步，且 `force_zero-unload_start≤dt+1e-9 s`。不允许由一条可编辑的 release_time 把评分窗口推迟。

### 6.3 完整评分窗口

为了覆盖整个瞬态，主评分起点采用 **unload_start**，同时报告从 force_zero 起算的原有指标。每个快速案例必须满足：

| 指标 | 固定门槛 |
|---|---|
| 受阻前提 | 独立重建成立，实际标志和日志一致 |
| 卸载持续时间 | ≤一个物理步；载荷由 −X 3 N 撤为零 |
| unload_start 后 0.5 s 实际 TCP 线速度峰值 | ≤0.03 m/s |
| 同窗口沿原 +X 推进方向额外位移 | `max(0, d·(p(t)-p_start))≤0.005 m` |
| force_zero 后相同速度/位移窗口 | 同样满足 0.03 m/s、5 mm |
| unload_start 后 0.8–1.0 s 稳定窗口 | 连续线速度 ≤0.005 m/s、角速度 ≤0.05 rad/s |
| 后续行为 | 至少覆盖完整 1 s；外力零、无推进命令、无故障、无旧任务追补 |

实际速度用模型 TCP twist，并用位置差分交叉核对；位移起点用事件边界真实状态，不能用下一帧或窗口最小位置。窗口含边界样本，不允许空窗口以 `np.any(empty)==false` 通过。校正量、任务积分与状态恢复分别记录。

慢卸载的成绩继续保留，但报告必须写清它只证明缓慢卸载响应，不能抵扣快速释放缺项。若新起点评分暴露问题，定位后修复，不移回“完全卸载后”来隐藏失败。

### 6.4 新增九例的 A/B/C

沿用 V3 全指标比较：位置/姿态 RMS 和峰值、固定接触通道、全场景接触合力、补偿后力/矩、七轴实际力矩峰值、累计载荷冲量、必需事件。快速族的必需事件至少包括 `blocked_first、unload_start、force_zero`，任一缺失即失败。

门槛不变：位置差≤max(较大值10%,0.1 mm)，姿态≤max(10%,0.05°)，力≤max(10%,0.02 N)，矩≤max(10%,0.002 N·m)，各轴力矩≤max(10%,0.05 N·m)，冲量≤max(10%,1e-4 N·s)，事件时间差≤20 ms。快速族无目标弹簧接触时，验证该通道实际为零，不伪造 contact 事件。

上轮九例快速试验只提供方向和初步性能依据；本轮必须将这些比较全部写入 comparisons 并独立重算，不能只引用速度/位移通过。

## 7. 本轮反例与正例必须一一对应

### 7.1 新增必需反例

所有物理规则反例从一个已经完整通过的 V4 包复制，先确认未改副本通过，再只改一项并更新该证据文件哈希；这样失败来自目标规则。哈希测试另保留旧哈希。最终 CLI 和核心检查都要覆盖，不能只断言核心函数失败。

| 反例 | 必须定位的错误 |
|---|---|
| K 或 D 中某个元素 NaN/Inf | `nonfinite`，定位 case/field/index |
| 有限但不符合档位/过渡的 K/D；过渡终值错误 | `gain_mismatch`/`gain_transition` |
| 暂停期间 v_ref=5 m/s | `reference_speed`/`pause_motion` |
| 暂停速度仍为零，但任务积分非零 | `pause_motion`，不能只看 v_ref |
| q 某一轴某帧=100 rad | `joint_range`，显示真实该轴上下界 |
| 正常轨迹 mode 全 ABORTED，或只有一帧 STOPPING | `unexpected_mode`/`unexpected_fault` |
| mode 是未知字符串；phase 不在定义中 | `invalid_enum`，不能降级 STOP/FREE |
| 必需字段 shape 错一维、缺一行、额外一列、数值字符串或 bool 冒充实数 | `bad_shape`/`bad_dtype` |
| 空 contact 数组但 offsets 不为零；offsets 回退、负数、越界或小数 | `bad_offsets`/`bad_dtype` |
| r_ref/tcp_rot 有限但不正交或 det≠1 | `invalid_rotation` |
| 快速族用原 1.2 s ramp 替代 | `unload_duration`；其余窗口全通过也不能放行 |
| 快速族删受阻前提、修改 unload_start 推迟评分、截掉卸载前后帧 | `event_mismatch`/`sample_count` |
| 快速族释放窗口速度超限、位移超限或稳定窗口不满足 | 各自明确的释放指标失败 |
| 修改 comparisons、report 或完整执行清单，保留 passed | 派生摘要/执行覆盖不一致 |

保留 V3 的 F6 固定门槛、空证据清单、错误文件类型、两帧释放、缺 A、错误 solver、失败 M1 和输入变化等反例。反例注册表明确 case、变化、预期错误码及字段；测试不能只要求“reasons 非空”。

### 7.2 不递归的两阶段执行

1. 核心阶段运行真实单元/物理案例、采样并生成核心指标；核心正例先通过。
2. 反例阶段在核心包的独立副本中执行固定篡改集，再记录 nodeid、setup/call/teardown、退出码和目标错误。不要让反例再次触发自身，递归生成副本。
3. 最终报告要求两阶段都完成；独立 CLI 先核验底层证据和反例执行记录，可重新运行同一篡改集，但必须通过核心入口复算临时副本。
4. 新四种漏检至少各有一个完整 CLI 集成测试，证明最终入口没有绕过失败。最终错误码是目标错误，可同时包含摘要不一致，不能仅靠后者认为漏检已修复。

反例结果列表按固定 ID 排序，字段顺序及事件集合排序确定，避免 V3 初版的进程间顺序差异。完整合法包是正例；不能用缺许多文件的空目录来证明某个单项反例有效。

## 8. 报告身份与执行状态

采用新 `schema_version=m2-fix-v4`、报告 scope=`fixes_v4`、CLI `--scope fixes-v4`。旧 V1/V2/V3 包可以历史读取或明确拒绝，不得用旧范围代替 V4 认证。当前脚本尚不支持此新契约，需在本轮实现。

保留字段：`evidence_valid、tests_passed、fixes_passed、scope_passed、m3_ready、hybrid_force_status、reasons、remaining_m2_requirements`。定义如下：

- 结构、身份、形状、有限性或派生摘要不一致：evidence_valid=false，fixes_passed=false。
- 结构完整但物理性能不合格：可以 evidence_valid=true，但 fixes_passed=false；保留失败原始数据。
- 测试节点未完成、teardown 失败或临时副本运行错误：不得将修复范围判通过。
- `fixes_passed` 由本轮完整规则决定；`scope_passed` 控制当前 CLI 退出码。full 缺项不应伪造修复包损坏，也不应把 tests_passed 改成“测试没通过”。
- 完整 M2 尚缺 T08、T09、物理挡面 T10、T11、T15、T16、完整 T17、种子 3–8 及完整矩阵，逐项保留；只显示第一条缺项不代表仅剩 T08。

生成过程中保留 incomplete；只有两阶段执行及证据文件写齐后才完成最终状态。生成器必须在反例阶段结束后重新核对输入 after，而不是仅在第一轮 pytest 后记录一次。任何共享源码/配置/测试变化都使本次 run 失败，不重写 before 掩盖变化。

证据 manifest 固定绑定所有必需文件和两阶段执行结果；comparisons/report 从原始数据复算比对，不参与循环自哈希。当前工作区模式重新收集输入集合；历史模式明确其只验证包内一致性。独立核验不改正式报告。

## 9. 执行命令与新产物

本轮只使用新目录：`outputs/m2/acceptance/m2-fix-v4/`、`outputs/m1/acceptance/m2-fix-v4-regression/`。目录存在时拒绝覆盖，改用显式的新 run_id；旧 V3/r2、旧 M1 基线不动。

开工前冻结源码、配置和规格，先局部验证字段校验、临时目录工具、快速释放，再执行全部 M2 测试。无须每修改一个反例都重新跑完整物理矩阵；最终冻结后必须有一次完整记录。

```bash
conda run -n feedingrobot python -m pytest tests/test_m2_*.py -q

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m1_report \
  --run outputs/m1/acceptance/m1-fix-v3b

MUJOCO_GL=egl conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 \
  --config configs/m1_scene.json \
  --output outputs/m1/acceptance/m2-fix-v4-regression

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m1_report \
  --run outputs/m1/acceptance/m2-fix-v4-regression --check-current-workspace
```

以下 fixes-v4 为待实现契约，生成入口需包含核心执行及反例执行，不能要求手工补 passed 标志：

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m2 \
  --scope fixes-v4 \
  --scene-config configs/m1_scene.json \
  --controller-config configs/m2_controller.json \
  --acceptance-config configs/m2_acceptance.json \
  --m1-baseline outputs/m1/acceptance/m1-fix-v3b \
  --m1-regression outputs/m1/acceptance/m2-fix-v4-regression \
  --output outputs/m2/acceptance/m2-fix-v4

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m2_report \
  --scope fixes-v4 --run outputs/m2/acceptance/m2-fix-v4 \
  --check-current-workspace

conda run -n feedingrobot python -m feedingrobot.scripts.verify_m2_report \
  --run outputs/m2/acceptance/m2-fix-v4 --check-current-workspace
```

前两条在本轮全部通过时退出 0；第三条在完整 M2 缺项时退出 1。默认核验不得依赖额外的 TMPDIR 设置；并发核验测试由独立测试脚本启动两个进程、等待两者完成并比较结果。

产物延续 V3：输入 before/after、M1 绑定、模型/配置身份、54 个物理案例和单元证据、完整 comparisons、两个执行清单、反例结果、环境及标定说明。补充快速释放三事件、各评分窗口、临时目录测试结果及源包前后哈希证明。

最终 M1 生成后若又改共同输入文件，重新生成受影响回归，不仅重写绑定。正式脚本所用配置必须与 CLI 声明和快照一致；不支持其他配置时明确拒绝。

## 10. 最终验收与停止条件

- [ ] 四个原始漏检反例在核心函数及完整 fixes-v4 CLI 中均被正确拒绝，合法包仍通过。
- [ ] 必需字段规格完整；每种 shape/dtype/有限性/枚举/索引约束都有对应反例，无关键字段仅检查存在。
- [ ] 有限但错误的增益、参考速度、关节位置、运行状态不能通过；物理语义从实际日志复算。
- [ ] 默认临时目录、显式临时根、两个并发核验、异常清理均通过；不使用硬链接，源包哈希不变。
- [ ] 原 45 例保留；新增九例快速撤载完整执行，所有窗口与全指标 A/B/C 达标，慢卸载不替代快速释放。
- [ ] 原 F2/F3/F4/F5/F6/F7、reset、重放和已有 M2 回归保留；测试数可以因参数化改变，覆盖不得缩水。
- [ ] 新 M1 完整回归及工作区校验、固定 v3b 历史校验通过；模型/配置及本轮证据身份可追溯。
- [ ] 生成器、独立校验、反例执行清单和派生摘要一致，无按错误字符串猜测结论的旁路。
- [ ] 2000 N·m/s、阶段接触白名单、混合 disabled、M1 资产和 20 ms 门槛均保持。
- [ ] fixes-v4 退出 0 有完整依据；完整 M2 缺项时 m3_ready=false，默认 full 非零退出。

以上全部满足后，关闭 V4 并继续 M2Plan 的正式任务验收。最终汇报给出具体修复、反例拒绝原因、快速释放最差指标、全矩阵结果及新报告路径；不以测试数量、哈希一致或 full 仍失败替代本轮关闭证据。
