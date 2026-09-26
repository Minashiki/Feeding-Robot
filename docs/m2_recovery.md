# M2 中断恢复与本机资源限制

现场检查点：`outputs/m2/checkpoints/recovery-20260926T133445Z/`。
`files/` 保存修改和未跟踪源文件，`tracked.patch` 与 `HEAD.txt` 保存 Git 基点和差异；`source_manifest.json` 给出副本 SHA-256。原始轨迹仍在原目录，`evidence_inventory.json` 是清单，不是第二份轨迹备份。历史正式包不修改。

恢复时确认 `full-stop-005/results.json` 有 117 个初步标定组、当时评分无失败；`full-probe-007/progress.json` 只有 8 个完成组。所有这些记录仍是诊断资料，不能作为当前源码的完整 M2 放行证据。T10 按用户确认采用真实受阻释放与独立偏差上限联合验收。

## 执行限制

一律从项目根目录、feedingrobot Python 环境通过以下入口启动测试：

```bash
python -m feedingrobot.scripts.m2_limited \
  --output outputs/m2/resources/<新的唯一目录> -- \
  python -m pytest tests/test_m2_runtime.py -q
```

入口需要本机用户级 systemd。不可用时拒绝运行，不降级成无限制执行。资源限制作用于整个进程组及子进程：

- CPUQuota=100%，合计最多一个逻辑 CPU 的调度时间，不是全机 CPU 的 100%。
- MemoryHigh=4 GiB，MemoryMax=6 GiB，MemorySwapMax=0。
- nice=15，ionice=idle；数值库线程数全部固定为 1。
- 默认 1 worker，本机入口最多允许 2，但总 CPU 限额仍为一核。
- 项目文件锁与残留 scope 检查阻止同时启动多批任务。
- 每次保存 launch.json、limits.json、result.json；缺少结果文件的任务不能认定正常结束。

`launch.json` 中记录 systemd unit 名。需要停止时执行 `systemctl --user stop <unit>`；不要按 python 进程名批量杀进程。中断不产生 M3 放行。

## 小批诊断与正式验收

`m2_probe` 只接受标定种子 0–2，以完整 A/B/C 组顺序运行，使用独立诊断 schema，永远保持 m3_ready=false：

```bash
python -m feedingrobot.scripts.m2_limited \
  --output outputs/m2/resources/<新的唯一目录> -- \
  python -m feedingrobot.scripts.m2_probe \
  --case stop-p0-a0-d1-f0-k500-delay0-noise0-speed-free \
  --seed 0 --output outputs/m2/calibration/<新的唯一目录>
```

完整矩阵默认 worker 已改为 1，并核验实际 cgroup 限制；未冻结、反例阶段未完成或缺项时不得进入 M3。源码改变后必须新建诊断/验收目录，不把修改前后的数组混成一个正式包。当前完整验收链仍在修复中，不能把运行时回归通过当作 M2 全部完成。
