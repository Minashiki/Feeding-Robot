# M1 接口

`FeedingScene.step_physics(tau_arm)` 推进一步物理，`dt = 0.001 s`。`tau_arm` 是七维关节力矩，单位 N·m，软件限幅与 motor 的 `forcerange` 一致：关节 1–4 为 ±87，关节 5–7 为 ±12。头和下颌由场景内部驱动，不占用这七维。

状态里的机器人量通过关节名索引，不假设 `qpos[:7]`。`tcp_pos` / `tcp_mat` 是勺碗承载面中心：工具 +X 指向勺尖，+Z 指向开口上方。`raw_wrench_sensor` 是 `ft_site` 上 MuJoCo 力/力矩传感器原值（父体对子体、site 坐标）。`wrench_sign = -1` 把它变成外界作用于工具的世界系 wrench，力矩仍关于 `ft_site`。`tcp_wrench_world` 再把力矩平移到 TCP。`compensated_wrench` 在 M2 之前固定为 `None`。

`reset` 会清掉 ctrl、外力和 Python 侧计数。沉降后的 `data.time` 保留，回合时间用 `episode_time = sim_time - time_offset`。`diagnostic_pd_tau` 只给测试和演示用。
