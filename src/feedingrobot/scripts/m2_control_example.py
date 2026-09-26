"""Public M3 stepping contract: python -m feedingrobot.scripts.m2_control_example."""
import argparse
import json

import numpy as np

from feedingrobot.controllers.cartesian_impedance import CartesianImpedance, control_step
from feedingrobot.controllers.guard import Guard
from feedingrobot.sim.model import load_config
from feedingrobot.sim.scene import FeedingScene


def run(pose=0):
    config = load_config('configs/m2_controller.json')
    scene_config = load_config('configs/m1_scene.json')
    scene_config['presets']['food_on_plate']['qpos'] = scene_config['q_torque_poses'][pose]
    scene = FeedingScene(scene_config)
    guard = Guard(config, config['joint_speed_limit_rad_s'])
    controller = CartesianImpedance(scene, config)
    controller.guard = guard
    state = scene.reset(seed=0, settle_steps=0)
    controller.reset(state, scene.last_tau)

    def drive(duration, twist):
        for k in range(round(duration/scene.dt)):
            if k % round(.05/scene.dt) == 0:
                result = controller.set_command(twist, controller.now(), controller.now()+.1, 'TRANSPORT')
                assert result['accepted'], result
            state, info = control_step(controller, guard)
            assert info['apply'] and guard.failure is None, guard.failure
        return state

    drive(1., np.zeros(6))
    drive(.3, np.array([0., .01, 0., 0., 0., 0.]))
    controller.set_command(np.zeros(6), controller.now(), controller.now()+.1, 'STOP')
    for _ in range(round(.8/scene.dt)):
        control_step(controller, guard)
    assert guard.stopped_ok and guard.failure is None
    assert not controller.set_command(np.ones(6), controller.now(), controller.now()+.1, 'TRANSPORT')['accepted']

    # Reset is explicit; a new episode starts with a fresh physical state.
    controller.reset(scene.reset(seed=0, settle_steps=0), scene.last_tau)
    drive(1., np.zeros(6))
    controller.set_command(np.full(6,np.nan), controller.now(), controller.now()+.1, 'TRANSPORT')
    tick, time = scene.tick, scene.data.time
    for _ in range(3):
        _, info = control_step(controller, guard)
        assert not info['apply'] and scene.tick == tick and scene.data.time == time
    controller.reset(scene.reset(seed=0, settle_steps=0), scene.last_tau)
    drive(1., np.zeros(6))
    return {'pose': pose, 'passed': True, 'tick': scene.tick}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose',type=int,choices=(0,1,2),default=0)
    print(json.dumps(run(parser.parse_args().pose)))


if __name__ == '__main__':
    main()
