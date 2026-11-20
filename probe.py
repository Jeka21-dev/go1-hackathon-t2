#!/usr/bin/env python3
"""Ручная проверка: шлёт команду в /lowcmd_bare и показывает, что приходит в /lowcmd."""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd

PRESETS = {
    "stand": [0.0, 0.9, -1.8] * 4,
    "crouch": [0.0, 1.2, -2.4] * 4,
    "extreme": [3.0, 9.0, 1.0] * 4,
    "nan": [0.0, 0.9, -1.8] * 4,
    "silence": None,
}


def main():
    if len(sys.argv) < 3 or (sys.argv[1] == "q" and len(sys.argv) < 4):
        print(__doc__)
        return
    name = sys.argv[1]
    if name == "q":
        target = [float(x) for x in sys.argv[2].split()]
        sec = float(sys.argv[3])
        assert len(target) == 12, "нужно 12 чисел"
    else:
        target, sec = PRESETS[name], float(sys.argv[2])

    rclpy.init()
    node = Node("probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
    pub = node.create_publisher(LowCmd, "/lowcmd_bare", 10)
    last = {"q": None, "n": 0, "nan": 0}

    def on_out(m):
        q = np.array([m.motor_cmd[i].q for i in range(12)])
        last["q"], last["n"] = q, last["n"] + 1
        vals = [getattr(c, f) for c in m.motor_cmd for f in ("q", "dq", "tau", "kp", "kd")]
        if not np.all(np.isfinite(vals)):
            last["nan"] += 1

    node.create_subscription(LowCmd, "/lowcmd", on_out,
                             QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=50))
    now = lambda: node.get_clock().now().nanoseconds * 1e-9
    while rclpy.ok() and now() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)

    t0 = now()
    next_pub, next_print, n_prev = t0, t0 + 0.5, 0
    while rclpy.ok() and now() - t0 < sec:
        t = now()
        if target is not None and t >= next_pub:
            m = LowCmd()
            for i in range(12):
                m.motor_cmd[i].mode = 10
                m.motor_cmd[i].q = float(target[i])
                m.motor_cmd[i].kp = 40.0
                m.motor_cmd[i].kd = 1.0
            if name == "nan":
                m.motor_cmd[3].q = float("nan")
            pub.publish(m)
            next_pub += 0.02
        if t >= next_print:
            q = last["q"]
            if q is None:
                print(f"t={t - t0:4.1f} с  /lowcmd пока не приходит")
            else:
                dev = ""
                if target is not None and name != "nan":
                    dev = f"  |выход-цель|max={np.max(np.abs(q - np.array(target))):.3f} рад"
                print(f"t={t - t0:4.1f} с  FR q={np.round(q[:3], 3)}  "
                      f"{(last['n'] - n_prev) * 2} сообщ/с  NaN на выходе: {last['nan']}{dev}")
            n_prev, next_print = last["n"], next_print + 0.5
        rclpy.spin_once(node, timeout_sec=0.005)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
