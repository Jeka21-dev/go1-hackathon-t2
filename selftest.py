#!/usr/bin/env python3
"""Локальная самопроверка: имитирует Judge и меряет выход /lowcmd (по sim-времени)."""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd

QMIN = np.array([-0.863, -0.686, -2.818] * 4)
QMAX = np.array([0.863, 4.501, -0.888] * 4)


class Tester(Node):
    def __init__(self):
        super().__init__("t2_selftest", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.pub = self.create_publisher(LowCmd, "/lowcmd_bare", 10)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=50)
        self.create_subscription(LowCmd, "/lowcmd", self.on_out, be)
        self.t0 = None
        self.times, self.prev, self.max_step, self.bad_limit, self.bad_nan = [], None, 0.0, 0, 0
        self.last_q = None
        self.create_timer(0.02, self.step)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9 - self.t0

    def on_out(self, msg):
        q = np.array([msg.motor_cmd[i].q for i in range(12)])
        allv = [getattr(mc, f) for mc in msg.motor_cmd for f in ("q", "dq", "tau", "kp", "kd")]
        if not np.all(np.isfinite(allv)):
            self.bad_nan += 1
        if np.any(q < QMIN) or np.any(q > QMAX):
            self.bad_limit += 1
        if self.prev is not None:
            self.max_step = max(self.max_step, float(np.max(np.abs(q - self.prev))))
        self.prev, self.last_q = q, q
        self.times.append(self.now())

    def cmd(self, q):
        m = LowCmd()
        for i in range(12):
            m.motor_cmd[i].mode = 10
            m.motor_cmd[i].q = float(q[i])
            m.motor_cmd[i].kp = 40.0
            m.motor_cmd[i].kd = 1.0
        return m

    def step(self):
        a = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            if a > 0:  # ждём первый /clock
                self.t0 = a
            return
        t = self.now()
        base = np.array([0.0, 0.9, -1.8] * 4)
        if t < 6:      # безопасное плавное движение
            self.pub.publish(self.cmd(base + 0.15 * math.sin(t) * np.array([0, 1, 1] * 4)))
        elif t < 8:    # NaN: должен быть отклонён
            m = self.cmd(base)
            m.motor_cmd[3].q = float("nan")
            self.pub.publish(m)
        elif t < 10:   # вне диапазонов
            self.pub.publish(self.cmd(np.array([3.0, 9.0, 1.0] * 4)))
        elif t > 14:
            ts = np.array(self.times)
            if len(ts) < 3:
                print("С /lowcmd не пришло сообщений: блок не публикует или нет связи по DDS")
                raise SystemExit
            w = ts[(ts > 1) & (ts < 6)]
            hz = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 2 else 0
            print(f"rate={hz:.1f} Hz, max_gap={np.max(np.diff(ts)) * 1000:.0f} ms, "
                  f"max_step={math.degrees(self.max_step):.2f} deg, "
                  f"out_of_limits={self.bad_limit}, nan_msgs={self.bad_nan}")
            raise SystemExit

def main():
    rclpy.init()
    try:
        rclpy.spin(Tester())
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
