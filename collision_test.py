#!/usr/bin/env python3
"""Проверка самоколлизий и удержания цели. Запускать внутри контейнера Participant
на СВЕЖЕМ запуске симулятора (перезапустите run_local.py перед тестом)."""
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_block import CollisionChecker, Q_MIN, Q_MAX, STAND  # noqa: E402

LIMIT = 0.001  # запрет ТЗ: проникновение строго больше 1 мм


class Test(Node):
    def __init__(self):
        super().__init__("t2_collision_test",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        here = os.path.dirname(os.path.abspath(__file__))
        self.chk = CollisionChecker(os.path.join(here, "unitree_go1", "scene.xml"),
                                    os.path.join(here, "unitree_go1", "t2_collision_pairs.csv"),
                                    0.1, self.get_logger())
        self.pub = self.create_publisher(LowCmd, "/lowcmd_bare", 10)
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=50)
        self.create_subscription(LowCmd, "/lowcmd", self.on_out, be)
        self.out = []

    def on_out(self, msg):
        self.out.append((self.now(), np.array([msg.motor_cmd[i].q for i in range(12)])))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, sec):
        end = self.now() + sec
        while rclpy.ok() and self.now() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def send(self, q, sec):
        end = self.now() + sec
        while rclpy.ok() and self.now() < end:
            m = LowCmd()
            for i in range(12):
                m.motor_cmd[i].mode = 10
                m.motor_cmd[i].q = float(q[i])
                m.motor_cmd[i].kp = 40.0
                m.motor_cmd[i].kd = 1.0
            self.pub.publish(m)
            self.spin_for(0.02)

    def audit(self, start):
        """Все соседние выходные конфигурации, проверка с шагом 0,1 градуса."""
        qs = [q for _, q in self.out[start:]]
        worst, viol = 0.0, 0
        for a, b in zip(qs, qs[1:]):
            n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / math.radians(0.1))))
            for k in range(1, n + 1):
                p = self.chk.penetration(a + (b - a) * k / n)
                worst = max(worst, p)
                if p > LIMIT:
                    viol += 1
                    break
        return len(qs), viol, worst

    def run(self):
        while rclpy.ok() and self.now() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
        rng = np.random.default_rng(0)
        pen = self.chk.penetration

        print("A. цели с самоколлизией")
        bad = []
        for _ in range(20000):
            q = rng.uniform(Q_MIN, Q_MAX)
            if pen(q) > 0.003:
                bad.append(q)
            if len(bad) >= 15:
                break
        print(f"   найдено конфигураций с проникновением > 3 мм: {len(bad)}")
        self.send(STAND, 3.0)
        i0 = len(self.out)
        for q in bad:
            self.send(q, 0.5)
        self.spin_for(0.5)
        n, viol, worst = self.audit(i0)
        print(f"   сообщений {n}, нарушений {viol}, макс. проникновение {worst * 1000:.2f} мм")

        print("B. безопасные концы, путь между которыми проходит через коллизию")
        pairs = []
        for _ in range(800):
            a = np.clip(STAND + rng.uniform(-0.6, 0.6, 12), Q_MIN + 1e-3, Q_MAX - 1e-3)
            b = np.clip(STAND + rng.uniform(-0.6, 0.6, 12), Q_MIN + 1e-3, Q_MAX - 1e-3)
            if pen(a) <= 0.0008 and pen(b) <= 0.0008 and not self.chk.segment_ok(a, b, LIMIT, 0.0):
                pairs.append((a, b))
            if len(pairs) >= 5:
                break
        print(f"   найдено пар: {len(pairs)}")
        i0 = len(self.out)
        for a, b in pairs:
            self.send(a, 2.0)
            self.send(b, 2.0)
        n, viol, worst = self.audit(i0)
        print(f"   сообщений {n}, нарушений {viol}, макс. проникновение {worst * 1000:.2f} мм")

        print("C. удержание последней принятой цели при молчании входа")
        target = np.clip(STAND + np.array([0.05, 0.1, -0.1] * 4), Q_MIN + 1e-3, Q_MAX - 1e-3)
        self.send(target, 4.0)
        i0 = len(self.out)
        t0 = self.now()
        self.spin_for(1.5)
        tail = [(t, q) for t, q in self.out[i0:] if t - t0 > 0.5]
        if tail:
            dev = max(float(np.max(np.abs(q - target))) for _, q in tail)
            print(f"   выход после 0,5 с тишины: сообщений {len(tail)}, макс. отклонение от цели {dev:.4f} рад")
        else:
            print("   выход прекратился: НЕДОПУСТИМО")


def main():
    rclpy.init()
    t = Test()
    t.run()
    t.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
