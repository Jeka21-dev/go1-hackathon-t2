#!/usr/bin/env python3
import csv
import math
import os

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd

QMIN = np.array([-0.863, -0.686, -2.818] * 4)
QMAX = np.array([0.863, 4.501, -0.888] * 4)
STAND = np.array([0.0, 0.9, -1.8] * 4)
NAMES = [f"{l}_{j}_joint" for l in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
HERE = os.path.dirname(os.path.abspath(__file__))
STEP = math.radians(0.1)
NAN, INF = float("nan"), float("inf")


class Checker:
    def __init__(self):
        d = next(p for p in (os.path.join(HERE, "unitree_go1"), "/opt/t2/unitree_go1")
                 if os.path.exists(os.path.join(p, "scene.xml")))
        self.m = mujoco.MjModel.from_xml_path(os.path.join(d, "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr = [int(self.m.joint(n).qposadr[0]) for n in NAMES]
        with open(os.path.join(d, "t2_collision_pairs.csv"), newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r["diagnostic_policy"].strip() == "checked"]
        self.pairs = [(int(r["geom1_id"]), int(r["geom2_id"])) for r in rows]
        assert len(self.pairs) == 655, len(self.pairs)
        self.ft = np.zeros(6)

    def pen(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        worst = 0.0
        for a, b in self.pairs:
            dist = mujoco.mj_geomDistance(self.m, self.d, a, b, 0.05, self.ft)
            if -dist > worst:
                worst = -dist
        return worst

    def path_pen(self, a, b):
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / STEP)))
        return max(self.pen(a + (b - a) * k / n) for k in range(1, n + 1))


class Judge(Node):
    def __init__(self):
        super().__init__("judge_sim", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.chk = Checker()
        self.pub = self.create_publisher(LowCmd, "/lowcmd_bare", 10)
        self.create_subscription(LowCmd, "/lowcmd", self.on_out,
                                 QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=100))
        self.out = []

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_out(self, m):
        vals = [getattr(c, f) for c in m.motor_cmd for f in ("q", "dq", "tau", "kp", "kd")]
        self.out.append((self.now(), np.array([m.motor_cmd[i].q for i in range(12)]),
                         bool(np.all(np.isfinite(vals)))))

    def spin_for(self, s):
        end = self.now() + s
        while rclpy.ok() and self.now() < end:
            rclpy.spin_once(self, timeout_sec=0.005)

    def send(self, q, sec, mut=None):
        end = self.now() + sec
        while rclpy.ok() and self.now() < end:
            m = LowCmd()
            for i in range(12):
                m.motor_cmd[i].mode = 10
                m.motor_cmd[i].q = float(q[i])
                m.motor_cmd[i].kp = 40.0
                m.motor_cmd[i].kd = 1.0
            if mut:
                setattr(m.motor_cmd[mut[1]], mut[0], mut[2])
            self.pub.publish(m)
            self.spin_for(0.02)

    def since(self, i0):
        return self.out[i0:]

    def audit(self, i0):
        qs = [o[1] for o in self.out[i0:]]
        viol, worst, big = 0, 0.0, 0.0
        for a, b in zip(qs, qs[1:]):
            big = max(big, float(np.max(np.abs(b - a))))
            if np.array_equal(a, b):
                continue
            p = self.chk.path_pen(a, b)
            worst = max(worst, p)
            viol += p > 0.001
        return len(qs), int(viol), worst, math.degrees(big)

    def run(self):
        while rclpy.ok() and self.now() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
        rng = np.random.default_rng(1)
        pts = {}
        safe_t = np.clip(STAND + np.array([0.1, -0.2, 0.3] * 4), QMIN, QMAX)
        assert self.chk.pen(safe_t) <= 0.0008 and self.chk.pen(STAND) <= 0.0008

        print("1. Частота /lowcmd (окно 5 с sim-времени)")
        self.send(STAND, 1.5)
        i0 = len(self.out)
        t0 = self.now()
        self.send(STAND, 5.5)
        w = [o[0] for o in self.out[i0:] if t0 + 0.5 <= o[0] <= t0 + 5.5]
        hz = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 2 else 0
        gap = max(np.diff(w)) * 1000 if len(w) > 2 else 1e9
        print(f"   средняя {hz:.1f} Гц, макс. интервал {gap:.0f} мс")
        pts["Частота и интервал (4)"] = (hz >= 50 and gap <= 100, 4)

        print("2. Допустимые команды: выход должен прийти к цели")
        conv = []
        for tgt in (safe_t, STAND, safe_t):
            self.send(tgt, 3.0)
            conv.append(float(np.max(np.abs(self.out[-1][1] - tgt))))
        print(f"   отклонения выхода от целей: {[round(c, 4) for c in conv]} рад")
        ok_valid = max(conv) < 0.01

        print("3. NaN/Inf: сообщение отклоняется целиком")
        self.send(STAND, 2.0)
        i0 = len(self.out)
        muts = [("q", 3, NAN), ("q", 5, INF), ("dq", 2, -INF), ("tau", 7, NAN),
                ("kp", 1, INF), ("kd", 11, NAN)]
        for mut in muts:
            self.send(safe_t, 0.3, mut)
        self.spin_for(0.3)
        seg = self.since(i0)
        nonfinite = sum(not o[2] for o in seg)
        moved = max(float(np.max(np.abs(o[1] - STAND))) for o in seg)
        print(f"   нечисловых значений на выходе: {nonfinite}, макс. сдвиг цели от прежней: {moved:.4f} рад")
        pts["Нет NaN/Inf на выходе (2)"] = (nonfinite == 0 and moved < 1e-6, 2)
        self.send(STAND, 1.0, ("q", 15, NAN))
        print(f"   NaN в неиспользуемом элементе 15: нечисловых на выходе {sum(not o[2] for o in self.since(i0))}")

        print("4. Потеря входа: удержание последней цели через 0,5 с")
        self.send(safe_t, 3.0)
        i0 = len(self.out)
        ts = self.now()
        self.spin_for(3.0)
        seg = [o for o in self.since(i0) if o[0] >= ts + 0.5]
        dev = max(float(np.max(np.abs(o[1] - safe_t))) for o in seg) if seg else 9
        g = max(np.diff([o[0] for o in self.since(i0)])) * 1000 if len(self.since(i0)) > 2 else 1e9
        print(f"   сообщений {len(seg)}, отклонение от цели {dev:.4f} рад, макс. интервал {g:.0f} мс")
        pts["Удержание при потере входа (2)"] = (len(seg) > 0 and dev < 0.002 and g <= 100, 2)

        print("5. Пределы суставов: заведомо недопустимые цели")
        i0 = len(self.out)
        self.send(np.array([3.0, 9.0, 1.0] * 4), 2.0)
        for _ in range(100):
            self.send(rng.uniform(-6, 6, 12), 0.02)
        self.spin_for(0.3)
        qs = np.array([o[1] for o in self.since(i0)])
        out_lim = int(np.sum((qs < QMIN) | (qs > QMAX)))
        print(f"   значений вне диапазона: {out_lim}")
        pts["Угловые ограничения (2)"] = (out_lim == 0, 2)

        print("6. Самоколлизии (независимая проверка, шаг 0,1°)")
        self.send(STAND, 4.0)
        bad = []
        for _ in range(20000):
            q = rng.uniform(QMIN, QMAX)
            if self.chk.pen(q) > 0.003:
                bad.append(q)
            if len(bad) >= 8:
                break
        pairs = []
        for _ in range(400):
            a = np.clip(STAND + rng.uniform(-0.6, 0.6, 12), QMIN + 1e-3, QMAX - 1e-3)
            b = np.clip(STAND + rng.uniform(-0.6, 0.6, 12), QMIN + 1e-3, QMAX - 1e-3)
            if self.chk.pen(a) <= 0.0008 and self.chk.pen(b) <= 0.0008 and self.chk.path_pen(a, b) > 0.001:
                pairs.append((a, b))
            if len(pairs) >= 3:
                break
        print(f"   опасных целей: {len(bad)}, пар с коллизией на пути: {len(pairs)}")
        i0 = len(self.out)
        for q in bad:
            self.send(q, 0.5)
        for a, b in pairs:
            self.send(a, 2.0)
            self.send(b, 2.0)
        n, viol, worst, big = self.audit(i0)
        print(f"   сообщений {n}, нарушений {viol}, макс. проникновение {worst * 1000:.2f} мм, "
              f"макс. шаг {big:.2f}°")
        pts["Самоколлизии и обработка команд (5)"] = (viol == 0 and big <= 5.0 and ok_valid, 5)

def main():
    rclpy.init()
    j = Judge()
    j.run()
    j.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
