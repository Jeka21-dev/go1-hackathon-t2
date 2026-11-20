#!/usr/bin/env python3
"""T2 Safety Block: /lowcmd_bare -> проверка -> /lowcmd (Unitree Go1, MuJoCo Lowlevel)."""
import csv
import math
import os
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from unitree_go.msg import LowCmd, LowState

try:
    import mujoco
except ImportError: 
    mujoco = None

# Порядок: FR, FL, RR, RL; в каждой ноге hip, thigh, calf
Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)
LIMIT_MARGIN = 1e-3                      
STAND = np.array([0.0, 0.9, -1.8] * 4)   
JOINT_NAMES = [f"{leg}_{j}_joint" for leg in ("FR", "FL", "RR", "RL")
               for j in ("hip", "thigh", "calf")]


class CollisionChecker:
    """Самоколлизии по MuJoCo: глубина проникновения пар из t2_collision_pairs.csv."""

    def __init__(self, model_path, pairs_path, resolution_deg, log):
        self.ok = False
        self.res = math.radians(resolution_deg)
        if mujoco is None or not os.path.exists(model_path):
            log.error(f"MuJoCo или модель недоступны ({model_path}): самоколлизии НЕ проверяются")
            return
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.qadr = [int(self.m.joint(n).qposadr[0]) for n in JOINT_NAMES]
        self.allowed = self._load_pairs(pairs_path, log)
        self.ok = True

    def _load_pairs(self, path, log):
        if not os.path.exists(path):
          log.warn("CSV с парами не найден: проверяются все пары, кроме контактов с миром")
          return None
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
        if "geom1_id" not in cols or "geom2_id" not in cols:
            log.warn(f"Нет столбцов geom1_id/geom2_id (есть {cols}): проверяются все пары")
            return None
        ok_values = None
        if "diagnostic_policy" in cols:
            counts = Counter(r["diagnostic_policy"].strip() for r in rows)
            log.info(f"diagnostic_policy: {dict(counts)}")
            ok_values = {v for v, c in counts.items() if c == 655}
            if not ok_values:
                bad = ("exclu", "filter", "same", "weld", "parent", "mask")
                ok_values = {v for v in counts if not any(k in v.lower() for k in bad)}
        pairs = set()
        for r in rows:
            if ok_values is not None and r["diagnostic_policy"].strip() not in ok_values:
                continue
            a, b = int(r["geom1_id"]), int(r["geom2_id"])
            pairs.add((min(a, b), max(a, b)))
        log.info(f"Пар для проверки: {len(pairs)} (ожидается 655), geom в модели: {self.m.ngeom}")
        if len(pairs) != 655:
            log.warn("Число пар не 655: проверьте разбор CSV в _load_pairs")
        if pairs and max(max(p) for p in pairs) >= self.m.ngeom:
            log.warn("id геометрий из CSV выходят за число geom модели: используйте scene.xml")
        return pairs

    def penetration(self, q):
        """Максимальная глубина проникновения (м) среди проверяемых пар."""
        d, m = self.d, self.m
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            depth = -float(c.dist)
            if depth <= worst:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if m.geom_bodyid[g1] == 0 or m.geom_bodyid[g2] == 0:
                continue  
            if self.allowed is not None and (min(g1, g2), max(g1, g2)) not in self.allowed:
                continue
            worst = depth
        return worst

    def segment_ok(self, q0, q1, tol, base_pen):
        """Прямая в пространстве суставов q0 -> q1 с шагом не более res на любой сустав."""
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / self.res)))
        limit = max(tol, base_pen)  # если старт уже в контакте, глубже не пускаем
        for k in range(1, n + 1):
            if self.penetration(q0 + (q1 - q0) * k / n) > limit:
                return False
        return True


class SafetyBlock(Node):
    def __init__(self):
        super().__init__("safety_block",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        here = os.path.dirname(os.path.abspath(__file__))
        self.declare_parameter("model_path", os.path.join(here, "unitree_go1", "scene.xml"))
        self.declare_parameter("pairs_path", os.path.join(here, "unitree_go1", "t2_collision_pairs.csv"))
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("max_step_deg", 3.0)     
        self.declare_parameter("check_res_deg", 0.1)
        self.declare_parameter("pen_tol_m", 0.0008)     
        self.declare_parameter("kp_default", 40.0)
        self.declare_parameter("kd_default", 1.0)
        self.declare_parameter("kp_max", 1000.0)
        self.declare_parameter("kd_max", 100.0)
        g = self.get_parameter
        self.step = math.radians(g("max_step_deg").value)
        self.tol = g("pen_tol_m").value
        self.kp_def, self.kd_def = g("kp_default").value, g("kd_default").value
        self.kp_max, self.kd_max = g("kp_max").value, g("kd_max").value

        self.checker = CollisionChecker(g("model_path").value, g("pairs_path").value,
                                        g("check_res_deg").value, self.get_logger())
        self.q_cur = None   
        self.q_tgt = None   # последняя принятая безопасная цель
        self.kp = np.full(12, self.kp_def)
        self.kd = np.full(12, self.kd_def)
        self.t0 = None      
        self.n_rej = 0
        self.n_acc = 0

        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(LowCmd, "/lowcmd_bare", self.on_cmd, be)
        self.create_subscription(LowState, "/lowstate", self.on_state, be)
        self.pub = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.create_timer(1.0 / g("rate_hz").value, self.tick)
        self.create_timer(5.0, self.report)

    def on_state(self, msg):
        if self.q_cur is not None:
            return
        q = np.array([msg.motor_state[i].q for i in range(12)], float)
        if np.all(np.isfinite(q)):
            self._init(q)

    def _init(self, q):
        q = np.clip(q, Q_MIN + LIMIT_MARGIN, Q_MAX - LIMIT_MARGIN)
        self.q_cur, self.q_tgt = q.copy(), q.copy()

    def on_cmd(self, msg):
        parsed = self.parse(msg)
        if parsed is None:
            self.n_rej += 1
            return
        q, kp, kd = parsed
        q = np.clip(q, Q_MIN + LIMIT_MARGIN, Q_MAX - LIMIT_MARGIN)
        if self.q_cur is None:
            return
        if self.checker.ok and self.checker.penetration(q) > self.tol:
            self.n_rej += 1  # цель с самоколлизией: отклоняем, держим прежнюю цель
            return
        self.q_tgt, self.kp, self.kd = q, kp, kd
        self.n_acc += 1

    def parse(self, msg):
        """None, если в любом управляющем поле NaN/Inf: сообщение отклоняется целиком."""
        vals = []
        for mc in msg.motor_cmd:
            vals += [mc.q, mc.dq, mc.tau, mc.kp, mc.kd]
        if not np.all(np.isfinite(np.array(vals, float))):
            return None
        q = np.array([msg.motor_cmd[i].q for i in range(12)], float)
        kp = np.array([msg.motor_cmd[i].kp for i in range(12)], float)
        kd = np.array([msg.motor_cmd[i].kd for i in range(12)], float)
        kp = np.clip(kp, 0.0, self.kp_max)
        kd = np.clip(kd, 0.0, self.kd_max)
        return q, kp, kd

    def tick(self):
        if self.q_cur is None:
            now = self.get_clock().now()
            if now.nanoseconds == 0:
                return  
            if self.t0 is None:
                self.t0 = now
            if (now - self.t0).nanoseconds > 1e9:
                self._init(STAND)  
            else:
                return
        err = self.q_tgt - self.q_cur
        m = float(np.max(np.abs(err)))
        if m > 1e-9:
            base = self.checker.penetration(self.q_cur) if self.checker.ok else 0.0
            scale = min(1.0, self.step / m)
            for _ in range(5):
                q1 = self.q_cur + err * scale
                if (not self.checker.ok) or self.checker.segment_ok(self.q_cur, q1, self.tol, base):
                    self.q_cur = q1
                    break
                scale *= 0.5  # иначе остаёмся на месте и пробуем на следующем тике
        self.publish()

    def publish(self):
        out = LowCmd()
        for i in range(12):
            mc = out.motor_cmd[i]
            mc.mode = 0x0A
            mc.q = float(self.q_cur[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(self.kp[i])
            mc.kd = float(self.kd[i])
        self.pub.publish(out)

    def report(self):
        self.get_logger().info(f"accepted={self.n_acc} rejected={self.n_rej}")


def main():
    rclpy.init()
    node = SafetyBlock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
