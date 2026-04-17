from maix import camera, display, image, nn, app, uart, pinmap
from math import sqrt
import time


TARGET_IDS = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
    13,  # bench
    24,  # backpack
    28,  # suitcase
    56,  # chair
    57,  # couch
    58,  # potted plant
    60,  # dining table
}

LABELS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
    13: "bench",
    24: "backpack",
    28: "suitcase",
    56: "chair",
    57: "couch",
    58: "plant",
    60: "table",
}

# 视觉危险权重（供“主目标选择”使用，不是 DS 的最终权重）
CLASS_PRIORITY = {
    0: 1.0,
    1: 0.9,
    2: 1.0,
    3: 1.0,
    5: 1.0,
    7: 1.0,
    9: 0.3,
    11: 0.4,
    13: 0.8,
    24: 0.6,
    28: 0.6,
    56: 0.8,
    57: 0.8,
    58: 0.5,
    60: 0.7,
}

CONF_TH = 0.45
MATCH_DIST_TH = 80
OUTLIER_DIST_TH = 100
MAX_MISSED = 5
WH_SMOOTH = 0.7

# -----------------------------
# 频率控制（解决相机和 ToF 不同频率问题）
# -----------------------------
# MaixCam 只按固定频率往 STM32 发结果，不每帧都发。
# 例如 100ms -> 10Hz。STM32 可用 20Hz 左右跑 ToF 和融合。
REPORT_INTERVAL_MS = 100

# -----------------------------
# UART 配置
# 推荐：如果引脚够用，优先走 UART1（A19 TX / A18 RX），避免 UART0 启动日志干扰。
# 如果你已经把 STM32 接到 A16/A17，也可以把 USE_UART1 改成 False。
# -----------------------------
USE_UART1 = True
BAUD = 115200


def init_uart():
    try:
        if USE_UART1:
            pinmap.set_pin_function("A18", "UART1_RX")
            pinmap.set_pin_function("A19", "UART1_TX")
            device = "/dev/ttyS1"
        else:
            # UART0 默认可用，对应 A16/A17
            device = "/dev/ttyS0"
        serial_dev = uart.UART(device, BAUD)
        print("UART ready:", device, BAUD)
        return serial_dev
    except Exception as e:
        print("UART init failed:", e)
        return None


def center_of_obj(obj):
    return obj.x + obj.w / 2.0, obj.y + obj.h / 2.0


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


class Kalman1D:
    """
    一维常速度卡尔曼滤波:
    state = [pos, vel]
    measurement = pos
    """
    def __init__(self, process_var=300.0, measure_var=25.0):
        self.x = 0.0
        self.v = 0.0

        self.P00 = 1000.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 1000.0

        self.q = process_var
        self.r = measure_var
        self.initialized = False

    def init(self, value):
        self.x = float(value)
        self.v = 0.0
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return self.x

        self.x = self.x + dt * self.v

        q00 = 0.25 * self.q * dt * dt * dt * dt
        q01 = 0.5 * self.q * dt * dt * dt
        q10 = q01
        q11 = self.q * dt * dt

        P00 = self.P00 + dt * (self.P10 + self.P01) + dt * dt * self.P11 + q00
        P01 = self.P01 + dt * self.P11 + q01
        P10 = self.P10 + dt * self.P11 + q10
        P11 = self.P11 + q11

        self.P00, self.P01, self.P10, self.P11 = P00, P01, P10, P11
        return self.x

    def update(self, z):
        if not self.initialized:
            self.init(z)
            return

        z = float(z)
        y = z - self.x
        S = self.P00 + self.r

        K0 = self.P00 / S
        K1 = self.P10 / S

        P00_old = self.P00
        P01_old = self.P01
        P10_old = self.P10
        P11_old = self.P11

        self.x = self.x + K0 * y
        self.v = self.v + K1 * y

        self.P00 = (1.0 - K0) * P00_old
        self.P01 = (1.0 - K0) * P01_old
        self.P10 = P10_old - K1 * P00_old
        self.P11 = P11_old - K1 * P01_old


class Track:
    def __init__(self, obj, now, track_id):
        cx, cy = center_of_obj(obj)

        self.id = track_id
        self.class_id = obj.class_id
        self.score = obj.score

        self.kx = Kalman1D()
        self.ky = Kalman1D()
        self.kx.init(cx)
        self.ky.init(cy)

        self.cx = cx
        self.cy = cy
        self.w = float(obj.w)
        self.h = float(obj.h)

        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.acc = 0.0
        self.prev_speed = 0.0

        self.last_area = max(1.0, self.w * self.h)
        self.area_ratio = 0.0
        self.motion_z = "stable"
        self.motion_code = 0  # 0 stable, 1 approaching, 2 leaving

        self.last_t = now
        self.missed = 0
        self.hit_streak = 1
        self.age = 1

    def predict(self, now):
        dt = max(0.01, now - self.last_t)
        self.cx = self.kx.predict(dt)
        self.cy = self.ky.predict(dt)
        self.vx = self.kx.v
        self.vy = self.ky.v
        self.speed = sqrt(self.vx * self.vx + self.vy * self.vy)
        return self.cx, self.cy

    def update(self, obj, now):
        dt = max(0.01, now - self.last_t)
        meas_cx, meas_cy = center_of_obj(obj)

        if dist2((self.cx, self.cy), (meas_cx, meas_cy)) > OUTLIER_DIST_TH * OUTLIER_DIST_TH:
            self.missed += 1
            self.hit_streak = 0
            self.last_t = now
            return

        self.kx.update(meas_cx)
        self.ky.update(meas_cy)

        self.cx = self.kx.x
        self.cy = self.ky.x
        self.vx = self.kx.v
        self.vy = self.ky.v

        self.speed = sqrt(self.vx * self.vx + self.vy * self.vy)
        self.acc = (self.speed - self.prev_speed) / dt
        self.prev_speed = self.speed

        self.w = WH_SMOOTH * self.w + (1.0 - WH_SMOOTH) * obj.w
        self.h = WH_SMOOTH * self.h + (1.0 - WH_SMOOTH) * obj.h

        area = max(1.0, obj.w * obj.h)
        self.area_ratio = (area - self.last_area) / self.last_area
        self.last_area = area

        if self.area_ratio > 0.10:
            self.motion_z = "approaching"
            self.motion_code = 1
        elif self.area_ratio < -0.10:
            self.motion_z = "leaving"
            self.motion_code = 2
        else:
            self.motion_z = "stable"
            self.motion_code = 0

        self.class_id = obj.class_id
        self.score = obj.score
        self.missed = 0
        self.hit_streak += 1
        self.age += 1
        self.last_t = now

    def mark_missed(self, now):
        self.missed += 1
        self.hit_streak = 0
        self.last_t = now

    def draw(self, img):
        x = int(self.cx - self.w / 2)
        y = int(self.cy - self.h / 2)
        w = int(self.w)
        h = int(self.h)

        color = image.COLOR_GREEN if self.missed == 0 else image.COLOR_YELLOW
        img.draw_rect(x, y, w, h, color=color)

        label = LABELS.get(self.class_id, str(self.class_id))
        txt = "#{} {} hs:{} {:.1f} {}".format(self.id, label, self.hit_streak, self.speed, self.motion_z)
        img.draw_string(x, max(0, y - 14), txt, color=color)


def associate_tracks(trackers, detections):
    """
    简单贪心匹配：
    - 同 class 才匹配
    - 按中心距离最近原则匹配
    """
    pairs = []

    track_ids = list(trackers.keys())
    for tid in track_ids:
        tr = trackers[tid]
        for i, obj in enumerate(detections):
            if tr.class_id != obj.class_id:
                continue
            cx, cy = center_of_obj(obj)
            d = dist2((tr.cx, tr.cy), (cx, cy))
            if d <= MATCH_DIST_TH * MATCH_DIST_TH:
                pairs.append((d, tid, i))

    pairs.sort(key=lambda x: x[0])

    used_t = set()
    used_d = set()
    matches = []

    for _, tid, i in pairs:
        if tid in used_t or i in used_d:
            continue
        used_t.add(tid)
        used_d.add(i)
        matches.append((tid, i))

    unmatched_tracks = [tid for tid in track_ids if tid not in used_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in used_d]

    return matches, unmatched_tracks, unmatched_dets


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def track_priority(tr, img_w, img_h):
    """
    用于决定“发给 STM32 的主目标”是谁。
    这里不是最终 DS 公式，只是通信阶段的主目标排序。
    """
    class_p = CLASS_PRIORITY.get(tr.class_id, 0.4)
    score_p = tr.score
    area_p = clamp01((tr.w * tr.h) / 12000.0)
    dx = abs(tr.cx - img_w / 2.0) / (img_w / 2.0)
    center_p = clamp01(1.0 - dx)
    stable_p = clamp01(tr.hit_streak / 5.0)
    return class_p * score_p * area_p * center_p * stable_p


def select_main_track(trackers, img_w, img_h):
    best = None
    best_score = -1.0
    for tr in trackers.values():
        if tr.missed != 0:
            continue
        s = track_priority(tr, img_w, img_h)
        if s > best_score:
            best_score = s
            best = tr
    return best


def send_track(serial_dev, tr, now_ms):
    if serial_dev is None:
        return

    if tr is None:
        # 无目标包：detected=0，保留时间戳，STM32 可据此判断“当前无检测”
        line = "@CAM,0,-1,-1,0,0,0,0,0,0,0,0,{}\r\n".format(now_ms)
    else:
        line = "@CAM,1,{},{},{:.3f},{},{},{},{},{},{:.2f},{},{}\r\n".format(
            tr.id,
            tr.class_id,
            tr.score,
            int(tr.cx),
            int(tr.cy),
            int(tr.w),
            int(tr.h),
            tr.hit_streak,
            tr.speed,
            tr.motion_code,
            now_ms,
        )
    try:
        serial_dev.write_str(line)
    except Exception as e:
        print("UART write failed:", e)


# -----------------------------
# 模型与设备
# -----------------------------
detector = nn.YOLO11(model="/root/models/yolo11n.mud", dual_buff=True)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()
serial_dev = init_uart()

trackers = {}
next_track_id = 1
last_report_ms = 0

while not app.need_exit():
    img = cam.read()
    raw_objs = detector.detect(img)
    now = time.time()
    now_ms = int(now * 1000)

    # 1) 过滤类别和置信度
    detections = []
    for obj in raw_objs:
        if obj.score < CONF_TH:
            continue
        if obj.class_id not in TARGET_IDS:
            continue
        detections.append(obj)

    # 2) 所有 track 先做预测
    for tr in trackers.values():
        tr.predict(now)

    # 3) 匹配
    matches, unmatched_tracks, unmatched_dets = associate_tracks(trackers, detections)

    # 4) 更新已匹配 track
    for tid, det_idx in matches:
        trackers[tid].update(detections[det_idx], now)

    # 5) 未匹配 track：视为临时丢框
    for tid in unmatched_tracks:
        trackers[tid].mark_missed(now)

    # 6) 未匹配 detection：新建 track
    for det_idx in unmatched_dets:
        trackers[next_track_id] = Track(detections[det_idx], now, next_track_id)
        next_track_id += 1

    # 7) 删除长期丢失的 track
    dead_ids = []
    for tid, tr in trackers.items():
        if tr.missed > MAX_MISSED:
            dead_ids.append(tid)
    for tid in dead_ids:
        del trackers[tid]

    # 8) 选择主目标并定频发送到 STM32
    if now_ms - last_report_ms >= REPORT_INTERVAL_MS:
        best_tr = select_main_track(trackers, img.width(), img.height())
        send_track(serial_dev, best_tr, now_ms)
        last_report_ms = now_ms

    # 9) 画出来
    for tr in trackers.values():
        tr.draw(img)

    main_tr = select_main_track(trackers, img.width(), img.height())
    if main_tr is not None:
        dbg = "UART:{}Hz best=#{} {}".format(int(1000 / REPORT_INTERVAL_MS), main_tr.id, LABELS.get(main_tr.class_id, str(main_tr.class_id)))
    else:
        dbg = "UART:{}Hz best=None".format(int(1000 / REPORT_INTERVAL_MS))
    img.draw_string(4, 4, dbg, color=image.COLOR_RED)

    disp.show(img)
