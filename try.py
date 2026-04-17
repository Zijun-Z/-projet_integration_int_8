from maix import camera, display, image, nn, app
from math import sqrt
import time

# -----------------------------
# 你真正需要的类别过滤
# -----------------------------
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

CONF_TH = 0.45
MATCH_DIST_TH = 80       
OUTLIER_DIST_TH = 100    
MAX_MISSED = 5           
WH_SMOOTH = 0.7          


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
        self.x = 0.0   # pos
        self.v = 0.0   # vel

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

        # 状态预测
        self.x = self.x + dt * self.v

        # 协方差预测
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
        y = z - self.x                # innovation
        S = self.P00 + self.r         # innovation covariance

        K0 = self.P00 / S
        K1 = self.P10 / S

        P00_old = self.P00
        P01_old = self.P01
        P10_old = self.P10
        P11_old = self.P11

        # 状态更新
        self.x = self.x + K0 * y
        self.v = self.v + K1 * y

        # 协方差更新
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

        self.last_t = now
        self.missed = 0

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

        # 异常点拒绝：测量和预测差太大则不信这一帧
        if dist2((self.cx, self.cy), (meas_cx, meas_cy)) > OUTLIER_DIST_TH * OUTLIER_DIST_TH:
            self.missed += 1
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

        # 宽高平滑，不建议直接对 class_id 做滤波
        self.w = WH_SMOOTH * self.w + (1.0 - WH_SMOOTH) * obj.w
        self.h = WH_SMOOTH * self.h + (1.0 - WH_SMOOTH) * obj.h

        area = max(1.0, obj.w * obj.h)
        self.area_ratio = (area - self.last_area) / self.last_area
        self.last_area = area

        if self.area_ratio > 0.10:
            self.motion_z = "approaching"
        elif self.area_ratio < -0.10:
            self.motion_z = "leaving"
        else:
            self.motion_z = "stable"

        self.class_id = obj.class_id
        self.score = obj.score
        self.missed = 0
        self.last_t = now

    def mark_missed(self, now):
        self.missed += 1
        self.last_t = now

    def draw(self, img):
        x = int(self.cx - self.w / 2)
        y = int(self.cy - self.h / 2)
        w = int(self.w)
        h = int(self.h)

        color = image.COLOR_GREEN if self.missed == 0 else image.COLOR_YELLOW
        img.draw_rect(x, y, w, h, color=color)

        label = LABELS.get(self.class_id, str(self.class_id))
        txt = "#{} {} {:.1f} {}".format(self.id, label, self.speed, self.motion_z)
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


# -----------------------------
# 模型与设备
# -----------------------------
detector = nn.YOLO11(model="/root/models/yolo11n.mud", dual_buff=True)
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

trackers = {}
next_track_id = 1

while not app.need_exit():
    img = cam.read()
    raw_objs = detector.detect(img)
    now = time.time()

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

    # 8) 画出来
    for tr in trackers.values():
        tr.draw(img)

    disp.show(img)