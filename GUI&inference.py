import tkinter as tk
import customtkinter as ctk

import json
import cv2
import numpy as np
import mediapipe as mp

from collections import deque
from keras.models import load_model
from PIL import Image

import threading

# ─────────────────────────────────────────────
#  GLOBAL THEME
# ─────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BAR_COLORS = [
    "#00C8FF", "#32DC50", "#FF7832", "#6450FF",
    "#FFC800", "#B400C8", "#00B4FF", "#50FFC8",
]

#  MEDIAPIPE FEATURE EXTRACTION
def _dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(r * 180 / np.pi)
    return ang if ang <= 180 else 360 - ang


def extract_features(results):
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark

    def pt(i): return [lm[i].x, lm[i].y]

    hip_l, hip_r = pt(23), pt(24)
    center = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]
    shl, shr = pt(11), pt(12)
    ell, elr = pt(13), pt(14)
    wrl, wrr = pt(15), pt(16)
    knl, knr = pt(25), pt(26)
    akl, akr = pt(27), pt(28)
    mol, mor = pt(9),  pt(10)
    mc = [(mol[0] + mor[0]) / 2, (mol[1] + mor[1]) / 2]
    ts = _dist([(shl[0] + shr[0]) / 2, (shl[1] + shr[1]) / 2], center) or 1

    feats = []
    for l in lm:
        feats.extend([(l.x - center[0]) / ts, (l.y - center[1]) / ts, l.z / ts, l.visibility])
    angles = [
        _angle(shl, hip_l, knl), _angle(shr, hip_r, knr),
        _angle(hip_l, knl, akl), _angle(hip_r, knr, akr),
        _angle(shl, ell, wrl),   _angle(shr, elr, wrr)
    ]
    dists = [_dist(wrl, wrr), _dist(wrl, mc), _dist(wrr, mc)]
    feats.extend([a / 180 for a in angles])
    feats.extend([d / ts  for d in dists])
    return feats


# ─────────────────────────────────────────────
#  BAR CHART CANVAS
# ─────────────────────────────────────────────
class BarChartCanvas(tk.Canvas):
    def __init__(self, master, label_names: dict, **kwargs):
        kwargs.setdefault("bg", "#0D0D0D")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)
        self.label_names = label_names
        self.n      = len(label_names)
        self.probs  = [0.0] * self.n
        self.active = -1
        self.bind("<Configure>", lambda e: self._draw())

    def update_probs(self, probs, active_idx):
        self.probs  = list(probs)
        self.active = active_idx
        self._draw()

    def reset(self):
        """Xóa chart về trạng thái ban đầu"""
        self.probs  = [0.0] * self.n
        self.active = -1
        self._draw()

    @staticmethod
    def _dim(hex_color, factor=0.32):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"

    def _draw(self):
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 10 or H < 10 or self.n == 0:
            return

        pad_l, pad_r, pad_top, pad_bot = 10, 10, 25, 35
        chart_h = H - pad_top - pad_bot
        chart_w = W - pad_l - pad_r
        gap   = max(4, chart_w // (self.n * 5))
        bar_w = (chart_w - gap * (self.n + 1)) // self.n

        for i in range(self.n):
            x0    = pad_l + gap + i * (bar_w + gap)
            x1    = x0 + bar_w
            bh    = int(self.probs[i] * chart_h)
            ybot  = pad_top + chart_h
            ytop  = ybot - bh
            color = BAR_COLORS[i % len(BAR_COLORS)]
            act   = (i == self.active)

            # Ghost background
            self.create_rectangle(x0, pad_top, x1, ybot, fill="#1A1A1A", outline="")
            # Progress fill
            if bh > 0:
                self.create_rectangle(x0, ytop, x1, ybot,
                    fill=color if act else self._dim(color), outline="")
            # Glow border
            if act and bh > 0:
                self.create_rectangle(x0-1, ytop-1, x1+1, ybot+1,
                    fill="", outline=color, width=1)

            # % label
            self.create_text((x0+x1)//2, ytop - 10,
                text=f"{self.probs[i]*100:.0f}%",
                fill=color if act else "#666",
                font=("Consolas", 8, "bold"))
            # Action label
            self.create_text((x0+x1)//2, ybot + 15,
                text=self.label_names.get(i, str(i)).upper(),
                fill="white" if act else "#555",
                font=("Consolas", 7))


# ─────────────────────────────────────────────
#  SPLASH PAGE
# ─────────────────────────────────────────────
class SplashPage(ctk.CTkFrame):
    def __init__(self, master, on_enter, **kwargs):
        super().__init__(master, fg_color="white", corner_radius=0, **kwargs)
        self.on_enter = on_enter
        self._build()

    def _build(self):
        col = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        col.place(relx=0.5, rely=0.5, anchor="center")

        def lbl(parent, text, size=11, bold=False, color="#111", wrap=440, pady=(2, 0)):
            l = ctk.CTkLabel(parent, text=text,
                             font=ctk.CTkFont("Times New Roman", size,
                                              weight="bold" if bold else "normal"),
                             text_color=color, wraplength=wrap, justify="center")
            l.pack(pady=pady)
            return l

        lbl(col, "BỘ GIÁO DỤC VÀ ĐÀO TẠO", 11)
        lbl(col, "TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM", 11, bold=True)
        ctk.CTkFrame(col, height=1, fg_color="#444", corner_radius=0).pack(fill="x", padx=50, pady=8)

        try:
            img = Image.open("HCMUTE.png").resize((100, 100), Image.LANCZOS)
            self._logo = ctk.CTkImage(img, size=(100, 100))
            ctk.CTkLabel(col, image=self._logo, text="").pack(pady=(4, 8))
        except Exception:
            lbl(col, "[ LOGO HCMUTE ]", 10, color="#999", pady=(8, 8))

        lbl(col, "KHOA ĐIỆN - ĐIỆN TỬ", 12, bold=True)
        lbl(col, "LẬP TRÌNH VỚI PYTHON", 11)
        ctk.CTkFrame(col, height=1, fg_color="#ddd", corner_radius=0).pack(fill="x", padx=30, pady=10)

        lbl(col, "ĐỀ TÀI BÁO CÁO:", 13, bold=True, color="#1565C0")
        lbl(col, "NHẬN DIỆN HÀNH VI CON NGƯỜI\nTRÊN DỮ LIỆU VIDEO REAL-TIME",
            17, bold=True, color="#C62828", pady=(6, 4))
        ctk.CTkFrame(col, height=1, fg_color="#ddd", corner_radius=0).pack(fill="x", padx=30, pady=10)

        info = ctk.CTkFrame(col, fg_color="white", corner_radius=0)
        info.pack(pady=(0, 4))

        def info_row(label, value):
            row = ctk.CTkFrame(info, fg_color="white", corner_radius=0)
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label,
                         font=ctk.CTkFont("Times New Roman", 11, weight="bold"),
                         text_color="#111", width=148, anchor="e", justify="right"
                         ).pack(side="left", padx=(0, 10))
            ctk.CTkLabel(row, text=value,
                         font=ctk.CTkFont("Times New Roman", 11),
                         text_color="#111", justify="left",
                         wraplength=260, anchor="w"
                         ).pack(side="left")

        info_row("GV Hướng dẫn:", "TS. DƯƠNG MINH THIỆN")
        info_row("Sinh viên thực hiện:",
                 "25139001 - Trần Quốc An\n"
                 "25139044 - Nguyễn Minh Tập\n"
                 "25139050 - Trần Minh Thuận\n"
                 "25139048 - Nguyễn Tuấn Thịnh\n"
                 "25139060 - Trần Quang Quốc Tùng")

        ctk.CTkFrame(col, height=1, fg_color="#ddd", corner_radius=0).pack(fill="x", padx=30, pady=14)

        ctk.CTkButton(
            col, text="TRUY CẬP HỆ THỐNG",
            font=ctk.CTkFont("Times New Roman", 13, weight="bold"),
            height=48, corner_radius=24,
            fg_color="#1565C0", hover_color="#1E88E5",
            command=self.on_enter
        ).pack(ipadx=24)

        lbl(col, "TP. HỒ CHÍ MINH, THÁNG 05/2026", 10, bold=True, color="#444", pady=(14, 4))


# ─────────────────────────────────────────────
#  CONTROL PAGE
# ─────────────────────────────────────────────
class ControlPage(ctk.CTkFrame):
    def __init__(self, master, label_names: dict, **kwargs):
        super().__init__(master, fg_color="#111", corner_radius=0, **kwargs)
        self.label_names = label_names
        self._running    = False
        self._stop_event = threading.Event()

        # MediaPipe
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Model
        try:
            self._model = load_model("GRU_action_model.keras")
        except Exception as e:
            print(f"[ERROR] Load model thất bại: {e}")
            self._model = None

        self._seq_buf   = deque(maxlen=20)
        self._prob_hist = deque(maxlen=3)   # BUG FIX: 5 → 3 (response nhanh hơn)
        self._draw_skel = True
        self.ctk_img    = None              # giữ reference tránh GC xóa ảnh

        self._build()

    # ── UI ─────────────────────────────────────────────
    def _build(self):
        self.columnconfigure(0, weight=2)   # camera
        self.columnconfigure(1, weight=1)   # panel
        self.rowconfigure(0, weight=1)

        # LEFT — camera
        cam_bg = ctk.CTkFrame(self, fg_color="#000000", corner_radius=0)
        cam_bg.grid(row=0, column=0, sticky="nsew")

        self.cam_label = ctk.CTkLabel(
            cam_bg,
            text="Camera chưa được khởi động\n\nNhấn  START SYSTEM  để bắt đầu",
            font=ctk.CTkFont("Consolas", 13), text_color="#383838"
        )
        self.cam_label.pack(expand=True, fill="both")

        # RIGHT — panel
        panel = ctk.CTkFrame(self, fg_color="#181818",
                             corner_radius=0, border_width=1, border_color="#252525")
        panel.grid(row=0, column=1, sticky="nsew")

        # ── Pack bottom-up: button & toggle trước để giữ chỗ ──
        # Start/Stop button
        self.start_btn = ctk.CTkButton(
            panel, text="START SYSTEM",
            font=ctk.CTkFont("Consolas", 12, weight="bold"),
            height=45, fg_color="#00C853",
            hover_color="#00E676", text_color="#000",
            command=self._toggle_system
        )
        self.start_btn.pack(side="bottom", fill="x", padx=15, pady=(0, 16))

        # Skeleton toggle
        self.skel_switch = ctk.CTkSwitch(panel, text="Draw Skeleton",
                                          command=self._on_toggle)
        self.skel_switch.select()
        self.skel_switch.pack(side="bottom", pady=(0, 4))

        # ── Pack top-down: title & action box ──
        ctk.CTkLabel(panel, text="HAR SYSTEM  V3.0",
                     font=ctk.CTkFont("Consolas", 14, weight="bold"),
                     text_color="#2979FF"
                     ).pack(pady=(20, 10))

        # Action box
        abox = ctk.CTkFrame(panel, fg_color="#0D0D0D", corner_radius=8,
                            border_width=1, border_color="#252525")
        abox.pack(fill="x", padx=15, pady=(0, 10))

        self.action_name = ctk.CTkLabel(
            abox, text="—",
            font=ctk.CTkFont("Consolas", 28, weight="bold"),
            text_color="#00C8FF"
        )
        self.action_name.pack(pady=(10, 0))

        self.action_conf = ctk.CTkLabel(
            abox, text="waiting...",
            font=ctk.CTkFont("Consolas", 10), text_color="#484848"
        )
        self.action_conf.pack(pady=(0, 10))

        ctk.CTkLabel(panel, text="ACTIVITY PROBABILITY",
                     font=ctk.CTkFont("Consolas", 9), text_color="#555"
                     ).pack(anchor="w", padx=20, pady=(0, 4))

        # ── Bar chart — fill toàn bộ vùng còn lại ──
        chart_bg = ctk.CTkFrame(panel, fg_color="#0D0D0D", corner_radius=8,
                                border_width=1, border_color="#252525")
        chart_bg.pack(fill="both", expand=True, padx=15, pady=(0, 8))

        self.bar_chart = BarChartCanvas(chart_bg, self.label_names)
        self.bar_chart.pack(fill="both", expand=True, padx=5, pady=5)

    # ── callbacks ──────────────────────────────────────
    def _on_toggle(self):
        self._draw_skel = bool(self.skel_switch.get())

    def _toggle_system(self):
        if not self._running:
            self._start()
        else:
            self._stop()

    def _start(self):
        if self._model is None:
            print("[ERROR] Không có model, không thể start")
            return

        # BUG FIX: clear buffer cũ mỗi lần start lại
        # Nếu không clear, lần start thứ 2 trở đi sẽ predict ngay
        # bằng data của session trước → sai hoàn toàn
        self._seq_buf.clear()
        self._prob_hist.clear()

        self._running = True
        self._stop_event.clear()
        self.start_btn.configure(text="STOP SYSTEM",
                                  fg_color="#D50000", hover_color="#FF1744",
                                  text_color="#FFF")
        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self._running = False
        self._stop_event.set()
        self.start_btn.configure(text="START SYSTEM",
                                  fg_color="#00C853", hover_color="#00E676",
                                  text_color="#000")
        # BUG FIX: reset UI về trạng thái ban đầu
        self.cam_label.configure(image=None,
            text="Camera chưa được khởi động\n\nNhấn  START SYSTEM  để bắt đầu")
        self.action_name.configure(text="—", text_color="#00C8FF")
        self.action_conf.configure(text="waiting...")
        self.bar_chart.reset()  # BUG FIX: chart cũ vẫn hiện nếu không reset
        self.ctk_img = None

    # ── camera loop (background thread) ───────────────
    def _loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror — tự nhiên hơn cho người dùng
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = self._pose.process(rgb)

            if res.pose_landmarks:
                if self._draw_skel:
                    self._mp_draw.draw_landmarks(
                        frame, res.pose_landmarks,
                        self._mp_pose.POSE_CONNECTIONS,
                        self._mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                        self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                feats = extract_features(res)
                if feats:
                    self._seq_buf.append(feats)

                    if len(self._seq_buf) == 20:
                        inp    = np.array(self._seq_buf)[np.newaxis, ...]
                        raw    = self._model.predict(inp, verbose=0)[0]
                        self._prob_hist.append(raw)
                        smooth = np.mean(self._prob_hist, axis=0)
                        idx    = int(np.argmax(smooth))
                        # Cập nhật UI an toàn qua main thread
                        self.after(0, self.bar_chart.update_probs, smooth.tolist(), idx)
                        self.after(0, self._update_action,
                                   self.label_names[idx], float(smooth[idx]))

            # Render frame lên UI
            self._render_frame(frame)

        cap.release()

    def _render_frame(self, frame):
        """Chuyển OpenCV frame → CTkImage → cập nhật label (gọi từ background thread)"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w   = self.cam_label.winfo_width()
        h   = self.cam_label.winfo_height()
        if w > 10 and h > 10:
            img.thumbnail((w, h), Image.LANCZOS)

        # BUG FIX: tạo CTkImage trong background thread là OK (PIL thuần)
        # nhưng PHẢI gán self.ctk_img TRƯỚC khi gọi after()
        # để tránh race condition: after() chạy ngay trước khi ctk_img được gán
        new_img      = ctk.CTkImage(light_image=img, dark_image=img,
                                    size=(img.width, img.height))
        self.ctk_img = new_img  # giữ reference, tránh GC xóa ảnh
        self.after(0, self._set_frame)

    def _set_frame(self):
        """Chạy trên main thread — an toàn để update widget"""
        if self._running and self.ctk_img:
            self.cam_label.configure(image=self.ctk_img, text="")

    def _update_action(self, name: str, conf: float):
        self.action_name.configure(text=name.upper())
        self.action_conf.configure(text=f"CONFIDENCE:  {conf * 100:.1f}%")
        color = "#00C8FF" if conf >= 0.8 else "#FFC800" if conf >= 0.5 else "#FF5252"
        self.action_name.configure(text_color=color)

    def destroy(self):
        """BUG FIX: đảm bảo thread camera dừng khi đóng cửa sổ"""
        self._stop_event.set()
        self._running = False
        super().destroy()


# ─────────────────────────────────────────────
#  APP ENTRY
# ─────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HCMUTE — HAR Project")
        self.geometry("1400x750")
        self.minsize(1100, 560)  # tránh UI vỡ khi kéo quá nhỏ

        try:
            with open("label_map.json") as f:
                lm = json.load(f)
            self.label_names = {v: k for k, v in lm.items()}
        except FileNotFoundError:
            print("[WARN] Không tìm thấy label_map.json")
            self.label_names = {0: "Unknown"}

        self._page = None
        self._show_splash()

    def _show_splash(self):
        if self._page:
            self._page.destroy()
        self._page = SplashPage(self, on_enter=self._show_control)
        self._page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _show_control(self):
        if self._page:
            self._page.destroy()
        self._page = ControlPage(self, self.label_names)
        self._page.place(relx=0, rely=0, relwidth=1, relheight=1)


if __name__ == "__main__":
    App().mainloop()