import streamlit as st
import cv2
import time
import math
import numpy as np

# 確保 MediaPipe 引用正常
try:
    import mediapipe as mp
except AttributeError:
    pass

def calculate_line_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(abs(dy), abs(dx)) * 180.0 / math.pi

def normalized_to_pixel(norm_x, norm_y, w, h):
    return int(norm_x * w), int(norm_y * h)

# =========================
# Streamlit UI 介面
# =========================
st.set_page_config(page_title="AI 肩頸糾正助理", layout="wide")
st.title("AI 肩頸烏龜頸與駝背糾正助理 (雲端終極穩定版)")

# 初始化發聲時間紀錄
if 'last_voice_time' not in st.session_state:
    st.session_state.last_voice_time = 0.0

with st.sidebar:
    st.header("參數設定")
    DISTANCE_THRESHOLD = st.slider("烏龜頸距離閾值 (像素)", min_value=5, max_value=300, value=60, step=1, key="dist_slider")
    ANGLE_THRESHOLD = st.slider("駝背判定角度 (度)", min_value=30, max_value=90, value=55, step=1, key="angle_slider")
    DETECTION_DELAY = st.slider("姿勢錯誤緩衝時間 (秒)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, key="delay_slider")

    st.markdown("---")
    st.markdown("💡 **提示：** 此版本將所有警告字樣直接融入影像中，確保雲端執行絕對不崩潰！")

    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.button("開始 / 停止 (切換)"):
        st.session_state.running = not st.session_state.running

# 💡 全域唯一元件：整個網頁只更新這個圖片抽屜，其餘網頁結構完全不動，徹底根除前端衝突！
image_placeholder = st.empty()

# 💡 純 JavaScript 播放音效函數（不產生任何 HTML 標籤）
def play_warning_sound():
    now = time.time()
    if now - st.session_state.last_voice_time > 4.0:
        st.session_state.last_voice_time = now
        sound_url = "https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg"
        js_code = f"""
        <script>
            var audio = new Audio("{sound_url}");
            audio.play();
        </script>
        """
        # 隱藏呼叫
        st.components.v1.html(js_code, height=0, width=0)

# =========================
# 主程式執行區
# =========================
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        st.error("相機啟動失敗，請確認是否已允許網頁的相機權限。")
        st.session_state.running = False
    else:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        turtle_start = None
        hunch_start = None
        
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok or frame is None: break

            current_dist_thresh = st.session_state.dist_slider
            current_angle_thresh = st.session_state.angle_slider
            current_delay_thresh = st.session_state.delay_slider

            image_h, image_w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            show_dist = "X_dist: 0.0 px"
            show_angle = "Angle: 0.0 deg"
            status_text = "STATUS: GOOD"
            status_color = (0, 255, 0) # 綠色代表正常

            results = pose.process(image_rgb)
            if results and results.pose_landmarks and results.pose_landmarks.landmark:
                landmarks = results.pose_landmarks.landmark

                left_ear = landmarks[7]
                right_ear = landmarks[8]
                left_sh = landmarks[11]
                right_sh = landmarks[12]

                left_ear_px = normalized_to_pixel(left_ear.x, left_ear.y, image_w, image_h)
                right_ear_px = normalized_to_pixel(right_ear.x, right_ear.y, image_w, image_h)
                left_sh_px = normalized_to_pixel(left_sh.x, left_sh.y, image_w, image_h)
                right_sh_px = normalized_to_pixel(right_sh.x, right_sh.y, image_w, image_h)

                cv2.line(frame, left_ear_px, left_sh_px, (0, 255, 0), 2)
                cv2.line(frame, right_ear_px, right_sh_px, (0, 255, 0), 2)
                cv2.circle(frame, left_ear_px, 6, (0, 0, 255), -1)
                cv2.circle(frame, left_sh_px, 6, (0, 0, 255), -1)

                left_x_diff = left_ear_px[0] - left_sh_px[0]
                right_x_diff = right_ear_px[0] - right_sh_px[0]

                if abs(left_x_diff) >= abs(right_x_diff):
                    current_x_distance = left_x_diff
                    current_angle = calculate_line_angle(left_ear_px, left_sh_px)
                else:
                    current_x_distance = right_x_diff
                    current_angle = calculate_line_angle(right_ear_px, right_sh_px)

                turtle_detected = (current_x_distance > current_dist_thresh)
                hunch_detected = (current_angle < current_angle_thresh)

                now = time.time()
                
                # 💡 這裡改用 OpenCV 繪製警告字樣，不再呼叫 st.warning() 變動網頁
                if turtle_detected:
                    if turtle_start is None: turtle_start = now
                    elif (now - turtle_start) >= current_delay_thresh:
                        status_text = "WARNING: TURTLE NECK!"
                        status_color = (0, 165, 255) # 橘色警告
                        play_warning_sound()
                else:
                    turtle_start = None

                if hunch_detected:
                    if hunch_start is None: hunch_start = now
                    elif (now - hunch_start) >= current_delay_thresh:
                        status_text = "ALERT: HUNCHBACK!"
                        status_color = (0, 0, 255) # 紅色嚴重警告
                        play_warning_sound()
                else:
                    hunch_start = None

                show_dist = f"X_dist: {current_x_distance:.1f} px / Thresh: {current_dist_thresh}"
                show_angle = f"Hunch Angle: {current_angle:.1f} deg / Thresh: {current_angle_thresh}"

            # 繪製黑底狀態欄
            cv2.rectangle(frame, (10, 10), (550, 120), (0, 0, 0), -1)
            cv2.putText(frame, show_dist, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, show_angle, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_rgb, channels="RGB")
            time.sleep(0.01)
        cap.release()
else:
    image_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")