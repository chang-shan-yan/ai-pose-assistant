import streamlit as st
import cv2
import time
import math
import numpy as np
import streamlit.components.v1 as components

# 確保 MediaPipe 引用正常
try:
    import mediapipe as mp
    mp_success = True
except AttributeError:
    mp_success = False

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
st.title("AI 肩頸烏龜頸與駝背糾正助理 (雲端全功能語音版)")

# 初始化發聲時間紀錄
if 'last_voice_time' not in st.session_state:
    st.session_state.last_voice_time = 0.0

with st.sidebar:
    st.header("參數設定")
    DISTANCE_THRESHOLD = st.slider("烏龜頸距離閾值 (像素)", min_value=5, max_value=300, value=60, step=1, key="dist_slider")
    ANGLE_THRESHOLD = st.slider("駝背判定角度 (度)", min_value=30, max_value=90, value=55, step=1, key="angle_slider")
    DETECTION_DELAY = st.slider("姿勢錯誤緩衝時間 (秒)", min_value=0.1, max_value=10.0, value=1.5, step=0.1, key="delay_slider")

    st.markdown("---")
    st.markdown("💡 **語音提示已啟用：** 本版本支援雲端發聲！評審委員使用手機或筆電體驗時請開啟聲音。")

    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if st.button("開始 / 停止 (切換)"):
        st.session_state.running = not st.session_state.running

# 宣告固定網頁元件放置區（這些在迴圈中不會被重新創造）
image_placeholder = st.empty()
warning_placeholder = st.empty()

# 💡 乾淨的發聲機制：利用現成免版權罐頭音效，直接用 JS 播放，完全不更動網頁結構！
def play_warning_sound():
    now = time.time()
    # 4 秒語音冷卻機制，避免重複轟炸
    if now - st.session_state.last_voice_time > 4.0:
        st.session_state.last_voice_time = now
        
        # 這裡使用網路公開的標準嗶嗶警報聲 MP3
        sound_url = "https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg"
        js_code = f"""
        <script>
            var audio = new Audio("{sound_url}");
            audio.play();
        </script>
        """
        # 執行這行純 JS 播放聲音，因為沒有生出 HTML 標籤，所以絕對不會噴 removeChild 錯誤！
        components.html(js_code, height=0, width=0)

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
                
                if turtle_detected:
                    if turtle_start is None: turtle_start = now
                    elif (now - turtle_start) >= current_delay_thresh:
                        warning_placeholder.warning("⚠️ 偵測到烏龜頸！請稍微抬頭並收下巴。")
                        play_warning_sound()  # 💡 觸發乾淨音效
                else:
                    turtle_start = None

                if hunch_detected:
                    if hunch_start is None: hunch_start = now
                    elif (now - hunch_start) >= current_delay_thresh:
                        warning_placeholder.error("🚨 偵測到駝背！請挺胸，坐直並打開胸腔。")
                        play_warning_sound()  # 💡 觸發乾淨音效
                else:
                    hunch_start = None

                if not turtle_detected and not hunch_detected:
                    warning_placeholder.empty()

                show_dist = f"X_dist: {current_x_distance:.1f} px / Thresh: {current_dist_thresh}"
                show_angle = f"Hunch Angle: {current_angle:.1f} deg / Thresh: {current_angle_thresh}"

            cv2.rectangle(frame, (10, 10), (450, 85), (0, 0, 0), -1)
            cv2.putText(frame, show_dist, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, show_angle, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_rgb, channels="RGB")
            time.sleep(0.01)
        cap.release()
else:
    image_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")
    warning_placeholder.empty()