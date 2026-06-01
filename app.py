import streamlit as st

st.set_page_config(page_title="AI 肩頸糾正助理", layout="wide")

st.title("🤖 AI 肩頸烏龜頸與駝背糾正助理 (WebRTC+MediaPipe 終極流暢版)")
st.markdown("---")

# =========================================================
# 🎯 核心黑科技：純前端 JavaScript 處理所有影像與音效 (零後端衝突)
# =========================================================
html_code = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
    <style>
        .container { display: flex; flex-direction: column; align-items: center; font-family: sans-serif; color: #333; }
        .video-box { position: relative; width: 640px; height: 480px; background: #000; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
        #webcam { transform: scaleX(-1); width: 640px; height: 480px; display: none; }
        #output_canvas { position: absolute; left: 0; top: 0; transform: scaleX(-1); width: 640px; height: 480px; }
        .dashboard { width: 640px; margin-top: 15px; display: flex; justify-content: space-between; background: #f0f2f6; padding: 15px; border-radius: 8px; box-sizing: border-box; }
        .status-card { text-align: center; flex: 1; margin: 0 5px; background: white; padding: 10px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .status-val { font-size: 1.2rem; font-weight: bold; margin-top: 5px; }
        #btn-ctrl { background: #ff4b4b; color: white; border: none; padding: 12px 30px; font-size: 1rem; border-radius: 8px; cursor: pointer; font-weight: bold; width: 640px; margin-top: 10px; transition: 0.3s; }
        #btn-ctrl:hover { background: #e03e3e; }
        
        /* Toast 樣式 */
        #toast { visibility: hidden; min-width: 300px; background-color: #333; color: #fff; text-align: center; border-radius: 8px; padding: 16px; position: fixed; z-index: 999; left: 50%; bottom: 30px; transform: translateX(-50%); font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.4); border-left: 6px solid #ff4b4b; }
        #toast.show { visibility: visible; -webkit-animation: fadein 0.5s, fadeout 0.5s 2.5s; animation: fadein 0.5s, fadeout 0.5s 2.5s; }
        @-webkit-keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 30px; opacity: 1;} }
        @keyframes fadein { from {bottom: 0; opacity: 0;} to {bottom: 30px; opacity: 1;} }
        @-webkit-keyframes fadeout { from {bottom: 30px; opacity: 1;} to {bottom: 0; opacity: 0;} }
        @keyframes fadeout { from {bottom: 30px; opacity: 1;} to {bottom: 0; opacity: 0;} }
    </style>
</head>
<body>

<div class="container">
    <div class="video-box">
        <video id="webcam" autoplay playsinline></video>
        <canvas id="output_canvas" width="640" height="480"></canvas>
    </div>

    <button id="btn-ctrl">🔴 啟動相機與偵測</button>

    <div class="dashboard">
        <div class="status-card">
            <div>耳肩距離 (像素)</div>
            <div id="val-dist" class="status-val" style="color: #2b5c8f;">0.0 px</div>
        </div>
        <div class="status-card">
            <div>肩頸角度 (度)</div>
            <div id="val-angle" class="status-val" style="color: #2b5c8f;">0.0°</div>
        </div>
        <div class="status-card">
            <div>目前姿勢狀態</div>
            <div id="val-status" class="status-val" style="color: #28a745;">良好</div>
        </div>
    </div>
</div>

<div id="toast">🚨 偵測到坐姿不良！請立刻修正！</div>

<script>
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    
    const btnCtrl = document.getElementById('btn-ctrl');
    const txtDist = document.getElementById('val-dist');
    const txtAngle = document.getElementById('val-angle');
    const txtStatus = document.getElementById('val-status');
    const toast = document.getElementById('toast');

    let isRunning = false;
    let camera = null;
    let lastAlertTime = 0;

    // 瀏覽器語音合成發聲元件 (Web Speech API)
    function speakWarning(message) {
        let now = Date.now();
        if (now - lastAlertTime > 4000) { // 4秒冷卻，防止連續轟炸
            lastAlertTime = now;
            
            // 1. 顯示前端網頁 Toast
            toast.innerText = message;
            toast.className = "show";
            setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
            
            // 2. 瀏覽器語音提醒
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = 'zh-TW';
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    // 計算角度
    function calculateAngle(p1, p2) {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        return Math.atan2(Math.abs(dy), Math.abs(dx)) * 180.0 / Math.PI;
    }

    // MediaPipe 骨架回傳處理結果
    function onResults(results) {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // 畫入目前相機畫面
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

        if (results.poseLandmarks) {
            const landmarks = results.poseLandmarks;
            
            // 取得左耳(7)、右耳(8)、左肩(11)、右肩(12)
            const leftEar = landmarks[7];
            const rightEar = landmarks[8];
            const leftShoulder = landmarks[11];
            const rightShoulder = landmarks[12];

            // 換算成 640x480 的像素坐標
            const leftEarPx = { x: leftEar.x * 640, y: leftEar.y * 480 };
            const leftShPx = { x: leftShoulder.x * 640, y: leftShoulder.y * 480 };
            const rightEarPx = { x: rightEar.x * 640, y: rightEar.y * 480 };
            const rightShPx = { x: rightShoulder.x * 640, y: rightShoulder.y * 480 };

            // 判斷哪一側離鏡頭比較近
            let xDistance = 0;
            let angle = 0;
            let activeEar = null, activeSh = null;

            if (Math.abs(leftEarPx.x - leftShPx.x) >= Math.abs(rightEarPx.x - rightShPx.x)) {
                xDistance = leftEarPx.x - leftShPx.x;
                angle = calculateAngle(leftEarPx, leftShPx);
                activeEar = leftEarPx; activeSh = leftShPx;
            } else {
                xDistance = rightEarPx.x - rightShPx.x;
                angle = calculateAngle(rightEarPx, rightShPx);
                activeEar = rightEarPx; activeSh = rightShPx;
            }

            // 繪製骨架連線與節點
            if (activeEar && activeSh) {
                canvasCtx.beginPath();
                canvasCtx.moveTo(activeEar.x, activeEar.y);
                canvasCtx.lineTo(activeSh.x, activeSh.y);
                canvasCtx.strokeStyle = '#00ff00';
                canvasCtx.lineWidth = 4;
                canvasCtx.stroke();

                canvasCtx.beginPath();
                canvasCtx.arc(activeEar.x, activeEar.y, 8, 0, 2 * Math.PI);
                canvasCtx.arc(activeSh.x, activeSh.y, 8, 0, 2 * Math.PI);
                canvasCtx.fillStyle = '#ff0000';
                canvasCtx.fill();
            }

            // 更新儀表板數據
            txtDist.innerText = Math.abs(xDistance).toFixed(1) + " px";
            txtAngle.innerText = angle.toFixed(1) + "°";

            // 判定姿勢
            if (xDistance > 60) {
                txtStatus.innerText = "⚠️ 烏龜頸警告";
                txtStatus.style.color = "#ff9800";
                speakWarning("🚨 偵測到烏龜頸！請收下巴！");
            } else if (angle < 55) {
                txtStatus.innerText = "🚨 駝背嚴重";
                txtStatus.style.color = "#f44336";
                speakWarning("🚨 偵測到駝背！請挺胸坐直！");
            } else {
                txtStatus.innerText = "良好";
                txtStatus.style.color = "#28a745";
            }
        }
        canvasCtx.restore();
    }

    // 初始化 MediaPipe Pose 模型
    const pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({
        modelComplexity: 0, // 使用最輕量模型確保極致流暢度
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    pose.onResults(onResults);

    // 控制按鈕點擊事件 (WebRTC 喚醒相機)
    btnCtrl.addEventListener('click', async () => {
        if (isRunning) {
            // 停止
            isRunning = false;
            btnCtrl.innerText = "🔴 啟動相機與偵測";
            btnCtrl.style.background = "#ff4b4b";
            if (camera) { await camera.stop(); }
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        } else {
            // 啟動
            isRunning = true;
            btnCtrl.innerText = "⏹️ 停止偵測";
            btnCtrl.style.background = "#4b79ff";
            
            camera = new Camera(videoElement, {
                onFrame: async () => {
                    if(isRunning) await pose.send({image: videoElement});
                },
                width: 640,
                height: 480
            });
            camera.start().catch(err => {
                alert("相機開啟失敗，請確認是否給予網頁相機權限！");
                isRunning = false;
                btnCtrl.innerText = "🔴 啟動相機與偵測";
                btnCtrl.style.background = "#ff4b4b";
            });
        }
    });
</script>
</body>
</html>
"""

# 用 streamlit 元件包裹，只渲染一次
st.components.v1.html(html_code, height=650, scroller=False)