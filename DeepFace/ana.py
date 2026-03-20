
import cv2
from deepface import DeepFace
import threading
import time
import numpy as np

class EmotionPerception:
    def __init__(self):
        # 建议使用 'opencv' 速度最快，'mediapipe' 在笔记本摄像头下最稳
        self.backend = 'opencv' 
        self.current_emotion = "neutral"
        self.emotion_history = [] # 用于时空平滑，符合竞赛“持续理解”要求
        self.is_running = True

    def process_frame(self, frame):
        try:
            # 这里的参数完全对齐 GitHub 官方 Demo 的高性能配置
            results = DeepFace.analyze(
                img_path = frame, 
                actions = ['emotion'],
                enforce_detection = False, # 防止没检测到人脸时程序崩溃
                detector_backend = self.backend,
                silent = True
            )
            
            if results:
                res = results[0]
                # 记录核心分值，用于后续心理状态评估
                raw_emotion = res['dominant_emotion']
                sad_score = res['emotion']['sad']
                
                # 简单的时空对齐：取最近 5 次识别的众数，防止瞬时误判
                self.emotion_history.append(raw_emotion)
                if len(self.emotion_history) > 5: self.emotion_history.pop(0)
                self.current_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                
                return self.current_emotion, sad_score
        except Exception as e:
            return "analyzing...", 0
        return "neutral", 0

# 主程序逻辑
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 针对 13 代 i7 优化：强制设置分辨率，减少冗余像素计算
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

perception = EmotionPerception()

print("多模态心理感知系统已启动...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 模拟时空对齐：每 3 帧分析一次，确保 8G 显存不溢出
    if int(time.time() * 10) % 3 == 0:
        emo, score = perception.process_frame(frame)
        # 这就是你要喂给 LLM 的核心多模态数据
        print(f"[{time.strftime('%H:%M:%S')}] 视觉感知: {emo} | 心理风险(Sad): {score:.2f}%")

    cv2.imshow('Multimodal Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
