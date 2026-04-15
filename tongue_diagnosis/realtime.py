"""
实时舌诊分析 - 摄像头版本
按空格键暂停并分析当前画面，再按空格恢复实时画面
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict
from datetime import datetime

from digital_human import TongueDetectionError, get_tongue_adapter


class RealTimeTongueAnalysis:
    """实时舌诊分析"""

    # 颜色配置 (BGR for OpenCV)
    COLORS = {
        'box': (0, 255, 0),           # 绿色检测框
        'mask': (255, 0, 255),         # 紫色分割掩码
        'text_bg': (0, 0, 0),          # 黑色文字背景
        'text': (255, 255, 255),       # 白色文字
        'paused_bg': (0, 100, 255),    # 暂停状态蓝色背景
    }

    def __init__(
        self,
        yolo_path: str = None,
        sam_path: str = None,
        resnet_paths: list = None,
        camera_id: int = 0,
        device: str = 'cpu'
    ):
        """初始化"""
        self.camera_id = camera_id
        self.adapter = get_tongue_adapter(
            yolo_path=yolo_path,
            sam_path=sam_path,
            resnet_paths=resnet_paths,
            device=device,
        )
        print(f"舌诊模型将在首次分析时加载（设备: {device}）...\n")

        # 状态
        self.paused = False
        self.display_frame = None  # 用于显示的帧（可能是分析结果）
        self.current_frame = None  # 当前摄像头原始帧

    def analyze_frame(self, frame: np.ndarray) -> tuple:
        """分析单帧画面"""
        try:
            result = self.adapter.detect_from_frame(frame)
            return result.to_runtime_dict(), None
        except TongueDetectionError as e:
            return None, e.message
        except Exception as e:
            return None, str(e)

    def draw_result(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """在画面上绘制结果（使用PIL绘制中文）"""
        output = frame.copy()

        if not result.get('success'):
            return output

        # 绘制分割掩码
        mask = result['mask']
        mask_overlay = np.zeros_like(output)
        mask_overlay[mask > 0] = self.COLORS['mask']
        output = cv2.addWeighted(output, 1, mask_overlay, 0.3, 0)

        # 绘制检测框
        x1, y1, x2, y2 = result['box']
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)),
                     self.COLORS['box'], 3)

        # 转换为PIL绘制中文文字
        output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(output_pil)

        # 尝试加载中文字体
        try:
            # Windows 中文字体
            font_large = ImageFont.truetype("msyh.ttc", 28)
            font_small = ImageFont.truetype("msyh.ttc", 24)
        except:
            try:
                # 尝试其他字体
                font_large = ImageFont.truetype("Arial Unicode MS", 28)
                font_small = ImageFont.truetype("Arial Unicode MS", 24)
            except:
                # 使用默认字体
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

        # 绘制结果面板背景
        panel_width = 280
        panel_height = 180
        panel_x = 20
        panel_y = 20

        # 半透明背景
        overlay = Image.new('RGBA', output_pil.size, (0, 0, 0, 180))
        output_pil_rgba = output_pil.convert('RGBA')
        output_pil_paste = output_pil_rgba.copy()
        output_pil_paste.paste(overlay, (0, 0), overlay)
        output_pil = output_pil_paste.convert('RGB')

        # 重新获取draw对象
        draw = ImageDraw.Draw(output_pil)

        # 绘制标题
        draw.text((panel_x, panel_y), "舌诊分析结果",
                 fill=(255, 255, 0), font=font_large)

        # 绘制结果
        texts = [
            f"舌色: {result['tongue_color']}",
            f"苔色: {result['coat_color']}",
            f"厚度: {result['thickness']}",
            f"腻度: {result['greasiness']}",
            f"置信度: {result['conf']:.2f}"
        ]

        y_offset = panel_y + 50
        for text in texts:
            draw.text((panel_x, y_offset), text, fill=self.COLORS['text'], font=font_small)
            y_offset += 30

        # 转换回OpenCV格式
        output = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

        return output

    def save_result(self, frame: np.ndarray) -> str:
        """保存结果"""
        Path("results").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/tongue_{timestamp}.png"
        cv2.imwrite(save_path, frame)
        return save_path

    def draw_text_with_pil(self, frame: np.ndarray, text: str, position: tuple, color: tuple) -> np.ndarray:
        """使用PIL绘制中文文字"""
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        try:
            font = ImageFont.truetype("msyh.ttc", 24)
        except:
            try:
                font = ImageFont.truetype("Arial Unicode MS", 24)
            except:
                font = ImageFont.load_default()

        draw.text(position, text, fill=color, font=font)
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    def open_camera(self):
        backend_candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(("Media Foundation", cv2.CAP_MSMF))
        backend_candidates.append(("Default", None))

        for backend_name, backend in backend_candidates:
            if backend is None:
                cap = cv2.VideoCapture(self.camera_id)
            else:
                cap = cv2.VideoCapture(self.camera_id, backend)

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"摄像头后端: {backend_name}")
                return cap

            cap.release()

        return None

    def run(self):
        """运行实时分析"""
        # 打开摄像头
        cap = self.open_camera()

        if cap is None or not cap.isOpened():
            print("无法打开摄像头")
            return

        print("📷 实时舌诊分析")
        print("=" * 50)
        print("操作说明:")
        print("  [空格] - 暂停并分析当前画面")
        print("         再次按空格恢复实时画面")
        print("  [q]    - 退出")
        print("=" * 50)

        while True:
            # 始终从摄像头读取最新帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            self.current_frame = frame

            # 根据状态决定显示内容
            if self.paused:
                # 暂停状态，显示分析结果
                display_frame = self.display_frame
            else:
                # 实时状态，显示摄像头画面
                display_frame = frame.copy()

            # 绘制状态栏
            if self.paused:
                status_text = "[PAUSED] 按空格恢复"
                status_color = self.COLORS['paused_bg']
            else:
                status_text = "[LIVE] 按空格分析"
                status_color = (0, 255, 0)

            # 状态栏背景
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, h - 60), (w, h), (0, 0, 0), -1)

            # 使用PIL绘制中文状态
            display_frame = self.draw_text_with_pil(display_frame, status_text, (20, h - 45), (255, 255, 255))

            cv2.imshow('Real-time Tongue Diagnosis', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # 退出
                break

            elif key == ord(' '):  # 空格
                if not self.paused:
                    # 当前是实时状态 → 暂停并分析
                    self.paused = True

                    print("\n" + "=" * 50)
                    print("🔍 暂停画面，正在分析...")

                    # 使用当前最新帧进行分析
                    result, error = self.analyze_frame(self.current_frame)

                    if result and result['success']:
                        # 绘制结果
                        self.display_frame = self.draw_result(self.current_frame, result)

                        # 自动保存
                        save_path = self.save_result(self.display_frame)

                        print("✅ 分析完成!")
                        print(f"   舌色: {result['tongue_color']}")
                        print(f"   苔色: {result['coat_color']}")
                        print(f"   厚度: {result['thickness']}")
                        print(f"   腻度: {result['greasiness']}")
                        print(f"💾 已自动保存: {save_path}")
                        print("=" * 50)
                        print("按空格键恢复实时画面...")
                    else:
                        # 分析失败，显示原始帧
                        self.display_frame = self.current_frame.copy()
                        print(f"❌ 分析失败: {error}")
                        print("按空格键恢复实时画面...")

                else:
                    # 当前是暂停状态 → 恢复实时
                    self.paused = False
                    self.display_frame = None
                    print("\n▶️  恢复实时画面")

        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 程序已退出")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实时舌诊分析')
    parser.add_argument('--camera', '-c', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='设备 (cpu/cuda)')

    args = parser.parse_args()

    analyzer = RealTimeTongueAnalysis(camera_id=args.camera, device=args.device)
    analyzer.run()


if __name__ == "__main__":
    main()
