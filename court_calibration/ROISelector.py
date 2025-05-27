import cv2
import numpy as np

class ROISelector:
    def __init__(self, image):
        self.image = image.copy()
        self.original_image = image.copy()
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.roi_selected = False

        # 图像尺寸
        self.height, self.width = image.shape[:2]

        # 窗口名称
        self.window_name = "ROI选择器 - 拖拽鼠标画框，按ESC退出，按SPACE确认"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新结束点
                self.end_point = (x, y)
                # 重绘图像
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            # 结束绘制
            self.drawing = False
            self.end_point = (x, y)
            self.roi_selected = True
            self.update_display()

    def update_display(self):
        # 复制原始图像
        display_image = self.original_image.copy()

        # 如果有选择区域，绘制矩形
        if self.start_point and self.end_point:
            cv2.rectangle(display_image, self.start_point, self.end_point, (0, 255, 0), 2)

            # 显示坐标信息
            if self.roi_selected:
                # 计算相对坐标
                rel_coords = self.get_relative_coordinates()
                coord_text = f"ROI: [{rel_coords[0]:.2f}, {rel_coords[1]:.2f}, {rel_coords[2]:.2f}, {rel_coords[3]:.2f}]"
                cv2.putText(display_image, coord_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, "Press SPACE to confirm, ESC to exit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, display_image)

    def get_relative_coordinates(self):
        if not (self.start_point and self.end_point):
            return None

        # 确保坐标顺序正确
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # 转换为相对坐标 (0-1范围)
        rel_x_min = x_min / self.width
        rel_y_min = y_min / self.height
        rel_x_max = x_max / self.width
        rel_y_max = y_max / self.height

        return [rel_x_min, rel_y_min, rel_x_max, rel_y_max]

    def get_pixel_coordinates(self):
        if not (self.start_point and self.end_point):
            return None

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        return [x_min, y_min, x_max, y_max]

    def select_roi(self):
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 显示初始图像
        self.update_display()

        print("使用说明:")
        print("1. 用鼠标拖拽画框选择ROI区域")
        print("2. 按SPACE键确认选择")
        print("3. 按ESC键退出")
        print("-" * 50)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC键退出
                print("用户取消操作")
                cv2.destroyAllWindows()
                return None

            elif key == 32:  # SPACE键确认
                if self.roi_selected:
                    cv2.destroyAllWindows()
                    return self.get_relative_coordinates()
                else:
                    print("请先选择ROI区域")

def extract_first_frame(video_path):
    """提取视频第一帧"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("无法读取视频第一帧")

    return frame

def main():
    # 硬编码视频路径 - 请修改为你的视频路径
    video_path = "videos/OneMatch.mp4"  # 修改为你的视频文件路径

    try:
        print(f"正在提取视频第一帧: {video_path}")

        # 提取第一帧
        first_frame = extract_first_frame(video_path)
        print(f"成功提取第一帧，图像尺寸: {first_frame.shape[1]}x{first_frame.shape[0]}")

        # 创建ROI选择器
        roi_selector = ROISelector(first_frame)

        # 开始ROI选择
        roi_coords = roi_selector.select_roi()

        if roi_coords:
            print("=" * 50)
            print("ROI选择完成!")
            print(f"图像尺寸: {first_frame.shape[1]}x{first_frame.shape[0]} (宽x高)")

            # 获取像素坐标
            pixel_coords = roi_selector.get_pixel_coordinates()
            print(f"像素坐标: [x_min={pixel_coords[0]}, y_min={pixel_coords[1]}, x_max={pixel_coords[2]}, y_max={pixel_coords[3]}]")

            # 输出相对坐标（保留4位小数）
            print(f"相对坐标 (0-1范围): [{roi_coords[0]:.4f}, {roi_coords[1]:.4f}, {roi_coords[2]:.4f}, {roi_coords[3]:.4f}]")

            # 输出最终配置格式
            print("\n最终配置:")
            print(f'"roi": [{roi_coords[0]:.2f}, {roi_coords[1]:.2f}, {roi_coords[2]:.2f}, {roi_coords[3]:.2f}],')

        else:
            print("未选择ROI区域")

    except Exception as e:
        print(f"错误: {e}")
        print("请检查视频路径是否正确，以及视频文件是否存在")

if __name__ == "__main__":
    main()