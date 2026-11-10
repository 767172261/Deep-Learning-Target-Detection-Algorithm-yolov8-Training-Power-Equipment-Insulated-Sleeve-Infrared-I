深度学习目标检测算法yolov8训练电力设备绝缘套管红外图像与可见光图像配准数据集
@[toc]

电力设备绝缘套管红外图像与可见光图像配准数据集（绝缘套管），200多张，一一对应
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/668e4819002742be8f00c232d9d75cb7.png)

基于高压绝缘子红外检测数据集的详细实现流程，安装依赖到构建 GUI 应用程序的完整步骤和代码。假设数据集为 YOLO 格式（包含 `.txt` 标签文件和 `yaml` 配置文件）。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0aa31d10848a455b8c20c96038e2dda2.png)

### **1. 安装依赖**

确保安装了必要的库：

```bash
pip install ultralytics opencv-python pyqt5 pandas torch torchvision
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/677776b75c644f5eb11a3a103a3cf09e.png)

### **2. 数据准备**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9cf2b0d22cad468abd0dfeef8f986b98.png)
#### **2.1 数据集结构**
假设数据集目录结构如下：
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

#### **2.2 标签格式**
每个标签文件是 `.txt` 格式，每一行表示一个目标，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

#### **2.3 数据集配置文件**
创建一个 `data.yaml` 文件，用于定义数据集路径和类别名称：

```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 2  # 类别数量
names: ['defect', 'normal']  # 类别名称
```

---

### **3. 配置 YOLOv8**

使用 `ultralytics` 提供的 YOLOv8 模型进行训练和推理。

---

### **4. 训练与评估**

#### **4.1 训练代码**

编写训练代码，使用 YOLOv8 进行训练：

```python
from ultralytics import YOLO

def train_model():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用 YOLOv8 Nano 预训练模型

    # 开始训练
    model.train(
        data='data.yaml',       # 数据集配置文件
        epochs=50,              # 训练轮数
        imgsz=640,              # 输入图片尺寸
        batch=16,               # 批次大小
        device='cuda',          # 使用 GPU
        workers=8,              # 数据加载线程数
        project='runs/train',   # 训练结果保存路径
        name='exp'              # 实验名称
    )

if __name__ == "__main__":
    train_model()
```

#### **4.2 评估代码**

编写评估代码，在验证集上评估模型性能：

```python
from ultralytics import YOLO

def evaluate_model():
    model = YOLO("runs/train/exp/weights/best.pt")  # 加载训练好的模型
    metrics = model.val()  # 在验证集上评估模型
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
```

---

### **5. 构建 GUI 应用程序**

使用 PyQt5 构建一个简约的系统 UI 界面，支持图片、视频和摄像头检测，并导出检测结果。

#### **完整 GUI 代码**

```python
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
import cv2
import pandas as pd
import os

class InsulatorDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高压绝缘子红外检测系统")
        self.setGeometry(100, 100, 800, 600)

        # 初始化模型
        self.model = YOLO("runs/train/exp/weights/best.pt")  # 加载训练好的模型
        self.class_names = ['defect', 'normal']  # 类别名称
        self.current_class = None  # 当前选择的目标类别

        # UI 元素
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 700, 400)

        self.btn_image = QPushButton("选择图片", self)
        self.btn_video = QPushButton("选择视频", self)
        self.btn_camera = QPushButton("打开摄像头", self)
        self.btn_export = QPushButton("导出结果", self)
        self.combo_classes = QComboBox(self)

        self.btn_image.setGeometry(50, 500, 150, 40)
        self.btn_video.setGeometry(220, 500, 150, 40)
        self.btn_camera.setGeometry(390, 500, 150, 40)
        self.btn_export.setGeometry(560, 500, 150, 40)
        self.combo_classes.setGeometry(320, 550, 150, 40)

        # 绑定按钮事件
        self.btn_image.clicked.connect(self.detect_single_image)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_export.clicked.connect(self.export_results)
        self.combo_classes.addItems(["All"] + self.class_names)
        self.combo_classes.currentTextChanged.connect(self.switch_class)

        # 视频捕获相关
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 存储检测结果
        self.results_data = []

    def detect_single_image(self):
        """检测单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if file_path:
            self.process_image(file_path)

    def detect_video(self):
        """检测视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def open_camera(self):
        """打开摄像头进行实时检测"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        """更新视频帧或摄像头捕获的画面"""
        ret, frame = self.cap.read()
        if ret:
            self.process_image(frame=frame, is_video=True)

    def process_image(self, file_path=None, frame=None, is_video=False):
        """处理图片并显示结果"""
        if not is_video:
            frame = cv2.imread(file_path)

        # 使用 YOLOv8 进行检测
        results = self.model(frame)

        # 统计目标数量
        counts = {cls: 0 for cls in self.class_names}
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                class_name = self.model.names[int(cls)]
                counts[class_name] += 1

                # 如果当前类别匹配，则绘制边界框
                if self.current_class == "All" or class_name == self.current_class:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 保存检测结果
            if not is_video:
                self.results_data.append({
                    "file": file_path,
                    **counts,
                    "detection_time": pd.Timestamp.now()
                })

        # 显示结果
        if is_video:
            self.display_frame(frame)
        else:
            self.display_image(frame)

    def display_image(self, frame):
        """显示图片"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def display_frame(self, frame):
        """显示视频帧"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def switch_class(self, class_name):
        """切换目标类别"""
        self.current_class = class_name if class_name != "All" else None

    def export_results(self):
        """导出检测结果为 Excel 或 CSV 文件"""
        if not self.results_data:
            QMessageBox.warning(self, "警告", "没有检测结果可导出！")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file_path:
            df = pd.DataFrame(self.results_data)
            if file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            elif file_path.endswith(".csv"):
                df.to_csv(file_path, index=False)
            QMessageBox.information(self, "成功", "结果已导出！")

if __name__ == "__main__":
    app = QApplication([])
    window = InsulatorDetector()
    window.show()
    app.exec_()
```

---

### **6. 功能总结**

1. **数据预处理**：YOLO 格式的数据集无需额外转换。
2. **训练与评估**：使用 YOLOv8 训练模型并评估性能。
3. **GUI 应用程序**：
   - 支持图片、视频和摄像头检测。
   - 导出检测结果为 Excel 或 CSV 文件。
4. **推理**：实时检测高压绝缘子缺陷（如 `defect` 和 `normal` 类别）。

---

### **7. 运行说明**

1. 将数据集放入正确路径，并确保 `data.yaml` 文件配置正确。
2. 使用训练代码训练模型。
3. 使用评估代码测试模型性能。
4. 启动 GUI 应用程序，加载训练好的权重文件进行检测。

希望这个详细的实现能满足你的需求！如果有进一步的问题，请随时提问。
