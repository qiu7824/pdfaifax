from collections import deque
from PySide6 import QtWidgets, QtCore
from PIL import Image
from PySide6.QtGui import QAction
import fitz
from fitz import tools
import os,time,shutil
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torchvision.transforms.v2 import ToTensor
class Worker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(str)

    def __init__(self, filename, options, page_size_option, page_orientation_option):
        super().__init__()
        self.filename = filename
        self.options = options
        self.page_size_option = page_size_option
        self.page_orientation_option = page_orientation_option
        
    def run(self):
        try:
            doc = fitz.open(self.filename)
            total = len(doc)
            self.output_dir = os.path.join(os.path.dirname(self.filename), f"{os.path.basename(self.filename)}_{time.strftime('%Y%m%d%H%M%S')}")
            os.makedirs(self.output_dir, exist_ok=True)
            pdf = fitz.open()  # 创建一个新的PDF
            for i in range(total):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72,300/72))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                for option, is_checked in self.options.items():  # 这里遍历的是字典的键和值
                    if is_checked:  # 如果选项被选中
                        if option == "漂白效果":
                            image = self.bleach_image(image)
                        if option == "去槽点":
                            image = self.remove_noise(image)
                        if option == "文本锐化":
                            image = self.sharpen_image(image)
                image_path = os.path.join(self.output_dir, f"{i}.png")
                if "压缩文件" in self.options and self.options["压缩文件"]:
                    # 在保存图像前进行压缩
                    image.save(image_path, 'JPEG', quality=70)
                else:
                    image.save(image_path)
                
                if "AI优化" in self.options and self.options["AI优化"]:
                    image = self.ai_optimize_image(image_path)
            
                            
                # 注意这里：根据用户选择的页面大小设置新页面的大小
                if self.page_size_option == "A4":
                    w_pt, h_pt = 595, 842  # A4 页面大小（单位：点）
                elif self.page_size_option == "A3":
                    w_pt, h_pt = 842, 1191  # A3 页面大小（单位：点）
                else:  # 默认为 A4 大小
                    w_pt, h_pt = 595, 842
                pdf_page = pdf.new_page(width=w_pt, height=h_pt)  # 创建新页面
                # 计算图像在页面上的位置：居中对齐
                x0 = (w_pt - image.width * 72 / 300) / 2
                y0 = (h_pt - image.height * 72 / 300) / 2
                x1 = x0 + image.width * 72 / 300
                y1 = y0 + image.height * 72 / 300
                rect = fitz.Rect(x0, y0, x1, y1)
                pdf_page.insert_image(rect, filename=image_path)  # 在新页面上插入图像
                self.progress.emit((i + 1) / total * 100)
            optimized_pdf_filename = os.path.join(os.path.dirname(self.filename), os.path.basename(self.filename).split(".")[0] + "_已优化.pdf")
            pdf.save(optimized_pdf_filename)
            shutil.rmtree(self.output_dir)
        finally:
            self.finished.emit("")
    @staticmethod
    def bleach_image(image):
        return image.convert("L").point(lambda x: min(x + 50, 255))

    def process_page_size(self, image):
        if self.page_size_option == "A4":
            # 如果页面大小选项是A4，那么调整图像的大小来适应A4尺寸
            a4_dims = (595, 842)
            image = image.resize(a4_dims)
        elif self.page_size_option == "A3":
            # 如果页面大小选项是A3，那么调整图像的大小来适应A3尺寸
            a3_dims = (842, 1191)
            image = image.resize(a3_dims)
        return image

    def process_page_orientation(self, image):
        if self.page_orientation_option == "横向":
            # 如果页面方向选项是横向，那么调整图像的方向为横向
            image = image.transpose(Image.ROTATE_90)
        return image

    def remove_noise(self, image):
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
 
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        elif len(image_array.shape) == 2: 
            gray_image = image_array
        else:
            raise Exception(f"Unexpected image shape: {image_array.shape}")
        denoised_image = cv2.fastNlMeansDenoising(gray_image, h=3, templateWindowSize=7, searchWindowSize=21)

        return Image.fromarray(denoised_image)
    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # 锐化核
        image_cv = np.array(image)  # 将 PIL 图像转换为 OpenCV 图像
        image_cv = cv2.filter2D(image_cv, -1, kernel)  # 应用锐化核
        return Image.fromarray(image_cv)  # 将 OpenCV 图像转换回 PIL 图像
    def ai_optimize_image(self, image_path):
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')

        # 加载模型
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'models/RealESRGAN-x4plus.pth'  # 确保模型文件存在于此路径
        
        # 使用 RealESRGANer 类实例化模型
        upsampler = RealESRGANer(
            scale=4, # 根据你的需求设置
            model_path=model_path, 
            model=model,
            tile=500, # 根据你的需求设置
            tile_pad=10, # 根据你的需求设置
            pre_pad=0, # 根据你的需求设置
            half=True, # 如果你想使用半精度浮点数（fp16），将这个设置为 True
            gpu_id=0 # 使用第一个 GPU
        )
        # 将 PIL 图像转换为 OpenCV 图像
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        to_tensor = ToTensor()
        # 运行模型
        print('AI optimization started.')
        try:
            out_img, _ = upsampler.enhance(image_cv, outscale=1) # 根据你的需求设置
        except Exception as e:
            print(f'Error when applying the model: {e}')
            return None

        # 将输出 OpenCV 图像转换回 PIL 图像
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(out_img)
        out_img.save(image_path)
        return out_img
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 主布局
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # 添加菜单栏
        self.menuBar = self.menuBar()
        
        # 添加帮助菜单项
        self.helpMenu = self.menuBar.addMenu('关于')
        
        # 添加关于 PySide6 的菜单项
        self.aboutPySideAction = QAction('关于 Real-ESRGAN', self)
        self.aboutPySideAction.triggered.connect(self.show_about_RealESRGAN)
        self.helpMenu.addAction(self.aboutPySideAction)
        self.aboutPySideAction = QAction('关于 PySide6', self)
        self.aboutPySideAction.triggered.connect(self.show_about_pyside)
        self.helpMenu.addAction(self.aboutPySideAction)
        self.worker = None
        self.filenames = deque()  # 使用队列存储多个文件名
        # 添加一个按钮用于选择PDF文件
        self.select_button = QtWidgets.QPushButton("选择PDF")
        self.select_button.clicked.connect(self.select_pdf)
        main_layout.addWidget(self.select_button)

        # 添加复选框用于选择优化选项
        self.options = {
            "漂白效果": QtWidgets.QCheckBox("漂白效果"),
            "去槽点": QtWidgets.QCheckBox("去槽点"),
            "文字纠正": QtWidgets.QCheckBox("文字锐化"),
            "AI优化": QtWidgets.QCheckBox("AI优化"),
            "压缩文件": QtWidgets.QCheckBox("压缩文件"),
        }
        for option in self.options.values():
            main_layout.addWidget(option)

        # 添加一个下拉菜单用于选择页面大小和方向
        self.page_size_label = QtWidgets.QLabel("页面大小:")
        main_layout.addWidget(self.page_size_label)
        self.page_size = QtWidgets.QComboBox()
        self.page_size.addItems(["自动识别","A4", "A3"])
        main_layout.addWidget(self.page_size)

        self.page_orientation_label = QtWidgets.QLabel("页面方向:")
        main_layout.addWidget(self.page_orientation_label)
        self.page_orientation = QtWidgets.QComboBox()
        self.page_orientation.addItems(["自动识别","横向", "竖向"])
        main_layout.addWidget(self.page_orientation)

        # 添加一个 "开始" 按钮，并初始设置为禁用状态
        self.start_button = QtWidgets.QPushButton("开始")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)  # 初始设置为禁用
        main_layout.addWidget(self.start_button)
        self.start_button.setStyleSheet("""
        QPushButton { background-color: #008CBA; border: none; padding: 5px; }
        QPushButton:hover { background-color: #007B9E; }
        QPushButton:pressed { background-color: #00586B; }
        QPushButton:disabled { background-color: #CCCCCC; }
        """)
        # 添加一个进度条
        self.progress_bar = QtWidgets.QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # 设置主窗口的布局和标题
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setWindowTitle("PDF智能优化")

        # 设置窗口默认大小
        self.resize(600, 300)
        self.setStyleSheet("""
QMainWindow { background-color: #ffffff; }
QPushButton { background-color: #dddddd; color: black; border: 1px solid #aaaaaa; padding: 5px; }
QPushButton:hover { background-color: #cccccc; }
QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background-color: #1D65A6; width: 20px; }
QCheckBox { color: black; }
QLabel { color: black; }
QComboBox { background-color: #dddddd; color: black; border: 1px solid #aaaaaa; padding: 2px; }
QComboBox:hover { background-color: #026ec8; }
QComboBox QAbstractItemView { background-color: #ffffff; color: black; selection-background-color: #dddddd; }
""")
        
    def show_about_pyside(self):
        QtWidgets.QMessageBox.information(self, 
            "关于 PySide6", 
            "此应用程序使用了 PySide6 库。你可以在此链接查看其源代码：https://wiki.qt.io/Qt_for_Python"
        )
    def show_about_RealESRGAN(self):
        QtWidgets.QMessageBox.information(self, 
            "关于 Real-ESRGAN", 
            "此应用程序使用了 Real-ESRGAN 库。你可以在此链接查看其源代码：https://github.com/xinntao/Real-ESRGAN/"
        )
    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def processing_finished(self):
        QtWidgets.QMessageBox.information(self, "完成", "PDF处理完成！")
        self.progress_bar.setValue(0)
        if self.filenames:  # 如果还有待处理的文件，继续处理下一个文件
            self.process_next_file()
        else:  # 否则，重新启用 "开始" 按钮，并清除已选择的文件
            self.start_button.setEnabled(True)
    def select_pdf(self):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open File", ".", "PDF Files (*.pdf)")
        if filenames:
            self.filenames.extend(filenames)  # 将所有选中的文件添加到队列中
            self.start_button.setEnabled(True)  # 如果成功选择了文件，启用 "开始" 按钮
    def start_processing(self):
        if self.filenames:
            self.process_next_file()
    def process_next_file(self):
        if self.filenames:
            filename = self.filenames.popleft()  # 从队列中取出下一个待处理的文件
            selected_options = {key: checkbox.isChecked() for key, checkbox in self.options.items()}
            selected_page_size = self.page_size.currentText()
            selected_page_orientation = self.page_orientation.currentText()

            self.worker = Worker(filename, selected_options, selected_page_size, selected_page_orientation)
            self.worker.progress.connect(self.update_progress_bar)
            self.worker.finished.connect(self.processing_finished)
            self.worker.start()

            self.start_button.setEnabled(False)  # 开始处理后，禁用 "开始" 按钮




if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
