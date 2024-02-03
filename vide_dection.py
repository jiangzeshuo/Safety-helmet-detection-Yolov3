# coding:utf-8
from __future__ import division
from PyQt5.Qt import (QThread, pyqtSignal)
import qimage2ndarray
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtWidgets import *

import time
import utils.gpu as gpu
from model.yolov3 import Yolov3
from video import *

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import qtawesome

global stop_mark  # 暂停标志
global new_video_path  # 视频路径
global camera_choose  # 摄像头标志
global quit_Video  # 退出检测标志
global fps_time  # 退出检测标志
global start  # 退出检测标志
global frames  # 退出检测标志


class Thread_2(QThread):  # 线程
    _signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.__video = cv2.VideoCapture()

    def run(self):
        global start  # 退出检测标志
        global frames
        global quit_Video
        global stop_mark
        if camera_choose:
            videofile = 1
            print("camera")
        else:
            videofile = new_video_path
        cap = cv2.VideoCapture(videofile)
        frames = 0
        start = time.time()
        font_size = 1
        while cap.isOpened():
            # 暂停设置
            alltime = time.time()
            while (stop_mark and not quit_Video):
                cv2.waitKey(100)
                start = time.time()
                frames = 0
            # 退出线程设置
            if quit_Video:
                quit_Video = False
                stop_mark = False
                cap.release()
                self.quit()
                return
            ret, frame = cap.read()
            # 开始检测
            if ret:
                bboxes = predict(frame, cfg.TEST["TEST_IMG_SIZE"], cfg.TEST["CONF_THRESH"], cfg.TEST["NMS_THRESH"],
                                 model)
                print("predict time:")
                print(time.time() - alltime)
                if (bboxes.shape[0] == 0):
                    frames += 1
                    fps = frames / (time.time() - start)
                    cv2.putText(frame, "FPS: {:5.2f}".format(fps), (10, 30),
                                cv2.FONT_HERSHEY_PLAIN, font_size, [0, 0, 225], 1);
                    print("can't find target")
                    image = qimage2ndarray.array2qimage(frame)
                    if fps > 15:
                        stop_time = fps * (time.time() - start) / 15 - (time.time() - start)
                        cv2.waitKey(stop_time * 1000)
                        fps = frames / (time.time() - start)
                    self._signal.emit(image)
                    continue
                frames += 1
                boxes = bboxes[..., :4]
                class_inds = bboxes[..., 5].astype(np.int32)
                scores = bboxes[..., 4]
                # 画框
                huakuang = time.time()
                img_result = visualize_boxes(image=frame, boxes=boxes, labels=class_inds, probs=scores,
                                             class_labels=cfg.DATA["CLASSES"])
                huakuang = time.time() - huakuang
                print("画框用时 ：")
                print(huakuang)
                fps = frames / (time.time() - start)
                if fps > 15:
                    stop_time = fps * (time.time() - start) / 15 - (time.time() - start)
                    cv2.waitKey(stop_time * 1000)
                    fps = frames / (time.time() - start)
                cv2.putText(frame, "FPS: {:5.2f}".format(fps), (10, 30),
                            cv2.FONT_HERSHEY_PLAIN, font_size, [0, 0, 225], 1);

                image = qimage2ndarray.array2qimage(img_result)
                self._signal.emit(image)
                print("用时 ：")
                print(time.time() - alltime)
            else:
                break
        cap.release()
        print("end")
        self.quit()


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        # 左侧菜单模块
        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.folder-open', color='white'), "打开文件")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.play-circle', color='white'), "开始")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.pause', color='white'), "暂停")
        self.left_button_3.setObjectName('left_button')
        self.left_button_4 = QtWidgets.QPushButton(qtawesome.icon('fa.power-off', color='white'), "退出")
        self.left_button_4.setObjectName('left_button')
        self.check_1 = QCheckBox('打开摄像头', self)
        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_button_1.setFixedHeight(50)
        self.left_button_2.setFixedHeight(50)
        self.left_button_3.setFixedHeight(50)
        self.left_button_4.setFixedHeight(50)
        self.check_1.setFixedHeight(50)

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 5, 0, 1, 3)
        self.left_layout.addWidget(self.check_1, 6, 0, 1, 3)
        self.left_widget.setFixedWidth(150)

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        # 右侧
        self.right_bar_widget = QtWidgets.QWidget()  # 右侧顶部搜索框部件
        self.right_bar_layout = QtWidgets.QGridLayout()  # 右侧顶部搜索框网格布局
        self.right_bar_widget.setLayout(self.right_bar_layout)
        self.search_icon = QtWidgets.QLabel(chr(0xf002))
        self.search_icon.setFont(qtawesome.font('fa', 16))
        self.right_bar_widget_search_input = QtWidgets.QLineEdit()
        self.right_bar_widget_search_input.setPlaceholderText("")
        self.right_bar_widget_search_input.setFixedHeight(40)
        self.search_icon.setFixedHeight(30)
        self.search_icon.setFixedWidth(30)

        self.right_bar_layout.addWidget(self.search_icon, 0, 0, 1, 1)
        self.right_bar_layout.addWidget(self.right_bar_widget_search_input, 0, 1, 1, 8)
        self.right_bar_widget.setFixedHeight(50)
        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 1, 10)

        # 右侧
        self.right_recommend_label = QtWidgets.QLabel()
        self.right_recommend_label.setObjectName('right_lable')
        # self.right_recommend_label.setScaledContents(True)

        self.right_layout.addWidget(self.right_recommend_label, 1, 0, 1, 9)

        # 窗口控制按钮
        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        # 左侧菜单按钮
        self.left_widget.setStyleSheet('''
            QCheckBox{
                color:white;
                background:black;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QPushButton{
                border:none;color:white;
                background:black;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QPushButton#left_label{

                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
            QWidget#left_widget{
                background:black;
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-left:1px solid white;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;
            }
        ''')

        self.right_bar_widget_search_input.setStyleSheet(
            '''QLineEdit{
                    border:1px solid gray;
                    width:300px;
                    border-radius:10px;
                    padding:2px 4px;
            }''')

        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        self.main_layout.setSpacing(0)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        self.setStyleSheet("background-color:white;")
        # self.setWindowOpacity(1)  # 设置窗口透明度
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.right_widget.setMinimumHeight(500)
        self.right_widget.setMinimumWidth(600)
        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 1)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 1, 12, 1)  # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        # signal connects
        self.left_button_1.clicked.connect(self.openFile)
        self.left_button_2.clicked.connect(self.startVideo)
        self.left_button_3.clicked.connect(self.pauseVideo)
        self.left_button_4.clicked.connect(self.quitVideo)

        self.check_1.stateChanged.connect(self.camera_choice)

        global stop_mark
        stop_mark = False
        global quit_Video
        quit_Video = False
        print('initialize finished')

    def openFile(self):
        # 选取文件
        global new_video_path
        FileName, FileType = QFileDialog.getOpenFileName(self, "选择视频进行检测", "",
                                                         "视频格式(*.avi *.mp4);;All Files(*)")
        new_video_path = FileName
        self.right_bar_widget_search_input.setPlaceholderText(new_video_path)

    def startVideo(self):
        # 开始播放  是否选定摄像头
        global camera_choose
        camera_choose = self.check_1.isChecked()
        # 启动多线程
        self.thread_2 = Thread_2()
        self.thread_2._signal.connect(self.showNextFrame)
        self.thread_2.start()

    def quitVideo(self):
        global quit_Video
        quit_Video = True
        self.right_recommend_label.clear()
        self.thread_2.quit()

    def camera_choice(self):
        # 是否打开摄像头
        global camera_choose
        camera_choose = self.check_1.isChecked()

    def pauseVideo(self):
        # 暂停播放
        global stop_mark
        if stop_mark:
            stop_mark = False
        else:
            stop_mark = True

    def showNextFrame(self, i):
        # 在画布上显示图片

        new_picture = i.scaled(self.right_recommend_label.width(), self.right_recommend_label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.right_recommend_label.setPixmap(QPixmap.fromImage(new_picture.rgbSwapped()))


if __name__ == "__main__":
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    print("载入神经网络....")
    model = Yolov3().to(gpu.select_device(args.device))
    weight = os.path.join(args.weightsfile)
    chkpt = torch.load(weight, map_location=gpu.select_device(args.device))
    model.load_state_dict(chkpt)
    del chkpt
    print("模型加载成功.")
    model.eval()
    CUDA = torch.cuda.is_available()  # GPU环境是否可用

    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())