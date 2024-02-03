from __future__ import division
from PyQt5.Qt import (QThread, QMutex, pyqtSignal)
import qimage2ndarray
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import argparse
import torch
import os
import cv2
import time
from model.yolov3 import Yolov3
from utils.tools import *
import utils.gpu as gpu
import config.yolov3_config_voc as cfg
from utils.data_augment import Resize
from utils.visualize import *
from video import *
global stop_mark
global new_video_path
global camera_choose
qmut_2 = QMutex()


class Thread_2(QThread):  # 线程2
    _signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.__video = cv2.VideoCapture()

    def run(self):
        # qmut_2.lock()  # 加锁
        if camera_choose:
            videofile = 0
            print("camera")
        else:
            print("视频路径为：", new_video_path)
            videofile = new_video_path
        cap = cv2.VideoCapture(videofile)
        assert cap.isOpened(), 'Cannot capture source'
        frames = 0
        start = time.time()
        while cap.isOpened():
            while (stop_mark):
                # print("wait time: ",stop_time*5,"s")
                # stop_time += 1
                cv2.waitKey(1000)
            ret, frame = cap.read()
            print('1')
            if ret:
                star=time.time()
                bboxes =predict(frame,cfg.TEST["TEST_IMG_SIZE"],cfg.TEST["CONF_THRESH"],cfg.TEST["NMS_THRESH"],model)
                frames += 1
                boxes = bboxes[..., :4]
                class_inds = bboxes[..., 5].astype(np.int32)
                scores = bboxes[..., 4]
                if(scores[0]<0.1):
                    frames+=1
                    cv2.putText(frame, "FPS: {:5.2f}".format(frames / (time.time() - start)), (10, 30),
                                cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 225], 1);
                    print("can't find target")
                    print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                    image = qimage2ndarray.array2qimage(frame)
                    self._signal.emit(image)
                    continue

                visualize_boxes(image=frame, boxes=boxes, labels=class_inds, probs=scores, class_labels=cfg.DATA["CLASSES"])
                cv2.putText(frame, "FPS: {:5.2f}".format(frames / (time.time() - start)), (10, 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 225], 1);
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                print(time.time()-star)
                image = qimage2ndarray.array2qimage(frame)
                self._signal.emit(image)
            else:
                print("end")
                break

class VideoWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        # 非widget资源
        self.__video_capture = cv2.VideoCapture()
        self.__show_frame_timer = QTimer()

        # widgets
        self.__image_label = QLabel()
        self.__video_path_edit = QLineEdit()

        self.check_1 = QCheckBox('摄像头', self)
        self.check_1.stateChanged.connect(self.camera_choice)

        self.__reset_button = QPushButton('打开')
        self.__start_button = QPushButton('开始')
        self.__pause_button = QPushButton('暂停/继续')
        self.__quit_button = QPushButton('退出')
        self.__buttons_widget = QWidget()

        # layouts
        self.__buttons_layout = QHBoxLayout()
        self.__buttons_layout.addWidget(self.__reset_button)
        self.__buttons_layout.addWidget(self.__start_button)
        self.__buttons_layout.addWidget(self.__pause_button)
        self.__buttons_layout.addWidget(self.__quit_button)
        self.__buttons_layout.addWidget(self.check_1)
        self.__buttons_widget.setLayout(self.__buttons_layout)

        self.__main_layout = QVBoxLayout()
        self.__main_layout.addWidget(self.__image_label)
        self.__main_layout.addWidget(self.__video_path_edit)
        self.__main_layout.addWidget(self.__buttons_widget)
        self.setLayout(self.__main_layout)

        # signal connects
        self.__reset_button.clicked.connect(self.resetVideo)
        self.__start_button.clicked.connect(self.startVideo)
        self.__pause_button.clicked.connect(self.pauseVideo)
        self.__quit_button.clicked.connect(self.quitVideo)

        self.__show_frame_timer.timeout.connect(self.showNextFrame)

        global camera_choose
        camera_choose = self.check_1.isChecked()
        print('initialize finished')

    def resetVideo(self):
        # 可能的资源泄漏：
        # 每次使用一个新的VideoCapture，GC能否关闭原VideoCapture ？
        global new_video_path
        FileName, FileType = QFileDialog.getOpenFileName(self, "打开/重播", "",
                                                         "视频格式(*.avi *.mp4);;All Files(*)")
        new_video_path = FileName
        self.__video_path_edit.setPlaceholderText(new_video_path)
        new_video_cap = cv2.VideoCapture()
        # new_video_path = self.__video_path_edit.text()
        open_success_flag = new_video_cap.open(new_video_path)
        new_frame_rate = new_video_cap.get(cv2.CAP_PROP_FPS)  # FPS

        if (not open_success_flag or new_frame_rate == 0):
            print("read vedio wrong")
            return
        else:
            # self.pauseVideo()
            self.__video_capture = new_video_cap
            self.__show_frame_timer.setInterval(1000 / new_frame_rate)  # ms
        # self.showNextFrame()

    def startVideo(self):
        global stop_mark
        stop_mark = False
        self.thread_2 = Thread_2()
        self.thread_2._signal.connect(self.showNextFrame)
        self.thread_2.start()

    # self.__show_frame_timer.start()

    def quitVideo(self):
        print("just a joke")

    # self.thread_2.quit()

    def camera_choice(self):
        global camera_choose
        camera_choose = self.check_1.isChecked()

    # self.__show_frame_timer.start()

    def pauseVideo(self):
        global stop_mark
        if stop_mark:
            stop_mark = False
        # print("暂停播放")
        else:
            stop_mark = True
        # print("开始播放")

    # self.__show_frame_timer.stop()

    def showNextFrame(self, i):
        # QImage(rgb) cv2.VideoCapture(bgr) have different color order,
        # use QImage.rgbSwapped() method to fix it.
        # D:\Learning\python_pycharm\yolov3-master\video\video1.avi
        self.__image_label.setPixmap(QPixmap.fromImage(i.rgbSwapped()))
    # D:\Learning\python_pycharm\yolov3-master\video\video1.avi

if __name__== "__main__":
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    print("载入神经网络....")
    model =Yolov3().to(gpu.select_device(args.device))
    weight = os.path.join(args.weightsfile)
    chkpt = torch.load(weight, map_location=gpu.select_device(args.device))
    model.load_state_dict(chkpt)
    del chkpt
    print("模型加载成功.")
    model.eval()
    CUDA = torch.cuda.is_available()  # GPU环境是否可用
    app = QApplication([])

    window = VideoWindow()
    window.show()

    sys.exit(app.exec_())