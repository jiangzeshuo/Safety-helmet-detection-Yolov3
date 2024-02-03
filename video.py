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
def arg_parse():
    """
    视频检测模块的参数转换
    """
    # 创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser(description='YOLO v3 检测模型')
    parser.add_argument("--weights", dest='weightsfile', help="模型权重",default="weight/best5.2270.pt", type=str)
    parser.add_argument("--video", dest="videofile", help="待检测视频目录", default=r"C:\Users\97265\Documents\Tencent Files\972653466\FileRecv\z实拍2.mp4", type=str)
    parser.add_argument("--device", dest="device", help="gpu", default="0", type=int)
    #parser.add_argument("shape" ,dest='shape',help = '')
    return parser.parse_args()

def predict(img, test_shape,conf_thresh,nms_thresh,model):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape
        img = Resize((test_shape,test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        img = torch.from_numpy(img[np.newaxis, ...]).float().to(next(model.parameters()).device)
        with torch.no_grad():
            _, p_d = model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        star = time.time()
        bboxes = convert_pred(pred_bbox, test_shape,(org_h, org_w), (0, np.inf),conf_thresh)
        #print(time.time()-star)
        new = time.time()
        bboxes = nms(bboxes,conf_thresh, nms_thresh)
        #bboxes = nms_all(bboxes,conf_thresh, nms_thresh)
        print(time.time()-star)
        return bboxes

def convert_pred(pred_bbox, test_input_size, org_img_shape, valid_scale,conf_thresh):
    """
    预测框进行过滤，去除尺度不合理的框
    """
    pred_coor = xywh2xyxy(pred_bbox[:, :4])
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2)将预测的bbox中超出原图的部分裁掉
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # (3)将无效bbox的coor置为0
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5)将score低于score_threshold的bbox去掉
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > conf_thresh
    mask = np.logical_and(scale_mask, score_mask)
    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    return bboxes

if __name__== "__main__":
    t= time.time()
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
    videofile = args.videofile
    cap = cv2.VideoCapture(videofile)  # 用 OpenCV 打开视频
    # cap = cv2.VideoCapture(0)  #for webcam(相机)
    assert cap.isOpened(), 'Cannot capture source'
    # frames用于统计图片的帧数
    frames = 0
    start = time.time()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 24
    savedPath = './savevideo.avi'  # 保存的地址和视频名
    ret, frame = cap.read()
    videoWriter = cv2.VideoWriter(savedPath, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后为视频图片的形状
    print("从读视频到处理时间为")
    print(time.time()-t)
    print("\n")
    while cap.isOpened():
        b=time.time()
        ret, frame = cap.read()
        #cv2.imshow('frame',frame)
        #cv2.waitKey(1)
        if ret:
            a = time.time()
            bboxes =predict(frame,cfg.TEST["TEST_IMG_SIZE"],cfg.TEST["CONF_THRESH"],cfg.TEST["NMS_THRESH"],model)
            print("检测用时")
            print(time.time()-a)
            print("\n")
            if(bboxes.shape[0]==0):
                frames+=1
                print("can't find target")
                print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                videoWriter.write(frame)
                continue
            frames += 1
            boxes = bboxes[..., :4]
            class_inds = bboxes[..., 5].astype(np.int32)
            scores = bboxes[..., 4]
            v=time.time()
            visualize_boxes(image=frame, boxes=boxes, labels=class_inds, probs=scores, class_labels=cfg.DATA["CLASSES"])
            print("可视化用时")
            print(time.time()-v)
            path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(frames))
            videoWriter.write(frame)
            print("合计用时")
            print(time.time() - b)
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            videoWriter.release()  # 结束循环的时候释放
            break
