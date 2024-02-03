from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.yolov3 import Yolov3
from tqdm import tqdm
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import os
import config.yolov3_config_voc as cfg
from utils.visualize import *
import time

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
'''
name -测试类

'''

class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 visiual=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__visiual = visiual
        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)   #调用eval


    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt


    def test(self):
        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            for v in imgs:
                star = time.time()
                path = os.path.join(self.__visiual, v)
                print("test images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img)
                print(bboxes_prd[1])
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    #print(bboxes_prd.shape())
                    #print(bboxes_prd[..., 4])
                    scores = bboxes_prd[..., 4]
                    # print(v)
                    # v = v.split(".")
                    # for i, score in enumerate(scores):
                    #     with open("data/testtxt/{}.txt".format(v[0]), "a") as f:
                    #         if(class_inds[i]==0):
                    #             f.write("hat ")
                    #             score = "%.3f" % score
                    #             f.write(score + " ")
                    #             boxes[i, 0] = "%.0f" % boxes[i, 0]
                    #             boxes[i, 1] = "%.0f" % boxes[i, 1]
                    #             boxes[i, 2] = "%.0f" % boxes[i, 2]
                    #             boxes[i, 3] = "%.0f" % boxes[i, 3]
                    #             print(boxes[i])
                    #             f.write(str(boxes[i, 0]) + " ")
                    #             f.write(str(boxes[i, 1]) + " ")
                    #             f.write(str(boxes[i, 2]) + " " + str(boxes[i, 3]) + " \n")
                    #             f.close()
                    #         elif(class_inds[i]==1):
                    #             if(score>0.3):
                    #                 f.write("person ")
                    #                 score = "%.3f" % score
                    #                 f.write(score + " ")
                    #                 boxes[i, 0] = "%.0f" % boxes[i, 0]
                    #                 boxes[i, 1] = "%.0f" % boxes[i, 1]
                    #                 boxes[i, 2] = "%.0f" % boxes[i, 2]
                    #                 boxes[i, 3] = "%.0f" % boxes[i, 3]
                    #                 print(boxes[i])
                    #                 f.write(str(boxes[i, 0]) + " ")
                    #                 f.write(str(boxes[i, 1]) + " ")
                    #                 f.write(str(boxes[i, 2]) + " " + str(boxes[i, 3]) + " \n")
                    #                 f.close()
                    #
                    #             elif(len(scores)>200 and score>0.2):
                    #                 f.write(len(scores))
                    #                 f.write("person ")
                    #                 score = "%.3f" % score
                    #                 f.write(score + " ")
                    #                 boxes[i, 0] = "%.2f" % boxes[i, 0]
                    #                 boxes[i, 1] = "%.2f" % boxes[i, 1]
                    #                 boxes[i, 2] = "%.2f" % boxes[i, 2]
                    #                 boxes[i, 3] = "%.2f" % boxes[i, 3]
                    #                 print(boxes[i])
                    #                 f.write(str(boxes[i, 0]) + " ")
                    #                 f.write(str(boxes[i, 1]) + " ")
                    #                 f.write(str(boxes[i, 2]) + " " + str(boxes[i, 3]) + " \n")
                    #                 f.close()
                    #     print("{}has writ".format(v[0]))

                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    path = os.path.join(cfg.PROJECT_PATH, "data/{}".format(v))
                    print(time.time()-star)
                    cv2.imwrite(path, img)
                    print("saved images : {}".format(path))


        if self.__eval:
            mAP = 0
            print('*' * 20 + "Validate" + '*' * 20)

            with torch.no_grad():
                APs = Evaluator(self.__model).APs_voc(self.__multi_scale_test, self.__flip_test)

                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                print('mAP:%g' % (mAP))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best5.2270.pt', help='weight file path')
    parser.add_argument('--visiual', type=str, default="data/test", help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=False, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( weight_path=opt.weight_path,
            gpu_id=opt.gpu_id,
            eval=opt.eval,
            visiual=opt.visiual).test()


