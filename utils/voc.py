import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import config.yolov3_config_voc as cfg
import os
from tqdm import tqdm



def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    """
    解析 pascal voc数据集的annotation, 表示的形式为[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: 数据集的路径 , 如 D:\doc\data\VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: 文件的类型， 'trainval''train''val'
    :param anno_path: 标签存储路径
    :param use_difficult_bbox: 是否适用difficult==1的bbox
    :return: 数据集大小
    """
    classes = cfg.DATA["CLASSES"]
    img_inds_file = os.path.join(data_path,'ImageSets', file_type+'.txt')
    print(img_inds_file)
    with open(img_inds_file, 'r') as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]
        hat=0
        person=0
    with open(anno_path, 'a') as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, 'images', image_id + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_id + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1): # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find('bndbox')
                class_id = classes.index(obj.find("name").text.lower().strip())
                if(class_id == 0):
                    hat = hat+1
                elif(class_id==1):person=person+1
                #print(str(class_id)+"+1")
        print("hat="+str(hat))
        print("person=" + str(person))
            #     xmin = bbox.find('xmin').text.strip()
            #     ymin = bbox.find('ymin').text.strip()
            #     xmax = bbox.find('xmax').text.strip()
            #     ymax = bbox.find('ymax').text.strip()
            #     annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
            # annotation += '\n'
            # #print(annotation)
            # if(annotation.find(',')!=-1):
            #     f.write(annotation)
        print("The number of images   {0}  ".format(len(image_ids)))
        print("The number of hat   {0}  ".format(hat))
        print("The number of person   {0}  ".format(person))
    return len(image_ids)


if __name__ =="__main__":
    # train_set :
    train_annotation_path = os.path.join('../data', 'train_annotation.txt')
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    # val_set   :
    val_annotation_path = os.path.join('../data', 'val_annotation.txt')
    if os.path.exists(val_annotation_path):
        os.remove(val_annotation_path)

    test_annotation_path = os.path.join('../data', 'test_annotation.txt')
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    len_train = parse_voc_annotation('F:/deeplearn/yolov3-master/data', "train", train_annotation_path, use_difficult_bbox=False)
    #len_val = parse_voc_annotation('F:/deeplearn/yolov3-master/data', "val", val_annotation_path, use_difficult_bbox=False)
    #len_test = parse_voc_annotation('F:/deeplearn/yolov3-master/data', "test", test_annotation_path, use_difficult_bbox=False)


