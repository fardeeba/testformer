import os
import cv2
import random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2

def get_BraTS_dicts(img_dir):
    
    imgs_anns_train = os.listdir(img_dir)

    if img_dir.endswith("training"):
        img_dir_seg = 'datasets/Brats_patches_flair_axial_view/annotations/training'
    else:
        img_dir_seg = 'datasets/Brats_patches_flair_axial_view/annotations/validation'

    dataset_dicts = []
    for idx in range(0,len(imgs_anns_train)):
        record = {}

        filename = os.path.join(img_dir, imgs_anns_train[idx])
        # Load the image
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # Get the height and width of the image
        height, width = image.shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        record['sem_seg_file_name'] = os.path.join(img_dir_seg, imgs_anns_train[idx])

        dataset_dicts.append(record)
    return dataset_dicts


# braTS_metadata = MetadataCatalog.get("BraTS20_training")
# print(braTS_metadata)

# dataset_dicts = get_BraTS_dicts("D:/Thesis/Brats_patches_flair_axial_view/flair/training")
# print(dataset_dicts)
