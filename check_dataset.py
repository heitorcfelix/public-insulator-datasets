import argparse
import cv2
from tqdm import tqdm
from glob import glob
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description='Data Label check')
    parser.add_argument('data_root', help='data file path')
    parser.add_argument('json_file_path', help='json file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    coco_file = COCO(args.json_file_path)

    for index in tqdm(range(len(coco_file.imgToAnns.keys()))):
        img_id = list(coco_file.imgToAnns.keys())[index]
        img_name = coco_file.imgs[img_id]['file_name']
        img = cv2.imread(args.data_root + img_name)
        for label in coco_file.imgToAnns[img_id]:
            if label['category_id'] == 0:
                color = (255, 0 , 0)
            elif label['category_id'] == 1:
                color = (0, 0, 255)
            elif label['category_id'] == 2:
                color = (0, 255, 0)
            img = cv2.rectangle(img, (label['bbox'][0], label['bbox'][1]), (label['bbox'][0]+label['bbox'][2], label['bbox'][1]+label['bbox'][3]), color, 2)
        cv2.imshow('debug', cv2.resize(img, (500, 500)))
        pressed = cv2.waitKey(0)
        while(not pressed):
            pressed = cv2.waitKey(0)
        if pressed == 27:
            break

if __name__ == "__main__":
    main()
