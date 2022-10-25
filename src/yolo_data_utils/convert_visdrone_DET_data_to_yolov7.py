import os
from pathlib import Path
import shutil
import cv2
import csv
import argparse
from types import new_class

def _convert_visidrone_DET_row_to_yolov7_row(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = int(box[0]) + int(box[2])/2.0
    y = int(box[1]) + int(box[3])/2.0
    w = int(box[2])
    h = int(box[3])
    c = int(box[5])-1
    # if x > size[1] or y > size[0]:
      # print(str((x,y,w,h)) + " ==> " + str(size))
    return (c, x*dw,y*dh,w*dw,h*dh)

  
def _adjust_visidrone_DET_row_for_image_resize(originalImageSize, newImageSize, box):
    oi_w = originalImageSize[1]
    oi_h = originalImageSize[0]
    ni_w = newImageSize[0]
    ni_h = newImageSize[1]
    # print(str(originalImageSize) + " ==> " + str(newImageSize))
    new_box = box.copy()
    new_box[0] = str(round(int(box[0])*(ni_w*1./oi_w)))
    new_box[1] = str(round(int(box[1])*(ni_h*1./oi_h)))
    new_box[2] = str(round(int(box[2])*(ni_w*1./oi_w)))
    new_box[3] = str(round(int(box[3])*(ni_h*1./oi_h)))
    # print("==========================")
    # print("new_box ==> " + str(new_box) + "; old_box ==> " + str(box))
    return new_box


def _visdrone_DET_to_yolov7_files(data_folder_dir, yolov7_output_folder_dir, new_image_size):
    """
    Converts visdrone-det annotations into coco annotation.
    Args:
        data_folder_dir: str
            'VisDrone2019-DET-train' folder directory
        yolov7_output_folder_dir: str
            Output folder path for yolov7 converted data
        image_size_dict: dict
            contains the image_size info for various videos
    """
    # init paths/folders    
    input_ann_folder = str(Path(data_folder_dir) / "annotations")
    input_images_folder = str(Path(data_folder_dir) / "images")    
    # print(input_ann_folder)
    annotation_filepath_list = os.listdir(input_ann_folder)
    output_annotations_folder = str(Path(yolov7_output_folder_dir)/"labels")
    output_images_folder = str(Path(yolov7_output_folder_dir)/"images")
    Path(output_images_folder).mkdir(parents=True, exist_ok=True)
    # shutil.copytree(input_images_folder, output_images_folder)
    Path(output_annotations_folder).mkdir(parents=True, exist_ok=True)
    annotation_dict = {}
    unique_image_sizes = set()
    
    i = 0
    for annotation_filename in annotation_filepath_list:
      # if i > 10:
        # break
      yolov7_rows = []
      input_image_path = os.path.splitext(str(Path(input_images_folder)/annotation_filename))[0]+".jpg"
      input_annotation_file =  Path(input_ann_folder)/annotation_filename
      output_image_path = os.path.splitext(str(Path(output_images_folder)/annotation_filename))[0]+".jpg"
      output_annotation_file = str(Path(output_annotations_folder)/annotation_filename)
      with open(input_annotation_file) as file_obj:
        # print(image_path)
        img = cv2.imread(input_image_path)
        original_image_size = img.shape
        unique_image_sizes.add(original_image_size)
        # print("image size for (h,w,c) " + annotation_filename + " ==> " + str(image_size))
        resized_image = cv2.resize(img, new_image_size)
        
        # print("writing image to: " + resized_image_path)
        try:
          cv2.imwrite(output_image_path, resized_image)
        except:
          print("cannot write to: " + output_image_path)
        reader_obj = csv.reader(file_obj)        
        for row in reader_obj:          
          # yolov7_annotation_file_name = os.path.splitext(annotation_filename)[0] + "_" + str(row[0]).zfill(7) + ".txt"
          if row[4] == '0' or row[5] == '11':#'VisDrone' 'ignored regions' class 0 or 'other' class 11
            continue
          resized_row  =  _adjust_visidrone_DET_row_for_image_resize(original_image_size, new_image_size, row)
          # print("resized_row ==> " + str(resized_row))
          yolov7_row = _convert_visidrone_DET_row_to_yolov7_row(new_image_size, resized_row)
          yolov7_rows.append(yolov7_row)
      with open(output_annotation_file, "w", newline="") as f:
        for row in yolov7_rows:
          f.write(' '.join(str(item) for item in row)+'\n')      
      i = i + 1
    for im_size in unique_image_sizes:
      print("image_size ==> " + str(im_size))


def create_visdrone_DET_data_in_yolov7_format(new_image_size = (960,544)):
    # the default image size is (960x544) as it has an aspect ratio close to 16:9, and also as both width and height 
    # are multiples of 32 as needed by the convolution layers of YOLOv7
  
  _visdrone_DET_to_yolov7_files('./VisDrone/VisDrone2019-DET-train', './VisDrone/VisDrone2019-DET-YOLOv7/train', new_image_size)
  _visdrone_DET_to_yolov7_files('./VisDrone/VisDrone2019-DET-val', './VisDrone/VisDrone2019-DET-YOLOv7/val', new_image_size)
  _visdrone_DET_to_yolov7_files('./VisDrone/VisDrone2019-DET-test-dev', './VisDrone/VisDrone2019-DET-YOLOv7/test-dev', new_image_size)
  ## Create data.yaml file
  try:
      with open('./VisDrone/VisDrone2019-DET-YOLOv7/data.yaml', 'w') as f:
          f.write('train: ../VisDrone/VisDrone2019-DET-YOLOv7/train/images\n')
          f.writelines('val: ../VisDrone/VisDrone2019-DET-YOLOv7/val/images\n')
          f.writelines('dev: ../VisDrone/VisDrone2019-DET-YOLOv7/test-dev/images\n')
          f.writelines('\n')
          f.writelines('nc: 10\n')
          f.writelines("names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']\n")
  except FileNotFoundError:
      print("The 'docs' directory does not exist")

def image_size_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


if __name__ == '__main__':      
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--output_image_size', type=image_size_type, metavar='path', required=True, help='the size of the image:(px width x px height)')
    args = parser.parse_args()
    image_size = tuple(args.output_image_size)
    create_visdrone_DET_data_in_yolov7_format(new_image_size = image_size)
    
