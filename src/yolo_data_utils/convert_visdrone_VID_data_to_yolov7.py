import os
from pathlib import Path
import shutil
import cv2
import csv
import argparse
def _visdrone_video_to_yolov7_image_files(data_folder_dir, yolov7_output_folder_dir, new_image_size):
    """
    Converts visdrone-det annotations into coco annotation.
    Args:
        data_folder_dir: str
            'VisDrone2019-VID-train' folder directory
        yolov7_output_folder_dir: str
            Output folder path for yolov7 converted data
    """
    # init paths/folders    
    input_ann_folder = str(Path(data_folder_dir) / "annotations")
    print(input_ann_folder)
    annotation_filepath_list = os.listdir(input_ann_folder)
    output_image_folder = str(Path(yolov7_output_folder_dir)/"images")
    Path(output_image_folder).mkdir(parents=True, exist_ok=True)
    image_size_dict = {}
    for annotation_filename in annotation_filepath_list:
      # print(annotation_filename)
      image_filepath_dir = os.path.splitext(str(Path(data_folder_dir) / "sequences"/annotation_filename))[0]
      image_filepath_list = os.listdir(image_filepath_dir)
      i = 0
      for image_name in image_filepath_list:
        yolov7_image_name = os.path.splitext(annotation_filename)[0] + "_" + image_name
        img_path = str(Path(image_filepath_dir)/image_name)
        img = cv2.imread(img_path)
        image_size_dict[annotation_filename] = img.shape
        resized_image = cv2.resize(img, new_image_size)
        resized_image_path = str(Path(output_image_folder)/yolov7_image_name)        
        try:
          cv2.imwrite(resized_image_path, resized_image)
        except:
          print("cannot write to: " + resized_image_path)
      
    return image_size_dict

def _convert_visidrone_video_row_to_yolov7_row(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = int(box[2]) + int(box[4])/2.0
    y = int(box[3]) + int(box[5])/2.0
    w = int(box[4])
    h = int(box[5])
    c = int(box[7])-1    
    return (c, x*dw,y*dh,w*dw,h*dh)

  
def _adjust_visidrone_video_row_for_image_resize(originalImageSize, newImageSize, box):
    oi_w = originalImageSize[1]
    oi_h = originalImageSize[0]
    ni_w = newImageSize[0]
    ni_h = newImageSize[1]
    # print(str(originalImageSize) + " ==> " + str(newImageSize))
    new_box = box.copy()
    new_box[2] = str(round(int(box[2])*(ni_w*1./oi_w)))
    new_box[3] = str(round(int(box[3])*(ni_h*1./oi_h)))
    new_box[4] = str(round(int(box[4])*(ni_w*1./oi_w)))
    new_box[5] = str(round(int(box[5])*(ni_h*1./oi_h)))
    # print("==========================")
    # print("new_box ==> " + str(new_box) + "; old_box ==> " + str(box))
    return new_box

def _visdrone_video_to_yolov7_annotation_files(data_folder_dir, yolov7_output_folder_dir, new_image_size, image_size_dict):
    """
    Converts visdrone-det annotations into coco annotation.
    Args:
        data_folder_dir: str
            'VisDrone2019-VID-train' folder directory
        yolov7_output_folder_dir: str
            Output folder path for yolov7 converted data
        image_size_dict: dict
            contains the image_size info for various videos
    """
    # init paths/folders    
    input_ann_folder = str(Path(data_folder_dir) / "annotations")
    print(input_ann_folder)
    annotation_filepath_list = os.listdir(input_ann_folder)
    output_annotations_folder = str(Path(yolov7_output_folder_dir)/"labels")
    Path(output_annotations_folder).mkdir(parents=True, exist_ok=True)
    annotation_dict = {}
    for annotation_filename in annotation_filepath_list:
      image_size = image_size_dict[annotation_filename]
      # print("image size for (h,w,c) " + annotation_filename + " ==> " + str(image_size))
      with open(Path(input_ann_folder)/annotation_filename) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
          if row[6] == '0' or row[7] == '11':#'VisDrone' 'ignored regions' class 0 or 'other' class 11
            continue          
          yolov7_annotation_file_name = os.path.splitext(annotation_filename)[0] + "_" + str(row[0]).zfill(7) + ".txt"
          resized_row  =  _adjust_visidrone_video_row_for_image_resize(image_size, new_image_size, row) 
          yolov7_row = _convert_visidrone_video_row_to_yolov7_row(new_image_size, resized_row)
          if yolov7_annotation_file_name not in annotation_dict:
            annotation_dict[yolov7_annotation_file_name] = []
          annotation_dict[yolov7_annotation_file_name].append(yolov7_row)

    for k, v in annotation_dict.items():
      with open(str(Path(output_annotations_folder)/k), "w", newline="") as f:
        for row in v:
          f.write(' '.join(str(item) for item in row)+'\n')

def _convert_visdrone_data_to_yolov7_format(data_folder_dir, yolov7_output_folder_dir, _new_image_size):
  """
    Converts visdrone-VID annotations into yolov7 annotation.
    Args:
        data_folder_dir: str
            'VisDrone2019-VID-train' folder directory
        yolov7_output_folder_dir: str
            Output folder path for yolov7 converted data
  """
  image_size_dictionary = _visdrone_video_to_yolov7_image_files(data_folder_dir, yolov7_output_folder_dir, _new_image_size)
  print(image_size_dictionary)
  _visdrone_video_to_yolov7_annotation_files(data_folder_dir, yolov7_output_folder_dir, _new_image_size, image_size_dictionary)

def create_visdrone_video_data_in_yolov7_format(new_image_size = (960, 544)):
    # the default image size is (960x544) as it has an aspect ratio close to 16:9, and also as both width and height 
    # are multiples of 32 as needed by the convolution layers of YOLOv7:
  
    _convert_visdrone_data_to_yolov7_format('./VisDrone/VisDrone2019-VID-train', './VisDrone/VisDrone2019-VID-YOLOv7/train', new_image_size)
    _convert_visdrone_data_to_yolov7_format('./VisDrone/VisDrone2019-VID-val', './VisDrone/VisDrone2019-VID-YOLOv7/val', new_image_size)
    _convert_visdrone_data_to_yolov7_format('./VisDrone/VisDrone2019-VID-test-dev', './VisDrone/VisDrone2019-VID-YOLOv7/test-dev', new_image_size)
    #Create data.yaml file
    try:
        with open('./VisDrone/VisDrone2019-VID-YOLOv7/data.yaml', 'w') as f:
            f.write('train: ../VisDrone/VisDrone2019-VID-YOLOv7/train/images\n')
            f.writelines('val: ../VisDrone/VisDrone2019-VID-YOLOv7/val/images\n')
            f.writelines('dev: ../VisDrone/VisDrone2019-VID-YOLOv7/test-dev/images\n')
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
    create_visdrone_video_data_in_yolov7_format(new_image_size = image_size)
