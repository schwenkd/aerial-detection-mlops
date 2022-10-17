import os
from pathlib import Path
import shutil
import cv2
import csv
class VisDroneVideoToYoloConverter:
    def _visdrone_video_to_yolov7_image_files(self, data_folder_dir, yolov7_output_folder_dir):
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
            print(annotation_filename)
            image_filepath_dir = os.path.splitext(str(Path(data_folder_dir) / "sequences"/annotation_filename))[0]
            image_filepath_list = os.listdir(image_filepath_dir)
            i = 0
            for image_name in image_filepath_list:
                yolov7_image_name = os.path.splitext(annotation_filename)[0] + "_" + image_name
                path = shutil.copyfile(image_filepath_dir + "/" + image_name,str(Path(output_image_folder)/yolov7_image_name))
                if i == 0:
                    img = cv2.imread(path)
                    image_size_dict[annotation_filename] = img.shape
                    i += 1
        return image_size_dict

    def _convert_visidrone_row_to_yolov7_row(self, size, box):
        dw = 1./size[1]
        dh = 1./size[0]
        x = int(box[2]) + int(box[4])/2.0
        y = int(box[3]) + int(box[5])/2.0
        w = int(box[4])
        h = int(box[5])
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (box[7], x,y,w,h)

    def _visdrone_video_to_yolov7_annotation_files(self, data_folder_dir, yolov7_output_folder_dir, image_size_dict):
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
            print("image size for (h,w,c) " + annotation_filename + " ==> " + str(image_size))
            with open(Path(input_ann_folder)/annotation_filename) as file_obj:
                reader_obj = csv.reader(file_obj)
                i = 0
                for row in reader_obj:          
                    yolov7_annotation_file_name = os.path.splitext(annotation_filename)[0] + "_" + str(row[0]).zfill(7) + ".txt" 
                    yolov7_row = self._convert_visidrone_row_to_yolov7_row(image_size, row)
                    if i == 0:
                        print(annotation_filename + "==>" + str(row))          
                        print(yolov7_annotation_file_name + "==>" + str(yolov7_row))
                    if yolov7_annotation_file_name not in annotation_dict:
                        annotation_dict[yolov7_annotation_file_name] = []
                    annotation_dict[yolov7_annotation_file_name].append(yolov7_row)
                    i += 1
        for k, v in annotation_dict.items():
            with open(str(Path(output_annotations_folder)/k), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(v)

    def convert_visdrone_data_to_yolov7_format(self, data_folder_dir, yolov7_output_folder_dir):
        """
            Converts visdrone-VID annotations into yolov7 annotation.
            Args:
                data_folder_dir: str
                    'VisDrone2019-VID-train' folder directory
                yolov7_output_folder_dir: str
                    Output folder path for yolov7 converted data
        """
        image_size_dictionary = self._visdrone_video_to_yolov7_image_files(data_folder_dir, yolov7_output_folder_dir)
        print(image_size_dictionary)
        self._visdrone_video_to_yolov7_annotation_files(data_folder_dir, yolov7_output_folder_dir, image_size_dictionary)



    

