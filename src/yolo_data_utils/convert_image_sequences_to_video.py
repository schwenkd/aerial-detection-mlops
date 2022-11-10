import os
from pathlib import Path
import shutil
import cv2
import argparse
from types import new_class
import traceback
import sys
sys.path.extend(['/home/ec2-user/aerial-detection-mlops/yolov7'])
print(sys.path)
import detect_fastapi_v2


class ImageSequenceToVideoConverter:
    def __init__(self):
        print("Initialize the new instance of ImageSequenceToVideoConverter.")
        self.model = detect_fastapi_v2.Yolov7Detect(model_weights = 'ae-yolov7-best.pt', device = 'cpu')
    
    def get_yolov7_annotated_image(self, input_save_directory, output_save_directory, filename, input_img):
        input_file_save_location = f"{input_save_directory}/{filename}"
        output_file_save_location = f"{output_save_directory}/{filename}"
        cv2.imwrite(input_file_save_location, input_img)
        self.model.detect(input_file_save_location, output_file_save_location, image_size=960)
        output_image = cv2.imread(output_file_save_location)
        return output_image
   
    def convert_image_sequences_to_mp4_video(self, image_sequence_folder, output_mp4_video_folder, new_image_size,  draw_bounding_box, fps =10.0):
        """
        Converts image sequences into an mp4 video
        Args:
            image_sequence_folder: str
                'VisDrone2019-VID-test-challenge/sequences' folder directory
            output_mp4_video_folder: str
                Output folder path for mp4 videos
            new_image_size: tuple
                the image size to which all image need to be resized e.g., (960,544)
            fps: double
                the frames per second specification for the video
        """
        # init paths/folders    
        Path(output_mp4_video_folder).mkdir(parents=True, exist_ok=True)
        i = 0
        video_sequence_directories = [x for x in Path(image_sequence_folder).iterdir() if x.is_dir()]
        for video_sequence_directory in video_sequence_directories:
            # if i > 1:
                # break
            images_dict = {}
            print(video_sequence_directory)
            image_filepath_list = os.listdir(video_sequence_directory)
            for image_name in image_filepath_list:
                image_number = int(os.path.splitext(image_name)[0])
                images_dict[image_number] = str(Path(video_sequence_directory)/image_name)
            
            images_dict = dict(sorted(images_dict.items(), key=lambda x: int(x[0])))

            ## now start building the Video
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            video_outpt_filename = str(Path(output_mp4_video_folder)/(os.path.basename(os.path.normpath(video_sequence_directory)) + ".mp4"))
            out = cv2.VideoWriter(video_outpt_filename, fourcc, fps =fps, frameSize = new_image_size)
            input_file_save_directory = f"tmp/inference/input/{video_outpt_filename}"
            output_file_save_directory = input_file_save_directory.replace('/input/', '/output/')
            Path(input_file_save_directory).mkdir(parents=True, exist_ok=True)
            Path(output_file_save_directory).mkdir(parents=True, exist_ok=True)
            for img_num, img_file in images_dict.items():
                # print(str(Path(image_filepath_dir)/image_name))
                img = cv2.imread(img_file)
                resized_image = cv2.resize(img, new_image_size)
                if draw_bounding_box:       
                    try:
                        ## we can send the resized_image to YOLOv7 model here to doo real-time object detection
                        file_name = os.path.basename(img_file)
                        yolov7_img = self.get_yolov7_annotated_image(input_file_save_directory, output_file_save_directory, file_name, resized_image)
                        out.write(yolov7_img)
                    except:
                        print("cannot write {} to: {}".format(img_file, video_outpt_filename))
                        traceback.print_exc()
                else:
                    out.write(resized_image)
                #if i > 20:
                    # break
                i = i + 1
            out.release()
        # print(image_size_dict)

def image_size_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

if __name__ == '__main__':
    #image_sequence_folder, output_mp4_video_folder, new_image_size,  fps
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--image_sequence_folder', metavar='path', required=True, help='the path to image sequences')
    parser.add_argument('--output_mp4_video_folder', metavar='path', required=True, help='path to images')
    parser.add_argument('--output_image_size', type=image_size_type, metavar='path', required=True, help='the size of the image:(px width x px height)')                    
    parser.add_argument('--fps', metavar='path', type=int, required=True, help='fps of the output video')
    parser.add_argument('--draw_bounding_box', action="store_true", default=False, required=False, help='whether to draw_bounding_box')
    args = parser.parse_args()
    image_sequence_folder = args.image_sequence_folder
    output_mp4_video_folder = args.output_mp4_video_folder
    image_size = tuple(args.output_image_size)
    new_fps = args.fps *1.0
    draw_bounding_box = args.draw_bounding_box
    converter = ImageSequenceToVideoConverter()
    converter.convert_image_sequences_to_mp4_video(image_sequence_folder, output_mp4_video_folder, new_image_size =image_size, draw_bounding_box = draw_bounding_box, fps =new_fps)