import os
from pathlib import Path
import shutil
import cv2
from yolov7.detect_fastapi import detect

class ImageSequenceToVideoConverter:
    def __init__(self):
        print("Initialize the new instance of ImageSequenceToVideoConverter.")
    
    def get_yolov7_annotated_image(self, filename, input_img):
        input_file_save_location = f"tmp/inference/input/{filename}"
        output_file_save_location = input_file_save_location.replace('/input/', '/output/')
        cv2.imwrite(input_file_save_location, input_img)
        detect(input_file_save_location, output_file_save_location, model_weights = '../ae-yolov7-best.pt', image_size=960)
        output_image = cv2.imread(output_file_save_location)
        return output_image
   
    def convert_image_sequences_to_mp4_video(self, image_sequence_folder, output_mp4_video_folder, new_image_size,  fps =10.0):
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
        
        video_sequence_directories = [x for x in Path(image_sequence_folder).iterdir() if x.is_dir()]
        for video_sequence_directory in video_sequence_directories:
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
            
            for img_num, img_file in images_dict.items():
                # print(str(Path(image_filepath_dir)/image_name))
                img = cv2.imread(img_file)
                resized_image = cv2.resize(img, new_image_size)       
                try:
                    ## we can send the resized_image to YOLOv7 model here to doo real-time object detection
                    file_name = os.path.basename(img_file)
                    yolov7_img = self.get_yolov7_annotated_image(file_name, resized_image)
                    out.write(yolov7_img)
                except:
                    print("cannot write {} to: {}".format(img_file, video_outpt_filename))
            out.release()
        # print(image_size_dict)
if __name__ == '__main__':
    converter = ImageSequenceToVideoConverter()
    converter.convert_image_sequences_to_mp4_video('/content/VisDroneVideo/VisDrone2019-VID-test-challenge/sequences', "/content/VisDroneVideo/VisDrone2019-VID-test-challenge/mp4_videos", new_image_size =(960,544), fps =10.0)