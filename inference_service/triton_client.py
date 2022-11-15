import numpy as np
import sys
import cv2
import argparse
import logging
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import VisDroneLabels
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class TritonClient:
    INPUT_NAMES = ["images"]
    OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
    # setup loggers
    
    # https://docs.python.org/3/library/argparse.html#argumentparser-objects
    def __init__(self, model = 'yolov7-visdrone-finetuned', triton_url='triton:8001'):
        """
        We instantiate the TritonClient class with the triton_url
        Args:
            - triton_url (str): path to the triton server
        """
        self.model = model #Inference model name, default yolov7
        self.model_info = False
        self.client_timeout = None

        
        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url= triton_url, # Inference server URL, default localhost:8001
                verbose= False, #Enable verbose client output
                ssl= False, #Enable SSL encrypted channel to the server
                root_certificates= None, #File holding PEM-encoded root certificates, default none
                private_key= None, #File holding PEM-encoded private key, default is none
                certificate_chain= None) #File holding PEM-encoded certicate chain default is none
        except Exception as e:
            logger.error("context creation failed: " + str(e))
            sys.exit()

        # Health check
        if not self.triton_client.is_server_live():
            logger.error("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            logger.error("FAILED : is_server_ready")
            sys.exit(1)

        if not self.triton_client.is_model_ready(self.model):
            logger.error("FAILED : is_model_ready")
            sys.exit(1)

        if self.model_info:
            # Model metadata
            try:
                self.metadata = self.triton_client.get_model_metadata(model)
                logger.info(self.metadata)
            except InferenceServerException as ex:
                if "Request for unknown model" not in ex.message():
                    logger.error("FAILED : get_model_metadata")
                    logger.error("Got: {}".format(ex.message()))
                    sys.exit(1)
                else:
                    logger.error("FAILED : get_model_metadata")
                    sys.exit(1)

            # Model configuration
            try:
                self.config = self.triton_client.get_model_config(model)
                if not (self.config.config.name == model):
                    logger.error("FAILED: get_model_config")
                    sys.exit(1)
                logger.info(self.config)
            except InferenceServerException as ex:
                logger.error("FAILED : get_model_config")
                logger.error("Got: {}".format(ex.message()))
                sys.exit(1)

    def detect_image(self, input_image_file: str, output_image_file: str, output_label_file: str, image_width = 960, image_height = 960):
        #     logger.info("Running in 'image' mode")
        if not input_image_file:
            logger.warn("FAILED: no input image")
            return
        if not output_image_file:
            logger.warn("FAILED: no output_image_file specified")
            return

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(self.INPUT_NAMES[0], [1, 3, image_width, image_height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[3]))

        logger.debug("Creating buffer from image file...")
        input_image = cv2.imread(str(input_image_file))
        if input_image is None:
            logger.warn(f"FAILED: could not load input image {str(input)}")
            return
        input_image_buffer = preprocess(input_image, [image_width, image_height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs[0].set_data_from_numpy(input_image_buffer)

        logger.debug("Invoking inference...")
        start_time = time.time()
        results = self.triton_client.infer(model_name=self.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=self.client_timeout)
        logger.info(f"Time taken to infer by TritonServer: {int((time.time()-start_time)*1000)} milli seconds")
        if self.model_info:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model)
            if len(statistics.model_stats) != 1:
                logger.warn("FAILED: get_inference_statistics")
                # sys.exit(1)
            logger.info(statistics)

        for output in self.OUTPUT_NAMES:
            result = results.as_numpy(output)
            if logger.isEnabledFor(level=logging.DEBUG):
                logger.debug(f"Received result buffer \"{output}\" of size {result.shape}. Naive buffer sum: {np.sum(result)}")

        num_dets = results.as_numpy(self.OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(self.OUTPUT_NAMES[1])
        det_scores = results.as_numpy(self.OUTPUT_NAMES[2])
        det_classes = results.as_numpy(self.OUTPUT_NAMES[3])
        detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [image_width, image_height])
        logger.debug(f"Detected objects: {len(detected_objects)}")

        output_labels = []
        for box in detected_objects:
            logger.debug(f"{VisDroneLabels(box.classID).name}: {box.confidence}")
            input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            size = get_text_size(input_image, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            input_image = render_text(input_image, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)
            box_ctr = box.center_normalized()
            box_size = box.size_normalized()
            output_labels.append(f'{box.classID},{box_ctr[0]},{box_ctr[1]},{box_size[0]},{box_size[1]}')

        if output_image_file:
            cv2.imwrite(output_image_file, input_image)
            logger.debug(f"Saved result to {output_image_file}")
        else:
            cv2.imshow('image', input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if output_label_file:
            try:
                with open(output_label_file, 'w') as f:
                    for row in output_labels:
                        f.writelines(f'{row}\n')
                logger.info(f"Written {output_label_file}")
            except Exception as e:
                logger.warn(f"{output_label_file} could not be written. Error: {str(e)}")
        self.print_statistics()

    def detect_video(self, input_video_file: str, output_video_file: str, image_width = 960, image_height = 960, fps=10.0):
        logger.info("Running in 'video' mode")
        if not input_video_file:
            logger.warn("FAILED: no input video")
            sys.exit(1)
        if not output_video_file:
            logger.warn("FAILED: no output_video_file specified")
            return

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(self.INPUT_NAMES[0], [1, 3, image_width, image_height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(self.OUTPUT_NAMES[3]))

        print("Opening input video stream...")
        cap = cv2.VideoCapture(input_video_file)
        if not cap.isOpened():
            logger.warn(f"FAILED: cannot open video {input_video_file}")
            sys.exit(1)

        counter = 0
        out = None
        print("Invoking inference...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("failed to fetch next frame")
                break

            if counter == 0 and output_video_file:
                logger.info("Opening output video stream...")
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
                out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame.shape[1], frame.shape[0]))

            input_image_buffer = preprocess(frame, [image_width, image_height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

            inputs[0].set_data_from_numpy(input_image_buffer)

            results = self.triton_client.infer(model_name=self.model,
                                          inputs=inputs,
                                          outputs=outputs,
                                          client_timeout=self.client_timeout)

            num_dets = results.as_numpy("num_dets")
            det_boxes = results.as_numpy("det_boxes")
            det_scores = results.as_numpy("det_scores")
            det_classes = results.as_numpy("det_classes")
            detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, frame.shape[1], frame.shape[0], [image_width, image_height])
            logger.debug(f"Frame {counter}: {len(detected_objects)} objects")
            counter += 1

            for box in detected_objects:
                logger.debug(f"{VisDroneLabels(box.classID).name}: {box.confidence}")
                frame = render_box(frame, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
                size = get_text_size(frame, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
                frame = render_filled_box(frame, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
                frame = render_text(frame, f"{VisDroneLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

            out.write(frame)
        cap.release()
        out.release()
        self.print_statistics()

    def print_statistics(self):
        if self.model_info:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model)
            if len(statistics.model_stats) != 1:
                logger.warn("FAILED: get_inference_statistics")
                sys.exit(1)
            logger.info(statistics)
