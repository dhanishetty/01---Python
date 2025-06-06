import torch
import numpy as np
import cv2
from ultralytics import YOLO # Import the YOLO class for YOLOv8

from openfilter.filter_runtime import Frame, Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

class YOLOv8PersonCounterFilter(Filter):
    def setup(self, config):
        """
        Initializes the YOLOv8 model and sets up class IDs for detection.
        This method is called once when the filter pipeline starts.
        """
        print(f'YOLOv8PersonCounterFilter setup: {config.my_option=}')

        # Load a pre-trained YOLOv8 model.
        self.model = YOLO('yolov8n.pt') 
        self.model.fuse() # Fuse model for faster inference

        # Get the class names from the model. This is a dictionary where keys are IDs and values are names.
        self.class_names_dict = self.model.names # Renamed for clarity
        print("YOLOv8 Class Names:", self.class_names_dict) 

        # Find the class ID for 'person' by iterating through the dictionary.
        self.person_class_id = -1 # Initialize with a default invalid value
        for class_id, class_name in self.class_names_dict.items():
            if class_name == 'person':
                self.person_class_id = class_id
                break
        
        if self.person_class_id == -1:
            raise ValueError("The loaded YOLOv8 model does not contain a 'person' class. "
                             "Please ensure you're using a model trained on COCO or a custom dataset with 'person'.")
        
        # Set a confidence threshold for detections.
        self.confidence_threshold = 0.5 

    def process(self, frames):
        """
        Processes each incoming video frame, performs object detection,
        counts persons, and draws results on the frame.
        """
        # Get the current frame as a NumPy array (RGB format).
        frame_data = frames['main'].rw_rgb
        image = frame_data.image  # NumPy array (H, W, C)
        data = frame_data.data    # Metadata dictionary

        # Perform YOLOv8 Inference on the image.
        results = self.model(image, verbose=False)

        # The results object from YOLOv8 needs to be handled slightly differently than YOLOv5.
        # 'results[0]' refers to the detections for the first (and in this case, only) image in the batch.
        # '.boxes.data' contains the raw detection data: [xmin, ymin, xmax, ymax, confidence, class_id]
        # Move to CPU and convert to NumPy for easier manipulation and drawing with OpenCV.
        detections = results[0].boxes.data.cpu().numpy()

        # Create a copy of the image to draw on to avoid modifying the original input.
        output_image = image.copy()

        # Initialize person count for the current frame.
        person_count = 0

        # Iterate through detected objects, filter, and draw.
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = detection

            # Convert to integer and float types for consistency.
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            confidence = float(confidence)
            class_id = int(class_id)

            # Apply confidence threshold and check if it's a person.
            if confidence >= self.confidence_threshold and class_id == self.person_class_id:
                person_count += 1

                # Prepare label for drawing.
                label = f'{self.class_names_dict[class_id]} {confidence:.2f}' # Use the dictionary for name lookup
                
                # Define bounding box color (Green in RGB).
                bbox_color = (0, 255, 0) 
                
                # Draw the bounding box rectangle.
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), bbox_color, 2)
                
                # Draw the label text.
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (xmin, ymin - text_height - baseline - 5), 
                              (xmin + text_width + 5, ymin), (0, 0, 0), -1) 
                cv2.putText(output_image, label, (xmin + 2, ymin - baseline - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA) 

        # --- Display the total person count on the image ---
        count_text = f'Total Persons: {person_count}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255) # White text
        background_color = (0, 0, 0) # Black background

        (count_text_width, count_text_height), count_baseline = cv2.getTextSize(count_text, font, font_scale, font_thickness)
        
        cv2.rectangle(output_image, (10, 10), 
                      (10 + count_text_width + 10, 10 + count_text_height + count_baseline + 10), 
                      background_color, -1) 

        cv2.putText(output_image, count_text, (10, 10 + count_text_height), 
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return Frame(output_image, data, 'RGB')

    def shutdown(self):
        """
        Cleanup method, called when the filter pipeline is shut down.
        """
        print('YOLOv8PersonCounterFilter shutting down')

if __name__ == '__main__':
    Filter.run_multi([
        (VideoIn, dict(sources='file://video.mp4!sync', outputs='tcp://*:5555')),
        (YOLOv8PersonCounterFilter, dict(sources='tcp://localhost:5555', outputs='tcp://*:5552', my_option='PersonCounting')),
        (Webvis, dict(sources='tcp://localhost:5552')),
    ])