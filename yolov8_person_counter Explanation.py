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
        """
        self.model.fuse()
        it merges certain layers of the neural network 
        (like convolutional and batch normalization layers) 
        into a single, more efficient operation. 
        This reduces computation time and memory usage during inference, 
        making the model run quicker without losing accuracy.
        """

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
        #verbose=False means "don't print detailed progress messages to the console."

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
                """
                confidence:.2f is a string formatting specifier in Python, usually found within an f-string or used with the format() method.

                Here's what it means in short:

                confidence: This refers to a variable (a floating-point number) holding the detection's confidence score.
                .2f: This is the format specifier:
                .2: Means to display the number with exactly two digits after the decimal point.
                f: Stands for fixed-point notation, indicating that the number is a float.
                So, confidence:.2f tells Python to take the value of the confidence variable and format it as a floating-point number with two decimal places.
                """
                
                # Define bounding box color (Green in RGB).
                bbox_color = (0, 255, 0) 
                
                # Draw the bounding box rectangle.
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), bbox_color, 2)
                """
                cv2.rectangle: The OpenCV function for drawing a rectangle.
                output_image: The image (a NumPy array) on which the rectangle will be drawn.
                (xmin, ymin): The top-left corner coordinates of the rectangle.
                (xmax, ymax): The bottom-right corner coordinates of the rectangle.
                bbox_color: The color of the rectangle (e.g., (0, 255, 0) for green in BGR/RGB).
                2: The thickness of the rectangle's border in pixels.
                """
                
                # Draw the label text.
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                """Purpose: This line calculates the dimensions (width and height in pixels) 
                    that the label text will occupy if drawn with the specified font, scale, and thickness. 
                    It also gives you the baseline offset, which is crucial for accurate text positioning.

                    label: The text string to be measured (e.g., "Person 0.95").
                    cv2.FONT_HERSHEY_SIMPLEX: Specifies the font type.
                    0.6: The font scale, determining how large the characters will appear.
                    2: The thickness of the text strokes.
                    Why it's needed: You need these dimensions (text_width, text_height, baseline) 
                    to correctly size and position the black background rectangle before drawing the 
                    text itself. Without knowing the text size, you can't create a perfectly fitting background."""
                
                cv2.rectangle(output_image, (xmin, ymin - text_height - baseline - 5), 
                              (xmin + text_width + 5, ymin), (0, 0, 0), -1) 
                
                """Purpose: This line draws a filled black rectangle on the output_image. 
                    This rectangle acts as a solid background for the text label, ensuring contrast and 
                    readability regardless of the video content behind it.

                    output_image: The image where the rectangle will be drawn.
                    (xmin, ymin - text_height - baseline - 5): This is the top-left corner of the background rectangle.
                    xmin: Aligns the left edge of the background with the left edge of the bounding box.
                    ymin - text_height - baseline - 5: Calculates the Y-coordinate to place the rectangle just above the 
                    top of the detected object's bounding box (ymin), accounting for the text_height, baseline, and adding 
                    a small 5-pixel padding for visual separation.

                    (xmin + text_width + 5, ymin): This is the bottom-right corner of the background rectangle.
                    xmin + text_width + 5: Sets the right edge, taking the text_width and adding a small 5-pixel padding.
                    ymin: Aligns the bottom edge of the background with the top edge of the bounding box.
                    (0, 0, 0): The color of the rectangle, which is black (RGB).
                    -1: This parameter means the rectangle should be filled (solid color), rather than just an outline."""
                cv2.putText(output_image, label, (xmin + 2, ymin - baseline - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA) 
                
                """Purpose: This line finally draws the label text onto the output_image, positioned inside the black background rectangle.
                    output_image: The image where the text will be drawn.
                    label: The actual text string (e.g., "Person 0.95").
                    (xmin + 2, ymin - baseline - 2): The bottom-left corner of the first text character.
                    xmin + 2: A small 2-pixel offset from the left edge for padding.
                    ymin - baseline - 2: Positions the text correctly relative to the ymin of the bounding box, 
                    taking into account the baseline and a small 2-pixel padding, ensuring it sits nicely within 
                    the drawn black rectangle.

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2: These are the font type, scale, color (white in RGB),
                    and thickness, respectively, just like in getTextSize.

                    cv2.LINE_AA: This is an anti-aliasing flag, which makes the drawn text smoother and less pixelated, improving its visual quality."""

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
        """This block of code is responsible for displaying the total count of persons clearly on the video frame, 
            typically positioned in the top-left corner. It ensures the count is readable by drawing it against a solid background.

            count_text = f'Total Persons: {person_count}'

            Purpose: This line creates the text string that will be displayed. It uses an f-string 
            to embed the dynamic person_count (which you calculated earlier) directly into the message.
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 255, 255) # White text
            background_color = (0, 0, 0) # Black background

            Purpose: These lines define the visual style for the count display. 
            They set the font type, its size (font_scale), the thickness of the characters (font_thickness), 
            and the colors for the text (white) and its background (black).
            (count_text_width, count_text_height), count_baseline = cv2.getTextSize(count_text, font, font_scale, font_thickness)

            Purpose: Similar to the label drawing, this line calculates the dimensions 
            (width and height in pixels) that the count_text string will occupy when rendered 
            with the chosen font and size. It also gets the count_baseline for precise vertical positioning.

            Why it's needed: Knowing the exact size of the text allows the code to draw a perfectly sized black 
            background rectangle around it, preventing the text from overlapping with elements in the video or blending into a busy background.
            ********************************************************************************************************
            cv2.rectangle(output_image, (10, 10), (10 + count_text_width + 10, 10 + count_text_height + count_baseline + 10), background_color, -1)

            Purpose: This draws a filled black rectangle on the output_image. This rectangle serves as 
            a contrasting background for the count text, making it highly visible and readable.
            (10, 10): This is the top-left corner of the rectangle, placing it near the top-left of the video frame with a 10-pixel margin.
            (10 + count_text_width + 10, 10 + count_text_height + count_baseline + 10): This defines the bottom-right corner of the rectangle. 
            It uses the measured count_text_width, count_text_height, and count_baseline, adding 10-pixel padding on all sides, 
            to create a slightly larger box that comfortably fits the text.
            background_color: The color for the rectangle, which is black ((0, 0, 0)).
            -1: This argument tells OpenCV to fill the rectangle with the specified background_color, making it solid.
            *********************************************************************************************************
            
            cv2.putText(output_image, count_text, (10, 10 + count_text_height), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            Purpose: This is the final step where the count_text is actually drawn onto the output_image, on top of the black background.
            (10, 10 + count_text_height): This specifies the bottom-left corner of the first character of the count_text. 
            It's positioned within the black rectangle, aligned with the ymin of the rectangle and offset from the left edge.
            The remaining arguments (font, font_scale, text_color, font_thickness, cv2.LINE_AA) apply the defined style 
            (white text, anti-aliased) to the drawn number.
            In essence, this set of lines makes sure your real-time person count stands out clearly on the video, 
            providing immediate, crucial feedback on the object detection results."""

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