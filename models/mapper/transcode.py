import cv2
import numpy as np
import os

def process_images_in_folder(folder_path):
    # Define the intermediate and final resolutions
    intermediate_width, intermediate_height = 960, 540
    target_width, target_height = 400, 224
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            image_path = os.path.join(folder_path, filename)
            
            # Read the image
            frame_bgr = cv2.imread(image_path)
            
            if frame_bgr is None:
                print(f"Error: Could not read image {image_path}")
                continue
            
            # Get image dimensions
            height, width, _ = frame_bgr.shape
            
            # Check if resizing is needed
            if width != target_width or height != target_height:
                # First resize to 960x540 without averaging (using nearest neighbor for pixelation)
                intermediate_image = cv2.resize(frame_bgr, (intermediate_width, intermediate_height), cv2.INTER_NEAREST)
                
                # Then resize to 400x224 using INTER_AREA
                small_image = cv2.resize(intermediate_image, (target_width, target_height), cv2.INTER_AREA).astype(np.float32)
                
                # Save the resized image in place
                cv2.imwrite(image_path, small_image)
                print(f"Image {image_path} resized and saved.")
            else:
                print(f"Image {image_path} already at desired resolution.")

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    process_images_in_folder(current_folder)