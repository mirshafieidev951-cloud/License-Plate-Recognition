import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_characters(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []

    # Crop the image to the license plate area.
    y_start, y_end = 31, 141
    x_start, x_end = 115, 727
    cropped_image = image[y_start:y_end, x_start:x_end]

    # convert to grayscale and apply an inverted binary threshold
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find the extreme outer contours of the shapes in the image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and aspect ratio to isolate characters.
    char_boxes = []
    image_height = cropped_image.shape[0]
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h == 0: continue
        aspect_ratio = w / h
        if (h > 0.4 * image_height) and (aspect_ratio < 1.2):
            char_boxes.append((x, y, w, h))

    # Sorting characters.
    char_boxes = sorted(char_boxes, key=lambda box: box[0])


    character_images = []
    padding = 5

    for box in char_boxes:
        x, y, w, h = box
        
        #Crop the character tightly from the original image.
        char_gray_tight = gray[y:y+h, x:x+w]
        char_thresh_tight = thresh[y:y+h, x:x+w]

        #Add a 5 pixel white border around the tightly cropped character.
        char_gray_padded = cv2.copyMakeBorder(char_gray_tight, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
        char_thresh_padded = cv2.copyMakeBorder(char_thresh_tight, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        character_images.append({'gray': char_gray_padded, 'thresh': char_thresh_padded})
        
    return character_images


def save_segmented_characters(character_images, output_dir='segmented_characters'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    for i, images in enumerate(character_images):
        save_path = os.path.join(full_output_dir, f"{i}.png")
        cv2.imwrite(save_path, images['gray'])

    print("Segmented characters saved successfully!")
    print(f"Saving location: {full_output_dir}")

    fig, axes = plt.subplots(1, len(character_images), figsize=(len(character_images) * 1.5, 1.5))
    if len(character_images) == 1:
        axes = [axes]

    for i, images in enumerate(character_images):
        save_path = os.path.join(full_output_dir, f"{i}.png")
        cv2.imwrite(save_path, images['gray'])

        ax = axes[i]
        ax.imshow(images['gray'], cmap='gray')
        ax.axis('off')

    plt.suptitle("Segmented Characters")
    plt.show()

def recognize_character(char_image, templates):
    best_match_char = None
    max_corr_value = -1.0

    h_char, w_char = char_image.shape

    for char_name, template_img in templates.items():
        # Resize the template to match the input character's dimensions for accurate comparison.
        resized_template = cv2.resize(template_img, (w_char, h_char))
        
        result = cv2.matchTemplate(char_image, resized_template, cv2.TM_CCOEFF_NORMED)
        corr_value = result[0][0]

        if corr_value > max_corr_value:
            max_corr_value = corr_value
            best_match_char = char_name
            
    return best_match_char, max_corr_value

def recognize_plate(character_images, template_dir='numbers'):
    if not character_images:
        print("No characters were found to recognize.")
        return

    # Load templates and convert them to binary format to match the segmented characters.
    templates = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_template_dir = os.path.join(script_dir, template_dir)
    
    for filename in os.listdir(full_template_dir):
        if filename.endswith('.jpg'):
            char_name = os.path.splitext(filename)[0]
            template_path = os.path.join(full_template_dir, filename)
            template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            _, template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
            templates[char_name] = template_binary
            
    # Recognize each character and assemble the final result.
    final_plate_text = ""
    recognition_report = []

    for images in character_images:
        recognized_char, corr_val = recognize_character(images['thresh'], templates)
        if recognized_char:
            final_plate_text += recognized_char
            recognition_report.append((recognized_char, corr_val))

    # Print the final results.
    print(f"\nRecognized Plate: {final_plate_text}")
    print("--- Correlation Report ---")
    for char, val in recognition_report:
        print(f"Character: {char} -> Correlation: {val:.2f}")

    
script_dir = os.path.dirname(os.path.abspath(__file__))
image_file_path = os.path.join(script_dir, 'results/p4.jpg')

print("Step 1: Segmenting characters from the license plate...")
segmented_chars = segment_characters(image_file_path)
save_segmented_characters(segmented_chars)

if segmented_chars:
    print("\nStep 2: Recognizing segmented characters...")
    recognize_plate(segmented_chars)
else:
    print("Could not proceed with recognition as no characters were segmented.")