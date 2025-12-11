import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# SECTION 1: HELPER FUNCTIONS (from ideal.py)

def segment_characters(image, scale_factor):
    if image is None: return []

    original_padding = 5
    scaled_padding = int(max(1, original_padding * scale_factor))
    
    orig_y_start, orig_y_end, orig_x_start, orig_x_end = 31, 141, 115, 727
    y_start, y_end = int(orig_y_start * scale_factor), int(orig_y_end * scale_factor)
    x_start, x_end = int(orig_x_start * scale_factor), int(orig_x_end * scale_factor)
    
    h_img, w_img = image.shape[:2]
    y_end, x_end = min(y_end, h_img), min(x_end, w_img)
    
    cropped_image = image[y_start:y_end, x_start:x_end]
    if cropped_image.size == 0: return []

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    image_height = cropped_image.shape[0]
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if h != 0 and (h > 0.4 * image_height) and ((w/h) < 1.2):
            char_boxes.append((x, y, w, h))
            
    char_boxes = sorted(char_boxes, key=lambda box: box[0])
    character_images = []
    for box in char_boxes:
        x, y, w, h = box
        char_thresh_tight = thresh[y:y+h, x:x+w]
        char_thresh_padded = cv2.copyMakeBorder(char_thresh_tight, scaled_padding, scaled_padding, scaled_padding, scaled_padding, cv2.BORDER_CONSTANT, value=0)
        character_images.append({'thresh': char_thresh_padded})
        
    return character_images

def recognize_plate(character_images, template_dir='numbers'):

    if not character_images: return "", 0.0
    
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
            
    final_plate_text, correlation_scores = "", []
    
    for image_data in character_images:
        char_image = image_data['thresh']
        best_match_char, max_corr_value = None, -1.0
        h_char, w_char = char_image.shape
        
        for char_name, template_img_orig in templates.items():
            resized_template = cv2.resize(template_img_orig, (w_char, h_char))
            result = cv2.matchTemplate(char_image, resized_template, cv2.TM_CCOEFF_NORMED)
            corr_value = result[0][0]
            if corr_value > max_corr_value:
                max_corr_value, best_match_char = corr_value, char_name
                
        if best_match_char:
            final_plate_text += best_match_char
            correlation_scores.append(max_corr_value)
            
    return final_plate_text, (np.mean(correlation_scores) if correlation_scores else 0.0)


# SECTION 2: VISUALIZATION FUNCTIONS

def visualize_downsampling_methods(original_image, failing_rate):
    """Compares the two down-sampling methods (Subsampling vs. Resizing)."""

    print(f"\nVisualizing the difference between Subsampling and Resizing at rate 1-in-{failing_rate}...")
    subsampled_image = original_image[::failing_rate, ::failing_rate]
    h, w = subsampled_image.shape[:2]
    resized_image = cv2.resize(original_image, (w, h), interpolation=cv2.INTER_AREA)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(subsampled_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Subsampled Method (Pixel Skipping)\nDimensions: {w}x{h}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Resize Method (Open CV library)\nDimensions: {w}x{h}")
    plt.axis('off')
    plt.suptitle(f"Down-sampling Method Comparison at Failure Rate (1 in {failing_rate})")
    plt.show()

def visualize_failing_character(original_image, failing_rate, ground_truth_plate, failing_char_index, recognized_as):
    """
    Visualizes the specific character that failed and compares its correlation
    with the true template vs. the falsely recognized template.
    """
    print(f"Visualizing the binary character comparison for the failing character at index {failing_char_index}...")
    
    subsampled_image = original_image[::failing_rate, ::failing_rate]
    scale_factor = 1.0 / failing_rate
    segmented_chars = segment_characters(subsampled_image, scale_factor=scale_factor)
    
    if not segmented_chars or failing_char_index >= len(segmented_chars):
        print("-> Could not visualize: Segmentation result is shorter than expected.")
        return

    char_from_plate = segmented_chars[failing_char_index]['thresh']
    true_char_name = ground_truth_plate[failing_char_index]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Helper function to get correlation for a specific template
    def get_correlation(template_char_name):
        template_path = os.path.join(script_dir, 'numbers', f"{template_char_name}.jpg")
        template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_gray is None: return 0.0
        _, high_res_template = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
        resized_template = cv2.resize(high_res_template, (char_from_plate.shape[1], char_from_plate.shape[0]))
        return cv2.matchTemplate(char_from_plate, resized_template, cv2.TM_CCOEFF_NORMED)[0][0]

    corr_with_true = get_correlation(true_char_name)
    corr_with_false = get_correlation(recognized_as)
    
    # Visualization: Compare the segmented char against the TRUE template
    true_template_path = os.path.join(script_dir, 'numbers', f"{true_char_name}.jpg")
    true_template_gray = cv2.imread(true_template_path, cv2.IMREAD_GRAYSCALE)
    _, true_high_res_template = cv2.threshold(true_template_gray, 127, 255, cv2.THRESH_BINARY_INV)
    display_template = cv2.resize(true_high_res_template, (char_from_plate.shape[1], char_from_plate.shape[0]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(char_from_plate, cmap='gray')
    plt.title(f"Segmented Char (from Subsampled Plate)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(display_template, cmap='gray')
    plt.title(f"True Template ('{true_char_name}') Resized")
    plt.axis('off')
    
    title_text = (f"Analysis for char '{true_char_name}' (read as '{recognized_as}')\n"
                  f"Correlation with '{true_char_name}': {corr_with_true:.4f} | "
                  f"Correlation with '{recognized_as}': {corr_with_false:.4f}")
    plt.suptitle(title_text)
    plt.show()

# SECTION 3: MAIN ANALYSIS SCRIPT
def analyze_plate_subsampling():
    """Main analysis function using the final, correct methodology."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file_path = os.path.join(script_dir, 'results/p4.jpg')
    original_image = cv2.imread(image_file_path)
    if original_image is None: return

    print("Step 1: Establishing dynamic ground truth...")
    full_res_chars = segment_characters(original_image, scale_factor=1.0)
    ground_truth_plate, _ = recognize_plate(full_res_chars)
    if not ground_truth_plate:
        print("Fatal Error: Could not set ground truth.")
        return
    print(f"-> Dynamic Ground Truth established as: '{ground_truth_plate}'.\n")

    print("Step 2: Analyzing effects of plate subsampling...")
    subsample_rates = range(1, 21)
    results = []
    for rate in subsample_rates:
        subsampled_image = original_image[::rate, ::rate]
        scale_factor = 1.0 / rate
        segmented_chars = segment_characters(subsampled_image, scale_factor=scale_factor)
        recognized_text, avg_corr = recognize_plate(segmented_chars)
        is_correct = 1 if recognized_text == ground_truth_plate else 0
        results.append({"rate": rate, "recognized_text": recognized_text, "is_correct": is_correct, "avg_correlation": avg_corr})

    print("\n--- Subsampling Analysis Results ---")
    print(f"{'Rate (1 in N)':<15} | {'Recognized Plate':<20} | {'Avg. Correlation':<20} | {'Correct?':<10}")
    print("-" * 75)
    for res in results:
        correct_str = "Yes" if res['is_correct'] else "No"
        recognized_text = res['recognized_text'] if res['recognized_text'] else "N/A"
        print(f"{res['rate']:<15} | {recognized_text:<20} | {res['avg_correlation']:.4f}{'':<14} | {correct_str:<10}")

    # Step 3: Find failure point and visualize it
    first_failing_rate = None
    for i, res in enumerate(results):
        if not res['is_correct'] and i > 0 and results[i-1]['is_correct']:
            first_failing_rate = res['rate']
            
            failed_result = results[i]
            recognized_text = failed_result['recognized_text']
            failing_char_index = -1
            for char_idx in range(len(ground_truth_plate)):
                if char_idx >= len(recognized_text) or ground_truth_plate[char_idx] != recognized_text[char_idx]:
                    failing_char_index = char_idx
                    break
            
            print(f"\n-> Failure point identified at 1-in-{first_failing_rate} pixels. Generating comparison plots...")
            visualize_downsampling_methods(original_image, first_failing_rate)
            
            if failing_char_index != -1:
                true_char = ground_truth_plate[failing_char_index]
                read_as = recognized_text[failing_char_index] if failing_char_index < len(recognized_text) else 'N/A'
                print(f"-> The character '{true_char}' at index {failing_char_index} was misread as '{read_as}'.")
                if read_as != 'N/A':
                    visualize_failing_character(original_image, first_failing_rate, ground_truth_plate, failing_char_index, read_as)
            
            break
    
    # Step 4: Plot final performance graph
    rates, accuracies, correlations = [r['rate'] for r in results], [r['is_correct'] for r in results], [r['avg_correlation'] for r in results]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title('Recognition Performance vs. Plate Subsampling Rate')
    ax1.set_xlabel('Subsampling Rate (N, where 1 in N pixels is taken)')
    ax1.set_ylabel('Recognition Accuracy (1 = Correct, 0 = Incorrect)', color='tab:blue')
    ax1.plot(rates, accuracies, 'o-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.1, 1.1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Character Correlation Score', color='tab:red')
    ax2.plot(rates, correlations, 's--', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.xticks(rates)
    plt.grid(True, linestyle='--')
    fig.tight_layout()
    plt.show()


analyze_plate_subsampling()