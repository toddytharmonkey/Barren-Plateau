import os
from VQEMonteCarlo import * 
"""
Run this file to graph all the prob_###... runs!
"""

if __name__ == "__main__":

    from PIL import Image, ImageDraw, ImageFont
    import os
    import glob

    # Settings
    base_folder = "./"  # Change this to your base folder path
    folder_pattern = "prob_*_HEA2_q8_l20_shots1_postTrue_{file_pattern}_aware_thetas10_version*"
    png_name = "all_run_plot.png"
    output_name = "combined_images_{file_pattern}.png"

    file_patterns = ["xxz_1_1_05", "z0z1"]

    for file_pattern in file_patterns:
        # Search for folders matching the pattern
        search_pattern = os.path.join(base_folder, folder_pattern.format(file_pattern=file_pattern))
        folders = sorted(glob.glob(search_pattern))

        # List to store images
        images = []

        # Open each image and append to the list
        for folder in folders:
            image_path = os.path.join(folder, png_name)
            if os.path.exists(image_path):
                img = Image.open(image_path)
                images.append(img)
            else:
                print(f"Image not found in folder: {folder}")

        # Calculate total width and height for the combined image
        max_width = max([img.width for img in images])
        max_height = max([img.height for img in images])

        total_width = max_width * 5  # 5 images per row
        total_height = max_height * 2 + 100  # 2 rows + padding for title

        # Create a blank canvas
        combined_img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(combined_img)
        font_size = 40
        font = ImageFont.truetype("arial.ttf", font_size)
        title_font = ImageFont.truetype("arial.ttf", font_size + 10)

        # Draw the title
        title = f"{file_pattern} | 20 Layers | 8 Qubits | HEA2 Ansatz"
        title_width, title_height = draw.textsize(title, font=title_font)
        draw.text(((total_width - title_width) / 2, 20), title, font=title_font, fill='black')

        # Paste each image on the canvas and annotate with label
        x_offset, y_offset = 0, 120  # Start pasting images below the title
        for idx, img in enumerate(images):
            combined_img.paste(img, (x_offset, y_offset))
            label = f"probability {idx * 0.1:.1f}"
            draw.text((x_offset + 10, y_offset + 10), label, font=font, fill='black')
            
            x_offset += img.width
            if (idx + 1) % 5 == 0:  # Move to next row after 5 images
                x_offset = 0
                y_offset += img.height

        # Save the combined image
        combined_img.save(output_name.format(file_pattern=file_pattern))

        print(f"Combined image for {file_pattern} saved as {output_name.format(file_pattern=file_pattern)}")
