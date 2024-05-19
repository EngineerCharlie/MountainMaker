import os
import cv2
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY


def array_to_svg(image_array, output_folder, original_filename):
    try:
        # Convert to grayscale if the image is in color
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to a binary image
        _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        print(f"Processing {original_filename}...please wait")

        # Create Bitmap from the binary image
        bm = Bitmap(binary_image)  # No need for from_array
        plist = bm.trace(turnpolicy=POTRACE_TURNPOLICY_MINORITY, alphamax=0.5, opttolerance=0.2)

        os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists

        # Construct the output filename using the original filename
        base_filename = os.path.splitext(original_filename)[0]
        output_filename = f"{base_filename}-output.svg"
        output_path = os.path.join(output_folder, output_filename)  # Path for the output file

        with open(output_path, "w") as fp:
            fp.write(
                f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{binary_image.shape[1]}" height="{binary_image.shape[0]}" viewBox="0 0 {binary_image.shape[1]} {binary_image.shape[0]}">'
            )

            for i, curve in enumerate(plist):
                parts = []
                fs = curve.start_point
                parts.append(f"M{fs.x},{fs.y}")

                for segment in curve:
                    if segment.is_corner:
                        a = segment.c
                        b = segment.end_point
                        parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                    else:
                        c1 = segment.c1
                        c2 = segment.c2
                        end = segment.end_point
                        parts.append(f"C{c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y}")

                fp.write(
                    f'<path stroke="none" fill="#000000" fill-rule="nonzero" d="{"".join(parts)}"/>'
                )

            fp.write("</svg>")
        print(f"SVG file saved as {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")
