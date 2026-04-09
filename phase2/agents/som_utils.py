import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_som_image(image: Image.Image):
    """
    Takes a PIL Image and generates a Set-of-Mark (SoM) augmented image with numbered tags.
    Returns the marked image and a dictionary mapping tag_id -> (x_center, y_center).
    """
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    ui_map = {}
    
    try:
        import cv2
        open_cv_image = np.array(image.convert("RGB"))
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text and UI bounding boxes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        idx = 1
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter extremely small or extremely large boxes to avoid noise or selecting the entire screen
            if 15 < w < image.width * 0.9 and 15 < h < image.height * 0.9:
                cx, cy = x + w//2, y + h//2
                ui_map[idx] = (cx, cy)
                
                # Draw Box
                draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
                # Draw Tag background
                draw.rectangle([x-2, max(0, y-15), x+24, max(0, y-15)+15], fill=(255, 0, 0))
                # Draw Tag Text
                draw.text((x+2, max(0, y-15)), str(idx), fill=(255, 255, 255))
                
                idx += 1
                if idx > 300: # hard limit to prevent excessive clutter
                    break
                    
    except ImportError:
        # Fallback if cv2 is not available: Do not augment visually, just return empty map
        print("[Warning] OpenCV (cv2) not installed. Skipping Set-of-Mark visual augmentation.")
        return image, {}
    except Exception as e:
        print(f"[Warning] SoM Augmentation failed: {e}")
        return image, {}
        
    return marked_image, ui_map