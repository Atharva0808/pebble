from PIL import Image

def crop_and_square(image_path):
    img = Image.open(image_path).convert("RGBA")
    
    # Get bounding box of non-transparent pixels
    bbox = img.getbbox()
    if not bbox:
        print("Image is entirely transparent!")
        return
        
    # Crop to bounding box
    img_cropped = img.crop(bbox)
    
    # Make it square
    width, height = img_cropped.size
    max_dim = max(width, height)
    
    # Create new square transparent image
    square_img = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 0))
    
    # Paste cropped image into center of square
    offset_x = (max_dim - width) // 2
    offset_y = (max_dim - height) // 2
    square_img.paste(img_cropped, (offset_x, offset_y))
    
    # Save back
    square_img.save(image_path, "PNG")
    print(f"Processed {image_path}: Original size {img.size}, new size {square_img.size}")

# Process both files
crop_and_square(r"C:\Users\Atharva\.gemini\antigravity\scratch\pebble\web\public\pebble_logo.png")
crop_and_square(r"C:\Users\Atharva\.gemini\antigravity\scratch\pebble\web\app\icon.png")
