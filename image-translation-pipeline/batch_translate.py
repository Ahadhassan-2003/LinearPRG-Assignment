import os
import glob
import sys
from pathlib import Path

# Ensure the app module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.graph import run_pipeline
from app.utils.image_utils import load_image_from_bytes

def main():
    input_dir = r"e:\LinearPRG Assignment\sample_images_for_candidates"
    output_dir = r"e:\LinearPRG Assignment\image-translation-pipeline\batch_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in the input directory
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    # De-duplicate paths
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images to process.")
    
    for img_path in sorted(image_paths):
        print(f"\nProcessing {os.path.basename(img_path)}...")
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
                
            pil_image, fmt, width, height = load_image_from_bytes(image_bytes)
            
            final_state = run_pipeline(
                image_bytes=image_bytes,
                image_width=width,
                image_height=height,
                image_format=fmt,
                target_language="English",
                source_language="auto",
                filename=os.path.basename(img_path),
            )
            
            if final_state.get("error"):
                print(f"  [Error] {final_state['error']}")
                continue
                
            out_bytes = final_state.get("output_image_bytes")
            if out_bytes:
                out_name = "translated_" + os.path.basename(img_path)
                out_path = os.path.join(output_dir, out_name)
                with open(out_path, "wb") as f:
                    f.write(out_bytes)
                print(f"  [Success] Saved to {out_path}")
            else:
                print(f"  [Warning] Pipeline completed but no output image was produced.")
                
        except Exception as e:
            print(f"  [Failed] Unexpected error: {e}")

if __name__ == "__main__":
    main()
