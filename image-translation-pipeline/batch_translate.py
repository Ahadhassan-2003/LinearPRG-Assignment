import os
import glob
import sys
import json
import argparse
from pathlib import Path

# Ensure the app module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.graph import run_pipeline
from app.utils.image_utils import load_image_from_bytes

def main():
    parser = argparse.ArgumentParser(description="Batch translate images")
    parser.add_argument("--ocr-lang", type=str, default="spa", help="Optional language code for Tesseract OCR (e.g. 'spa', 'chi_sim')")
    args = parser.parse_args()

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
                ocr_lang=args.ocr_lang,
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
                print(f"  [Success] Saved image to {out_path}")
                
                # Also save the JSON output
                json_name = os.path.splitext(out_name)[0] + ".json"
                json_path = os.path.join(output_dir, json_name)
                blocks = final_state.get("text_blocks") or []
                json_data = {
                    "detected_language": final_state.get("detected_language"),
                    "text_blocks": [b.model_dump() for b in blocks]
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"  [Success] Saved JSON to {json_path}")
            else:
                print(f"  [Warning] Pipeline completed but no output image was produced.")
                
        except Exception as e:
            print(f"  [Failed] Unexpected error: {e}")

if __name__ == "__main__":
    main()
