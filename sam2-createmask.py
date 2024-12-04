import numpy as np
from PIL import Image
import coremltools as ct
from coremltools.models import CompiledMLModel
import cv2
import os
import argparse
import ast

def parse_points_and_labels(points_str, labels_str):
    try:
        points = ast.literal_eval(points_str)
        if not all(isinstance(p, tuple) and len(p) == 2 for p in points):
            raise ValueError("Points must be list of (x,y) tuples")
        
        labels = ast.literal_eval(labels_str)
        if not all(isinstance(l, int) for l in labels):
            raise ValueError("Labels must be list of integers")
        
        if len(points) != len(labels):
            raise ValueError("Number of points must match number of labels")
        
        return points, labels
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid points or labels format: {str(e)}")

def load_image(image_path: str, target_size=(1024, 1024)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    original_size = image.size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image, original_size

def transform_coords(points, input_size=(1024, 1024)):
    width, height = input_size
    return [(x * width, y * height) for x, y in points]

def generate_mask(image_path: str, points: list, point_labels: list, model_dir: str = "modelsC"):
    compute_units = ct.ComputeUnit.CPU_AND_GPU
    
    print("Loading models...")
    image_encoder = CompiledMLModel(f"{model_dir}/SAM2_1SmallImageEncoderFLOAT16.mlmodelc", compute_units=compute_units)
    prompt_encoder = CompiledMLModel(f"{model_dir}/SAM2_1SmallPromptEncoderFLOAT16.mlmodelc", compute_units=compute_units)
    mask_decoder = CompiledMLModel(f"{model_dir}/SAM2_1SmallMaskDecoderFLOAT16.mlmodelc", compute_units=compute_units)
    
    print("Loading and preprocessing image...")
    image, original_size = load_image(image_path)
    image_embeddings = image_encoder.predict({"image": image})
    
    print("Processing prompt points...")
    transformed_points = transform_coords(points)
    points_array = np.array(transformed_points, dtype=np.float32)
    points_array = np.expand_dims(points_array, axis=0)
    labels_array = np.array(point_labels, dtype=np.int32)
    labels_array = np.expand_dims(labels_array, axis=0)
    
    prompt_input = {
        "points": points_array,
        "labels": labels_array
    }
    prompt_embeddings = prompt_encoder.predict(prompt_input)
    
    print("Generating mask...")
    decoder_input = {
        "image_embedding": image_embeddings["image_embedding"],
        "sparse_embedding": prompt_embeddings["sparse_embeddings"],
        "dense_embedding": prompt_embeddings["dense_embeddings"],
        "feats_s0": image_embeddings["feats_s0"],
        "feats_s1": image_embeddings["feats_s1"]
    }
    
    mask_output = mask_decoder.predict(decoder_input)
    
    scores = mask_output['scores'][0]
    best_mask_idx = np.argmax(scores)
    best_score = scores[best_mask_idx]
    print(f"Mask scores: {scores}")
    print(f"Selected mask {best_mask_idx} with score {best_score:.6f}")
    
    raw_mask = mask_output['low_res_masks'][0, best_mask_idx]
    prob_mask = 1 / (1 + np.exp(-raw_mask))
    
    print("Post-processing mask...")
    mask_image = Image.fromarray((prob_mask * 255).astype(np.uint8))
    mask_image = mask_image.resize(
        (original_size[0], original_size[1]),
        Image.Resampling.BICUBIC
    )
    prob_mask = np.array(mask_image) / 255.0
    
    prob_mask = np.rot90(prob_mask, k=-1)
    prob_mask = np.fliplr(prob_mask)
    
    threshold = -raw_mask.min() / (raw_mask.max() - raw_mask.min())
    binary_mask = (prob_mask > threshold).astype(np.float32)
    
    return binary_mask, prob_mask

def save_visualizations(image_path, binary_mask, prob_mask, points, point_labels, output_dir='output'):
    print(f"Saving output images to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    original_image = np.array(Image.open(image_path).convert('RGB'))
    h, w = original_image.shape[:2]
    
    if binary_mask.shape != (h, w):
        binary_mask = binary_mask.T
        prob_mask = prob_mask.T
    
    mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_dir, "mask.png"))
    
    # Create overlay with mask
    mask_colored = np.zeros_like(original_image)
    mask_colored[binary_mask > 0] = [255, 0, 0]
    blend = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
    
    # Draw points on the blend
    for point, label in zip(points, point_labels):
        x, y = int(point[0] * w), int(point[1] * h)
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(blend, (x, y), 5, color, -1)
        cv2.circle(blend, (x, y), 7, (255, 255, 255), 2)

    Image.fromarray(blend).save(os.path.join(output_dir, "overlay.png"))
    print(f"Done! Output saved to {output_dir}/mask.png and {output_dir}/overlay.png")

def main():
    parser = argparse.ArgumentParser(description='Run SAM2 segmentation')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--points', required=True, 
                      help='List of (x,y) coordinate tuples, e.g. "[(0.3,0.5), (0.5,0.5)]"')
    parser.add_argument('--labels', required=True,
                      help='List of point labels (1=foreground, 0=background), e.g. "[1, 1]"')
    parser.add_argument('--model-dir', default="modelsC", 
                      help='Directory containing CoreML models (default: modelsC)')
    parser.add_argument('--output-dir', default="output",
                      help='Directory for output images (default: output)')
    
    args = parser.parse_args()
    
    print("Parsing input arguments...")
    points, labels = parse_points_and_labels(args.points, args.labels)
    
    try:
        binary_mask, prob_mask = generate_mask(args.image, points, labels, args.model_dir)
        save_visualizations(args.image, binary_mask, prob_mask, points, labels, args.output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
