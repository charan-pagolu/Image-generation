# Mixed Image Generation Using Stable Diffusion

## Introduction
This project explores the use of Stable Diffusion, a state-of-the-art text-to-image generative model, to create hyper-realistic mixed images combining features from diverse demographic groups: Orientals, Indians, and Europeans. The project focuses on dataset preprocessing, implementing a pipeline for image generation, and visualizing results.

---

## Project Description
The goal of this project is to generate realistic images by combining features from different groups (e.g., Orientals, Indians, Europeans). The project includes:

- **Preprocessing:** Resizing images and splitting the dataset.
- **Model Training:** Training a CNN model for diffusion purposes (denoising).
- **Image Generation:** Using a pretrained Stable Diffusion model (`dreamlike-art/dreamlike-photoreal-2.0`) to generate mixed images.
- **Visualization:** Plotting accuracy and generated images.

---

## Summary of Tasks

### Dataset Preparation
- Resized images to uniform dimensions (64x64) and split them into training and testing sets.

### Model Integration
- Implemented the Stable Diffusion pipeline for hyper-realistic image generation.

### Image Generation
- Combined images from different groups and generated new images using prompts.

### Visualization
- Displayed the generated images for qualitative evaluation.

---

## Image Collection

- **Categories:** Orientals, Indians, Europeans.
- **Source:** Public datasets and curated websites like Getty Images.
- **Total Images:** Approximately 1,000 (split across categories).

---

## Proposed Method

This project integrates preprocessing, a custom-trained U-Net for diffusion-based denoising, and a pretrained Stable Diffusion model for image generation. Key steps include:

- Preprocessing images to a uniform size.
- Using textual prompts to guide image generation in the Stable Diffusion pipeline.
- Evaluating outputs visually to ensure realism and adherence to prompts.

---

## Implementation Details

### Step 1: Data Preparation
- **Task:** Resized all input images to a standard resolution (64x64) and split them into training (80%) and testing (20%) sets.

```python
from PIL import Image
import os, glob

def process_images(input_path, output_path, resize_dim=(64, 64)):
    os.makedirs(output_path, exist_ok=True)
    image_files = glob.glob(os.path.join(input_path, '*'))
    processed_count = 0
    for i, image_file in enumerate(image_files):
        try:
            with Image.open(image_file) as img:
                img_resized = img.resize(resize_dim)
                img_resized.save(os.path.join(output_path, f"image_{i + 1}.jpg"))
                processed_count += 1
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    return processed_count
```

- **Issue:** Missing or corrupted images led to errors during resizing, which were handled using `try-except` blocks.

### Step 2: Stable Diffusion Integration
- **Task:** Integrated the Stable Diffusion pipeline for generating realistic images based on textual prompts.

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully!")
```

- **Issue:** Cache migration caused pipeline hanging, resolved by resetting the cache and setting a custom cache directory.

### Step 3: Mixed Image Generation
- **Task:** Generated mixed images for specific combinations of groups (e.g., Orientals + Indians).

```python
def generate_mixed_images(pipe, data_dir, categories_to_mix, num_images_per_category=10, prompt=""):
    combined_images = []
    for category in categories_to_mix:
        category_path = os.path.join(data_dir, category)
        category_images = load_images_from_folder(category_path, num_images_per_category)
        combined_images.extend(category_images)

    generated_images = []
    for img in combined_images:
        img_tensor = preprocess(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            generated_image = pipe(prompt=prompt, init_image=img_tensor, strength=0.75, guidance_scale=12).images[0]
        generated_images.append(generated_image)
    return generated_images
```

- **Issue:** Slow inference time due to limited GPU memory, mitigated by reducing batch sizes and processing fewer images in parallel.

### Step 4: Visualization
- **Task:** Displayed generated images using Matplotlib.

```python
import matplotlib.pyplot as plt

def visualize_images(images, n=5, title="Generated Images"):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16)
    for i, img in enumerate(images[:n]):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
```

- **Issue:** Large image sizes caused display issues, resolved by resizing outputs for better visualization.

---

## Conclusion
This project successfully implemented Stable Diffusion to generate hyper-realistic mixed images combining features from multiple demographic groups. Challenges such as slow inference and corrupted data were addressed. Future work could include:

- Fine-tuning the model on a custom dataset.
- Using a larger dataset for training and evaluation.
- Automating prompt optimization for better outputs.

---
