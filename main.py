# Import necessary libraries
import os
import numpy as np
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import glob
from torchvision import transforms
import torch
from diffusers import StableDiffusionPipeline

# ---------------------- Data Preparation ---------------------- #

# Define base paths for dataset
base_path = '/content/drive/MyDrive/Colab Notebooks/project'
processed_dir = os.path.join(base_path, 'Processed_Dataset')
os.makedirs(processed_dir, exist_ok=True)

categories = {
    "Europeans": os.path.join(base_path, 'europian'),
    "Indians": os.path.join(base_path, 'Indian'),
    "Orientals": os.path.join(base_path, 'oriental')
}

# Resize images
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

# Split dataset into train and test
def split_dataset(category_name, input_path, output_dir, test_size=0.2):
    images = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    train_images, test_images = train_test_split(images, test_size=test_size)

    train_dir = os.path.join(output_dir, 'train', category_name)
    test_dir = os.path.join(output_dir, 'test', category_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(img, train_dir)
    for img in test_images:
        shutil.copy(img, test_dir)

# Process and split the dataset
for category, path in categories.items():
    processed_path = os.path.join(processed_dir, category)
    process_images(path, processed_path)
    split_dataset(category, path, processed_dir)

print("Image processing and dataset splitting complete.")

# ---------------------- Model Training ---------------------- #

def train_model(train_images, val_images, epochs=10, batch_size=32):
    # Normalize images
    train_images = np.array([
        np.array(Image.open(img).resize((64, 64)).convert("RGB")) / 255.0
        for img in train_images
    ])
    val_images = np.array([
        np.array(Image.open(img).resize((64, 64)).convert("RGB")) / 255.0
        for img in val_images
    ])

    # Define a simple CNN model
    model = tf.keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # Assuming 3 classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_images,
        np.zeros(len(train_images)),  # Dummy labels
        validation_data=(val_images, np.zeros(len(val_images))),  # Dummy labels
        epochs=epochs,
        batch_size=batch_size
    )
    return history

# ---------------------- Plot Training Metrics ---------------------- #
def plot_training_metrics(history):
    """
    Plot training and validation accuracy/loss.
    """
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------- Stable Diffusion ---------------------- #

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully!")

# ---------------------- Generate Mixed Images ---------------------- #

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

# ---------------------- Visualization ---------------------- #

def visualize_images(images, n=5, title="Generated Images"):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16)
    for i, img in enumerate(images[:n]):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# ---------------------- Execution ---------------------- #

# Training and Validation
train_images = glob.glob(os.path.join(processed_dir, 'train', '*/*'))
val_images = glob.glob(os.path.join(processed_dir, 'test', '*/*'))

history = train_model(train_images, val_images, epochs=10, batch_size=32)
plot_training_metrics(history)

# Generate and Visualize Images
combinations_and_prompts = [
    (["Orientals", "Indians"], "A hyper-realistic photo combining Orientals and Indians."),
    (["Orientals", "Europeans"], "A hyper-realistic photo combining Orientals and Europeans."),
    (["Indians", "Europeans"], "A hyper-realistic photo combining Indians and Europeans."),
    (["Orientals", "Indians", "Europeans"], "A hyper-realistic photo combining Orientals, Indians, and Europeans."),
]

for categories, prompt in combinations_and_prompts:
    print(f"Generating for combination: {categories}")
    mixed_images = generate_mixed_images(pipe, processed_dir, categories, num_images_per_category=10, prompt=prompt)
    visualize_images(mixed_images, n=5, title=f"Features: {', '.join(categories)}")
