import tensorflow as tf
import os
import cv2
import numpy as np

def load_image_and_masks(image_id_str, image_dir, mask_dir, target_size=(512, 512)):
    """Loads a grayscale image and its two masks (cardiac and thorax)."""
    img_path = os.path.join(image_dir, f"{image_id_str}.png")
    cardiac_path = os.path.join(mask_dir, f"{image_id_str}-cardiac.png")
    thorax_path = os.path.join(mask_dir, f"{image_id_str}-thorax.png")
    
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1) # Shape: (H, W, 1)

    # Load masks
    cardiac = cv2.imread(cardiac_path, cv2.IMREAD_GRAYSCALE)
    thorax = cv2.imread(thorax_path, cv2.IMREAD_GRAYSCALE)
    
    if cardiac is None or thorax is None:
        raise ValueError(f"Masks not found for {image_id_str}")
        
    cardiac = cv2.resize(cardiac, target_size, interpolation=cv2.INTER_NEAREST)
    thorax = cv2.resize(thorax, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Threshold to binary (0 or 1)
    cardiac = (cardiac > 127).astype(np.float32)
    thorax = (thorax > 127).astype(np.float32)
    
    # Combine masks into shape (H, W, 2)
    # Channel 0: cardiac, Channel 1: thorax
    mask = np.stack([cardiac, thorax], axis=-1)
    
    return img, mask

def tf_parse_wrapper(image_id, image_dir, mask_dir, target_size_w, target_size_h):
    def _parse(image_id_val, image_dir_val, mask_dir_val):
        image_id_str = image_id_val.numpy().decode('utf-8')
        img, mask = load_image_and_masks(
            image_id_str, 
            image_dir_val.numpy().decode('utf-8'), 
            mask_dir_val.numpy().decode('utf-8'),
            (target_size_w, target_size_h)
        )
        return img, mask
    
    img, mask = tf.py_function(
        _parse, 
        [image_id, image_dir, mask_dir], 
        [tf.float32, tf.float32]
    )
    img.set_shape([target_size_h, target_size_w, 1])
    mask.set_shape([target_size_h, target_size_w, 2])
    return img, mask

# Keras layers for spatial augmentation
rotator = tf.keras.layers.RandomRotation(factor=0.05, fill_mode='constant', fill_value=0.0)
translator = tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='constant', fill_value=0.0)

def augment_data(img, mask):
    """Applies data augmentation without cutting edges or flipping."""
    # Combine image and mask
    combined = tf.concat([img, mask], axis=-1)
    
    # Add batch dim for Keras layers: (1, H, W, 3)
    combined = tf.expand_dims(combined, 0)
    
    # Apply small random rotation and translation
    combined = rotator(combined)
    combined = translator(combined)
    
    # Remove batch dim: (H, W, 3)
    combined = tf.squeeze(combined, 0)
    
    # Split back
    img, mask = combined[..., :1], combined[..., 1:]
    
    # Photometric augmentation (only on image)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    # Ensure mask remains strictly binary after interpolation
    mask = tf.cast(mask > 0.5, tf.float32)
    
    return img, mask

def get_dataset(image_dir, mask_dir, batch_size=4, target_size=(512, 512), shuffle=True, augment=False):
    """Creates a tf.data.Dataset from image IDs."""
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_ids = [f.split('.')[0] for f in img_files]
    
    dataset = tf.data.Dataset.from_tensor_slices(image_ids)
    
    if shuffle:
        dataset = dataset.shuffle(len(image_ids))
        
    # Map the python function
    dataset = dataset.map(
        lambda x: tf_parse_wrapper(x, image_dir, mask_dir, target_size[0], target_size[1]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Map the augmentation function if requested
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
