import os
import tensorflow as tf
from dataset import get_dataset
from model import unet_model, bce_dice_loss, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def main():
    # Setup paths
    base_dir = "dataset"
    train_img_dir = os.path.join(base_dir, "training", "images")
    train_mask_dir = os.path.join(base_dir, "training", "annfiles_mask")
    val_img_dir = os.path.join(base_dir, "validation", "images")
    val_mask_dir = os.path.join(base_dir, "validation", "annfiles_mask")

    # Hyperparameters
    batch_size = 4
    epochs = 50
    target_size = (512, 512)
    os.makedirs('models', exist_ok=True)

    print("Loading datasets...")
    train_dataset = get_dataset(train_img_dir, train_mask_dir, batch_size=batch_size, target_size=target_size, shuffle=True, augment=True)
    val_dataset = get_dataset(val_img_dir, val_mask_dir, batch_size=batch_size, target_size=target_size, shuffle=False, augment=False)

    print("Building model...")
    model = unet_model(input_shape=(target_size[0], target_size[1], 1), num_classes=2)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint('models/best_unet_model_aug_v2.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    print("Training finished!")

if __name__ == "__main__":
    main()
