import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from noise2void.model import Noise2Void

def main():
    # Define parameters directly in the code
    input_file = '/lhome/ahmadfn/Paris/data/Bosh/NoisyA/reco64_Even/volume.hdf5'
    dataset_name = 'Volume'
    slice_index = 1200

    # Training hyperparameters
    patch_size = 64
    n_patches = 1000
    mask_percentage = 0.3
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Set to None to train a new model
    model_path = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    n2v = Noise2Void(
        device=device,
        patch_size=patch_size,
        n_patches=n_patches,
        mask_percentage=mask_percentage,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Load the HDF5 file and extract the specified slice
    with h5py.File(input_file, 'r') as f:
        noisy_slice = np.array(f[dataset_name][:,slice_index,:])

    # Load pre-trained model or train a new one
    if model_path is not None:
        print(f"Loading pre-trained model from {model_path}")
        n2v.load_model(model_path)
    else:
        print("Training Noise2Void model...")
        training_stats = n2v.train(noisy_slice=noisy_slice, num_epochs=num_epochs)

    # Denoise the slice
    print("Denoising the slice...")
    denoised_slice = n2v.denoise(noisy_slice)

    # Save individual images using plt.imsave
    plt.imsave("NoisySlice.png", noisy_slice, cmap='gray')
    plt.imsave("DenoisedSlice.png", denoised_slice, cmap='gray')

    # Create a combined figure with both images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_slice, cmap='gray')
    plt.title('Noisy Slice')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_slice, cmap='gray')
    plt.title('Denoised Slice')
    plt.axis('off')

    plt.tight_layout()
    # Save the combined figure
    plt.savefig("Comparison.png")
    # Display the combined figure
    plt.show()

if __name__ == "__main__":
    main()