# Noise2Void: A Tutorial on Image Denoising

This repository contains the source code for the Noise2Void deep learning method, along with a tutorial designed to teach students about image denoising.

## About Noise2Void

Noise2Void is a deep learning-based approach to image denoising that learns to restore images from noisy data alone. Unlike traditional methods that require pairs of clean and noisy images for training, Noise2Void leverages a clever "blind-spot" training strategy. This allows the model to learn the underlying structure of the image and effectively separate it from random noise, making it a powerful tool for a wide range of imaging applications.

The core implementation in the `noise2void/` directory is based on the original Noise2Void concept and is provided here for educational purposes.

## For Students: Your Task

Welcome! This project is designed to be a hands-on introduction to the world of image denoising. Your main task is to work through the `Denoising_Worksheet.ipynb` notebook.

### What You Will Learn:

1.  **What is Image Noise?** You will start by learning how to simulate different types of common image noise, such as Gaussian and Salt & Pepper noise.
2.  **Classical Denoising Techniques:** You will implement and experiment with traditional denoising algorithms like Gaussian filters, median filters, and the state-of-the-art BM3D method.
3.  **Deep Learning for Denoising:** You will train your own deep learning model for denoising using the Noise2Void method. You will prepare the data, build and train the model, and use it to denoise images.

### Instructions:

1.  **Open the Worksheet:** Start by opening the `Denoising_Worksheet.ipynb` file in a Jupyter environment.
2.  **Follow the Instructions:** The worksheet contains detailed explanations and instructions. You will find empty code cells marked with `# YOUR CODE HERE`. Your task is to fill in these cells with the correct code to complete the exercises.
3.  **Check Your Work:** If you get stuck, you can refer to the `Denoising_Tutorial.ipynb` notebook, which contains the complete solutions. Try to solve the exercises on your own first!
4.  **Experiment!** Feel free to experiment with different noise levels, filter parameters, and model settings to see how they affect the results.

Good luck, and have fun exploring the world of image denoising!