# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 01:33:21 2024
@author: zzha962
"""
import numpy as np

# Example 3D MRI volume
mri_volume = np.random.rand(64, 64, 64)  # Replace with actual MRI data

# Step 1: Compute 3D Fourier Transform
k_space = np.fft.fftn(mri_volume)

# Step 2: Shift the zero frequency component to the center
k_space_shifted = np.fft.fftshift(k_space)

# Step 3: Extract magnitude and phase
magnitude = np.abs(k_space_shifted)
phase = np.angle(k_space_shifted)

# Step 4: Extract specific k-space features (e.g., central region)
low_freq_mask = (magnitude > np.percentile(magnitude, 95))  # Example: top 20% of frequencies
low_freq_features = magnitude[low_freq_mask]

# Print results
print("K-Space Shape:", k_space.shape)
print("Low-Frequency Features Shape:", low_freq_features.shape)
