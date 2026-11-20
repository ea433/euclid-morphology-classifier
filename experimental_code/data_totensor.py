import torch
from experimental_code.image_loading import image_array

# Convert the NumPy array to a PyTorch tensor
image_tensor = torch.from_numpy(image_array.copy())

print("Original shape:", image_tensor.shape)
print("Original data type:", image_tensor.dtype)

# Convert pixel values to floating-point numbers
image_tensor = image_tensor.float()

# Normalize pixel values to the range [0, 1]
image_tensor = image_tensor / 255.0

# Add a channel dimension
image_tensor = image_tensor.unsqueeze(0)

print("Final shape:", image_tensor.shape)
print("Final data type:", image_tensor.dtype)
print("Minimum:", image_tensor.min().item())
print("Maximum:", image_tensor.max().item())



