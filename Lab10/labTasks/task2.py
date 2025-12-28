# Task 2 
# By Using Hugging Face model -
# 1. Text to image 2. Image to text 3. Image to image


# Task 2: Hugging Face Pipelines
# 1. Text → Image
# 2. Image → Text
# 3. Image → Image

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import pipeline
from PIL import Image

# -------------------------------
# 1. TEXT TO IMAGE (Using Diffusers)
# -------------------------------
print("--- Loading Text-to-Image Model ---")
# Using SD v1.5 as it is more accessible for labs
model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline
# Use device="cuda" if you have an NVIDIA GPU, else "cpu"
pipe_t2i = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe_t2i.to("cpu") 

prompt = "A futuristic city at sunset, cyberpunk style"
print(f"Generating image for: {prompt}")
image = pipe_t2i(prompt).images[0]
image.save("one.png")


# -------------------------------
# 2. IMAGE TO TEXT (Using Transformers)
# -------------------------------
print("\n--- Running Image-to-Text ---")
# The task name for captioning is "image-to-text"
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Load the image we just created
input_image = Image.open("one.png")
caption = image_to_text(input_image)
print("Image Caption:", caption[0]["generated_text"])


# -------------------------------
# 3. IMAGE TO IMAGE (Using Diffusers)
# -------------------------------
print("\n--- Running Image-to-Image ---")
# We reuse the same base model but with the Img2Img pipeline
pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe_i2i.to("cpu")

# Strength (0.0 to 1.0) controls how much to change the original image
edited_image = pipe_i2i(
    prompt="Convert this image into a watercolor painting",
    image=input_image,
    strength=0.75
).images[0]

edited_image.save("image_to_image.png")
print("Task 2 Complete. Images saved as 'one.png' and 'image_to_image.png'")


