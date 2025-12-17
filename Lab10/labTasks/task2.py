# Task 2 
# By Using Hugging Face model -
# 1. Text to image 2. Image to text 3. Image to image


# Task 2: Hugging Face Pipelines
# 1. Text → Image
# 2. Image → Text
# 3. Image → Image

from transformers import pipeline
from PIL import Image

# -------------------------------
# 1. TEXT TO IMAGE
# -------------------------------
text_to_image = pipeline(
    task="text-to-image",
    model="stabilityai/stable-diffusion-2-1"
)

prompt = "A futuristic city at sunset, cyberpunk style"
image = text_to_image(prompt)[0]["image"]
image.save("one.png")


# -------------------------------
# 2. IMAGE TO TEXT (Image Captioning)
# -------------------------------
image_to_text = pipeline(
    task="image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

input_image = Image.open("text_to_image.png")
caption = image_to_text(input_image)
print("Image Caption:", caption[0]["generated_text"])


# -------------------------------
# 3. IMAGE TO IMAGE
# -------------------------------
image_to_image = pipeline(
    task="image-to-image",
    model="stabilityai/stable-diffusion-2-1"
)

edited_image = image_to_image(
    image=input_image,
    prompt="Convert this image into a watercolor painting"
)[0]["image"]

edited_image.save("image_to_image.png")


