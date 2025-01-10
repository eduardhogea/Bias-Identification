import torch
from torchvision import datasets, transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import os


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize CIFAR-100 images to [0, 1] range for BLIP-2
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image):
    # Preprocess image for BLIP-2
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    # Generate caption
    generated_ids = model.generate(**inputs)
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

os.makedirs("output_textual", exist_ok=True)

num_images_to_caption = 5
for i in range(num_images_to_caption):
    original_image, label = dataset.data[i], dataset.targets[i]
    original_image_pil = Image.fromarray(original_image)

    caption = generate_caption(original_image_pil)

    image_filename = f"output_textual/image_{i}.png"
    original_image_pil.save(image_filename)

    caption_filename = f"output_textual/caption_{i}.txt"
    with open(caption_filename, "w") as f:
        f.write(caption)

    print(f"Image saved to {image_filename}, Caption saved to {caption_filename}")

print("Captions and images saved successfully!")
