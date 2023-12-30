# import torch
# import clip
# from PIL import Image
# import torch.nn as nn
# import os
# import itertools
# import matplotlib.pyplot as plt
# from functools import lru_cache
# from datetime import datetime

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image1 = "gv70.png"
# image2= "chair.jpg"

# cos = torch.nn.CosineSimilarity(dim=0)

# image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to(device)
# image1_features = model.encode_image( image1_preprocess)

# image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to(device)
# image2_features = model.encode_image( image2_preprocess)

# similarity = cos(image1_features[0],image2_features[0]).item()
# similarity = (similarity+1)/2

# # Function to plot images and their similarity scores
# def plot_images_with_similarity(image_paths, similarities):
#     # Ensure the directory exists
#     os.makedirs("./Top3/", exist_ok=True)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
#         image = Image.open(path)
#         axes[i].imshow(image)
#         axes[i].set_title(f"Similarity: {similarity:.2f}")
#         axes[i].axis('off')
    
#     # Use the current timestamp as a unique file name
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f"./Top3/Top3_{timestamp}.png")

# # Call the function to plot the top 3 images
# # plot_images_with_similarity(top_3.keys(), top_3.values())

# print("Image similarity", similarity)

#-----------------------------------
#* pick top-3 images for cosine similarity in various images.

import torch
import clip
from PIL import Image
import os
import itertools
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import lru_cache
from datetime import datetime
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_folder = './val2017'
#Load all the images into an array
images = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith('jpg'):
            images.append(  root  + '/'+ file)
            
# text = clip.tokenize(['a person waking on a beach']).to(device)
# text_features = model.encode_text(text)

preprocess_pipeline = transforms.Compose([
    transforms.CenterCrop(112),  # for example, center crop to 224x224
    preprocess,  # your existing preprocessing steps
])

image = Image.open("train.jpg")
input_image = preprocess_pipeline(image).unsqueeze(0).to(device)

#Embedding of the input image
# input_image = preprocess(Image.open("train.jpg")).unsqueeze(0).to(device)
input_image_features = model.encode_image(input_image)

result = {}
cos = torch.nn.CosineSimilarity(dim=0)

@lru_cache(maxsize=None)  # Setting maxsize to None means the cache can grow without bound
def get_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

#For each image, compute its cosine similarity with the prompt and store the result in a dict
# Now, instead of directly calling the model inside the loop:
for img in tqdm(images):
    image_features = get_image_features(img)
    sim = cos(image_features[0], input_image_features[0]).item()
    sim = (sim+1)/2
    
    result[img]=sim

#Sort the dict and retrieve the first 3 values
sorted_value = sorted(result.items(), key=lambda x:x[1], reverse=True)
sorted_res = dict(sorted_value)
top_3 = dict(itertools.islice(sorted_res.items(), 3))

# Function to plot images and their similarity scores
def plot_images_with_similarity(image_paths, similarities):
    # Ensure the directory exists
    os.makedirs("./Top3/", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
        image = Image.open(path)
        axes[i].imshow(image)
        axes[i].set_title(f"Similarity: {similarity:.2f}")
        axes[i].axis('off')
    
    # Use the current timestamp as a unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./Top3/Top3_{timestamp}.png")

# Call the function to plot the top 3 images
plot_images_with_similarity(top_3.keys(), top_3.values())

print(top_3)