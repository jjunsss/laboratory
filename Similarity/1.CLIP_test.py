# #import modules needed
# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"


# #Load the model and preprocessor
# model, preprocess = clip.load("ViT-B/32", device=device)

# print(device)

# #Preprocess the image
# image = preprocess(Image.open("Similarity/dog.jpeg")).unsqueeze(0).to(device)

# #Generate the tokenizer for our 5 classes
# # text = clip.tokenize(["a dog", "a cat", "a man", "a tree", "food"]).to(device)
# text = clip.tokenize(["a photo of a dog", "a photo of a cat", "a photo of a man", "a photo of a animal", "a photo of a puppy"]).to(device)
# labels = ["dog", "cat", "man", "animal", "puppy"]

# #Do not forget this if you don't want to overuse your GPU
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     #Compute score between the image and the tokenizer
#     logits_per_image, logits_per_text = model(image, text)
    
#     #get score with softmax, i.e. total of scores = 1
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# # print(f"Label probs: {probs}")
# # Combine the labels and probabilities in a readable format

# label_probs = {label: f"{prob * 100:.2f}%" for label, prob in zip(labels, probs[0])}

# print(label_probs)


#-----------------------------------
#* check for text similarity in various images.

# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# #my batch of images
# images = ['bus.jpg', 'car.jpg', 'chair.jpg', 'dog.jpeg', 'train.jpg']

# text = clip.tokenize(['a bus']).to(device)

# imgs = [preprocess(Image.open(img)) for img in images]
# with torch.no_grad():
#   #prediction with one prompt, several images. We need to stack the images together
#   logits_per_image, logits_per_text = model(torch.stack(imgs).to(device),text) # logits per image : a score of specific text about a lot of images / logits per text : a score of specific image about a lot of texts.
#   probs = logits_per_text.softmax(dim=-1).cpu().numpy()

# label_probs = {label: f"{prob * 100:.2f}%" for label, prob in zip(images, probs[0])}
# print(label_probs)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_folder = './val2017'
#Load all the images into an array
images = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith('jpg'):
            images.append(  root  + '/'+ file)
text = clip.tokenize(['a person waking on a beach']).to(device)
text_features = model.encode_text(text)
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
    sim = cos(image_features[0], text_features[0]).item()
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