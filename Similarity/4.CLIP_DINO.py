import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import numpy as np
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model and processor
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Load DINOv2 model and processor
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

#Retrieve all filenames
images = []
for root, dirs, files in os.walk('./val2017/'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)


#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features

def extract_features_dino(image):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)

# #Create 2 indexes.
# index_clip = faiss.IndexFlatL2(512)
# index_dino = faiss.IndexFlatL2(768)

#Iterate over the dataset to extract features X2 and store features in indexes
# for image_path in images:
#     img = Image.open(image_path).convert('RGB')
#     clip_features = extract_features_clip(img)
#     add_vector_to_index(clip_features,index_clip)
#     dino_features = extract_features_dino(img)
#     add_vector_to_index(dino_features,index_dino)

# #store the indexes locally
# faiss.write_index(index_clip,"clip.index")
# faiss.write_index(index_dino,"dino.index")

#Input image
source='dog.jpeg'
image = Image.open(source)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# #Load model and processor DINOv2 and CLIP
# processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

#Extract features for CLIP
with torch.no_grad():
    inputs_clip = processor_clip(images=image, return_tensors="pt").to(device)
    image_features_clip = model_clip.get_image_features(**inputs_clip)

#Extract features for DINOv2
with torch.no_grad():
    inputs_dino = processor_dino(images=image, return_tensors="pt").to(device)
    outputs_dino = model_dino(**inputs_dino)
    image_features_dino = outputs_dino.last_hidden_state
    image_features_dino = image_features_dino.mean(dim=1)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

image_features_dino = normalizeL2(image_features_dino)
image_features_clip = normalizeL2(image_features_clip)

#Search the top 5 images
index_clip = faiss.read_index("clip.index")
index_dino = faiss.read_index("dino.index")

#Get distance and indexes of images associated
d_dino,i_dino = index_dino.search(image_features_dino,5)
d_clip,i_clip = index_clip.search(image_features_clip,5)

np_imgs = np.array(images)
top5_images_dino = np_imgs[i_dino].tolist()[0]
top5_images_clip = np_imgs[i_clip].tolist()[0]

import matplotlib.pyplot as plt
def plot_images_with_similarity(image_paths, similarities, model_name, descript):
    # os.makedirs("./Top5/", exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
        image = Image.open(path)
        axes[i].imshow(image)
        axes[i].set_title(f"Similarity: {similarity:.2f}")
        axes[i].axis('off')
    
    # Use the current timestamp as a unique file name
    plt.savefig(f"./{model_name}/{descript}.png")

plot_images_with_similarity(top5_images_dino, d_dino.tolist()[0], "DINO", "DINO5_dog")
plot_images_with_similarity(top5_images_clip, d_clip.tolist()[0], "CLIP", "CLIP5_dog")