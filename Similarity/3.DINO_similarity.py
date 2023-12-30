import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
from functools import lru_cache
from datetime import datetime
from torchvision import transforms

#load the model and processor
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

#Populate the images variable with all the images in the dataset folder
images = []
for root, dirs, files in os.walk('./val2017'):
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

#Create Faiss index using FlatL2 type with 384 dimensions as this
#is the number of dimensions of the features
index = faiss.IndexFlatIP(384) 
# index = faiss.IndexFlatIP(768) # for base embedding
# index = faiss.IndexFlatIP(1024) # for large size embedding
# index = faiss.IndexFlatIP(1536) # for giant size embedding

import time
t0 = time.time()
for image_path in images:
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    add_vector_to_index( features.mean(dim=1), index)

print('Extraction done in :', time.time()-t0)

#Store the index locally
faiss.write_index(index,"vector.index")



#-------------------------------------------------
import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

#input image
image = Image.open('train.jpg')

# #Load the model and processor
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
# model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

#Extract the features
with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

#Normalize the features before search
embeddings = outputs.last_hidden_state
embeddings = embeddings.mean(dim=1)
vector = embeddings.detach().cpu().numpy()
vector = np.float32(vector)
faiss.normalize_L2(vector)

#Read the index file and perform search of top-3 images
index = faiss.read_index("vector.index")
d,i = index.search(vector,3)

np_imgs = np.array(images)
top3_images = np_imgs[i].tolist()[0]

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
    plt.savefig(f"./Top3/small_size_train.png")
    

# Call the function to plot the top 3 images
plot_images_with_similarity(top3_images, d.tolist()[0])

print('distances:', d, 'indexes:', i)