from PIL import Image
from tqdm import tqdm
import torch
import clip
import numpy as np
import faiss
import os

# #read all image names
images = []
for root, dirs, files in os.walk('val2017'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)

#Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# # #Define the emb variable to store embeddings
# emb = {}

# #Extract embeddings and store them in the emb variable
# for img in tqdm(images):
#     with torch.no_grad():
#         image = preprocess(Image.open(img)).unsqueeze(0).to(device)
#         image_features = model.encode_image(image)
#         emb[img] = image_features

# # L2 algorithm. / InnerProduct algorithm.
# # index = faiss.IndexFlatL2(512)
# index = faiss.IndexFlatIP(512) 

# #Convert embeddings and add them to the index
# for key in emb:
#     #Convert to numpy
#     vector = emb[key].detach().cpu().numpy()
#     #Convert to float32 numpy
#     vector = np.float32(vector)
#     #Normalize vector: important to avoid wrong results when searching
#     faiss.normalize_L2(vector) # This is adapted to both vectors like image embed and token.
#     #Add to index
#     index.add(vector)

# #Store the index locally
# faiss.write_index(index,"baseCLIP.index")

#Read the index for using search algorithms.
index  = faiss.read_index("baseCLIP.index")

prompt = "a photo of a person and a train"
#Tokenize the prompt to search using CLIP
text_token = clip.tokenize(prompt).to(device)
text_features = model.encode_text(text_token)

#Preprocess the tensor
text_np = text_features.detach().cpu().numpy()
text_np = np.float32(text_np)
faiss.normalize_L2(text_np)

#Search the top 5 images
probs, indices = index.search(text_np, 5)

np_imgs = np.array(images)
top5_images = np_imgs[indices].tolist()[0]

print('probs',probs)
print('indice' ,indices)

import matplotlib.pyplot as plt
def plot_images_with_similarity(image_paths, similarities, prompt):
    # Ensure the directory exists
    os.makedirs("CLIP/", exist_ok=True)  # 경로 수정

    fig, axes = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
    for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
        image = Image.open(path)
        axes[i].imshow(image)
        axes[i].set_title(f"Similarity: {similarity:.2f}")
        axes[i].axis('off')

    # Use the prompt to create a unique file name
    filename = prompt.replace(' ', '_')
    print(f"filename: {filename}")
    plt.savefig(f"CLIP/{filename}.png")
    plt.close(fig)  # 추가: 그래프 닫기

# 예를 들어 함수를 호출할 때
plot_images_with_similarity(top5_images, probs.tolist()[0], prompt)