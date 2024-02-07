from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import faiss
import os
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer, AutoModel


# #read all image names
images = []
for root, dirs, files in os.walk('val2017'):
    for file in files:
        if file.endswith('jpg'):
            images.append(root  + '/'+ file)

#Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

#Load metaCLIP
#Load CLIP model, processor and tokenizer
processor = AutoProcessor.from_pretrained("facebook/metaclip-b16-fullcc2.5b")
model = AutoModel.from_pretrained("facebook/metaclip-b16-fullcc2.5b",  torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-b16-fullcc2.5b")

#! you have to use this code for writing your own dataset to faiss.
###############################################################
# #Define the emb variable to store embeddings
# emb = {}

# #Extract features of a given image
# def extract_features_clip(image):
#     with torch.no_grad():
#         inputs = processor(images=image, return_tensors="pt").to(device)
#         image_features = model.get_image_features(**inputs)
#         return image_features

# #Extract embeddings and store them in the emb variable
# for img_dir in tqdm(images):
#     with torch.no_grad():
#         img = Image.open(img_dir)
#         image_features = extract_features_clip(img)
#         emb[img_dir] = image_features

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
# faiss.write_index(index,"vector.index")


################################################################
#Read the index for using search algorithms.
index  = faiss.read_index("vector.index")


prompt = "a photo of a person and a train"

#Tokenize the prompt and extract features
text_token = tokenizer([prompt], return_tensors="pt").to(device)
text_features = model.get_text_features(**text_token)

#Preprocess the tensor
text_np = text_features.detach().cpu().numpy()
text_np = np.float32(text_np)
faiss.normalize_L2(text_np)

#Search the top 5 images
probs, indices = index.search(text_np, 50)

np_imgs = np.array(images)
top5_images = np_imgs[indices].tolist()[0]

print('probs',probs)
print('indice' ,indices)

import matplotlib.pyplot as plt
# def plot_images_with_similarity(image_paths, similarities, prompt):
#     # Ensure the directory exists
#     os.makedirs("metaCLIP/", exist_ok=True)  # 경로 수정

#     fig, axes = plt.subplots(6, 5, figsize=(3 * len(image_paths), 30))
#     # fig, axes = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
#     for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
#         image = Image.open(path)
#         axes[i].imshow(image)
#         axes[i].set_title(f"Similarity: {similarity:.2f}")
#         axes[i].axis('off')

#     # Use the prompt to create a unique file name
#     filename = prompt.replace(' ', '_')
#     print(f"filename: {filename}")
#     plt.savefig(f"metaCLIP/{filename}.png")
#     plt.close(fig)  # 추가: 그래프 닫기

def plot_images_with_similarity(image_paths, similarities, prompt, rows=10, cols=5):
    # 경로 확인 및 생성
    os.makedirs("metaCLIP/", exist_ok=True)

    # 이미지와 유사도를 나열할 플롯 생성
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # 다차원 배열을 1차원 배열로 변환

    for i, (path, similarity) in enumerate(zip(image_paths, similarities)):
        image = Image.open(path)
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f"Similarity: {similarity:.2f}", fontsize=10)
        ax.axis('off')

    # 남은 서브플롯 비활성화
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    # 프롬프트를 기반으로 한 고유 파일 이름 사용
    filename = prompt.replace(' ', '_')
    plt.savefig(f"metaCLIP/{filename}.png")
    plt.close(fig)  # 그래프 닫기
    
# 예를 들어 함수를 호출할 때
plot_images_with_similarity(top5_images, probs.tolist()[0], prompt)