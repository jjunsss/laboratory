# %matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt

import json
from pathlib import Path
import os
import shutil
import argparse
import os
from tqdm import tqdm

import ddp_utils
import numpy as np


@ddp_utils.ddp_on_and_off
def main():
    # load crawling dataset
    find_path = Path("path/to/dataset")
    image_paths = list(find_path.glob("**/*.jpg")) + list(find_path.glob("**/*.jpeg")) + list(find_path.glob("**/*.png"))
    image_paths = [str(path) for path in image_paths]  # Path 객체를 문자열로 변환
    print(f"total : {len(image_paths)}")
    
    image_length = len(image_paths)
    
    # sliced dataset by gpu number(local number)
    my_slice = ddp_utils.ddp_data_split(image_length, image_paths)
    
    # tqdm usage, "disable" for only main gpu print.
    pbar = tqdm(len(my_slice), disable=not ddp_utils.is_main_process())
    pbar.set_description("processing", )
    
    # iteration process
    for image_path in my_slice:
        # your works
        pass
    
    
if __name__ == "__main__":
    main()