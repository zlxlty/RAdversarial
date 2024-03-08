# Label IDs and corresponding labels:
# https://image-net.org/challenges/LSVRC/2013/browse-synsets.php

import os #so that we can move between directories
import random #random selection
import shutil 
from math import floor, ceil
import time

from classifiers import get_target_model
from PIL import Image
import requests
import torch.nn as nn

source_folder = "datasets/source"
destination_folder = "datasets/benchmark1000rand"
desired_image_quantity = 1000
worst_case_quantity_reduction_to_account_for = .72 #testing showed that on average 77% of images survive.
starting_size = ceil(desired_image_quantity/worst_case_quantity_reduction_to_account_for) #number of images to be pulled from original data
print(f"\n Generating Testing Dataset:  ({desired_image_quantity} images)")
print(f"\t- Sampling {starting_size} images from \"{source_folder}\".")

# get to image folder:
random.seed('glorp')

#label file to dict:
label_dict = {}
f = open(os.path.join(source_folder, "benchmark_label_ids.txt"), "r")
lines = f.readlines()
f.close()

for line in lines:
    split_line = line.split(": ")
    label_dict[split_line[0]] = split_line[1].strip()

# create image list
img_name_list = os.listdir(os.path.join(source_folder, "images/benchmark_source_images"))
random.shuffle(img_name_list)
chosen_image_names = img_name_list[:starting_size]

# create image to true label dict
print(f"\t- Finding true labels images.")
image_to_true_label_dict = {}
for name in chosen_image_names:
    label_id = name.split(".")[0].split("_")[-1]
    label_text = label_dict[label_id]
    image_to_true_label_dict[name] = label_text

# progress bar taken and altered from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    start_time = time.time()
    update_space = 10

    #numbers cleaner
    def add_zero(val):
        val = str(val)
        if len(val) < 2:
            val = "0" + val
        return val

    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        if (iteration % update_space) == 0:
            if iteration >= 50:
                time_elapsed = (time.time() - start_time)
                remaining = (time_elapsed / iteration) * float(total) - time_elapsed
                hours = max(int(remaining // 3600), 0)
                mins = max(int(remaining// 60 - 60 * hours), 0)
                secs = max(int(remaining//1 - (60 * mins) - (3600 * hours)), 0)
                mins = add_zero(mins)
                secs = add_zero(secs)
                remaining = f"Done in: {hours}:{mins}:{secs}"
            else:
                remaining = "Done in: calculating..."
            print(f'\r{prefix} {bar} {percent}% {(" "*(5-len(str(percent))))}{remaining}         {suffix}', end = printEnd)
        else:
            print(f'\r{prefix} {bar} {percent}% {(" "*(5-len(str(percent))))}{suffix}', end = printEnd)

        
        
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()




#put images through target model and save image names it classifies correctly

target_model = get_target_model("MobileViT")
count = 0
correct_count = 0
total = len(chosen_image_names)
usable_image_names = []

for image_name in progressBar(chosen_image_names, prefix = "\t- Identifying misclassified images."):
    full_path = os.path.join(source_folder, "images/benchmark_source_images", image_name) 
    curr_image = Image.open(full_path) 
    try:
        input_tensor = target_model.preprocess(curr_image)
        logit = target_model.predict(input_tensor)
        max_class = logit.max(dim=1)[1].item()
        predicted_label = target_model.id2label(max_class)
        true_label = image_to_true_label_dict[image_name]
        is_correct = predicted_label == true_label
        usable_image_names.append(image_name)
    except:
        pass
    count += 1
        
#reduce to desired size
print("\t- Removing misclassified and excess images.")
try:
    usable_image_names = usable_image_names[:desired_image_quantity]
except:
    print("FAILED: insufficient starting images - lower worst_case_quantity_reduction_to_account_for")
    exit()

usable_image_names.sort()

#make image folder and label pairings file
print(f"\t- Adding {min(desired_image_quantity, len(usable_image_names))} images to folder.")
path = os.path.join(destination_folder,"images")
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

image_label_pairs = []
for name in usable_image_names:
    #copy images
    source_string = os.path.join(source_folder, "images/benchmark_source_images", name) 
    destination_string = os.path.join(destination_folder,"images", name)
    shutil.copyfile(source_string, destination_string)
    #prepare file lines
    line_string = f"{name}: {image_to_true_label_dict[name]}"
    image_label_pairs.append(line_string)

print("\t- Writing image/label pairs to file.")

f = open(os.path.join(destination_folder, "labels.txt"), "w")
for line in image_label_pairs:
    f.write(line + "\n")
f.close()

print("\nDataset Generation Complete.\n")