import os #so that we can move between directories
import random #random selection
import shutil 
from math import floor, ceil
import time

from classifiers import get_target_model
from PIL import Image
import requests
import torch.nn as nn

#input desired amount of output images
source_folder = "datasets/source/images/surrogate_source_images"
destination_folder = "datasets/surrogate"
desired_image_quantity = 80000
#multiplier to account for loss when model does not accept image
increase_multiplier = 1.08
starting_size = ceil(desired_image_quantity * increase_multiplier)

print(f"\n Generating Surrogate Training Dataset:  ({desired_image_quantity} images)")

original_working_directory = os.getcwd()
random.seed('glorp')

#take starting_size random images
print(f"\t- Sampling {starting_size} images from \"{source_folder}\".")
img_name_list = os.listdir(source_folder)
random.shuffle(img_name_list)
chosen_image_names = img_name_list[:starting_size]


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

target_model = get_target_model("MobileViT")
count = 0
total = len(chosen_image_names)
usable_image_names = []
pairings_dict = {}
success_count = 0
done = False

#get labels for images by getting predictions from model. stop after desired amount is reached
for image_name in progressBar(chosen_image_names, prefix = "\t- Getting labels from target model."):
    if done:
        pass
    else:
        full_path = os.path.join(source_folder, image_name) 
        curr_image = Image.open(full_path)
        try:
            input_tensor = target_model.preprocess(curr_image)
            logit = target_model.predict(input_tensor)
            max_class = logit.max(dim=1)[1].item()
            label = target_model.id2label(max_class)

            usable_image_names.append(image_name)
            pairings_dict[image_name] = label
            success_count += 1
            if success_count == desired_image_quantity:
                done = True
        except:
            pass
        count += 1

#remove excess:

if success_count < desired_image_quantity:
    print("DID NOT HAVE ENOUGH IMAGES. INCREASE MULTIPLIER")
    exit()

#ensures nicely alphabetized list (not neccessary)
usable_image_names.sort()

#add images to folder and write pairings to file
print(f"\t- Adding images to folder")
path = os.path.join(destination_folder,"images")
if os.path.exists(path):
    shutil.rmtree(path)
try:
    os.mkdir(path)
except:
    print(f"Failed to make path \"{path}\"")
    exit()

image_label_pairs = []
for name in usable_image_names:
    source_string = os.path.join(source_folder, name)
    split_name = name.split("test")
    new_name = f"{split_name[0]}surrogate{split_name[1]}"
    destination_string = os.path.join(destination_folder,"images", new_name)
    shutil.copyfile(source_string, destination_string)

    label = pairings_dict[name]
    image_label_pair = f"{new_name}: {label}"
    image_label_pairs.append(image_label_pair)

print("\t- Writing image/label pairs to file.")

f = open(os.path.join(destination_folder,"labels.txt"), "w")
for line in image_label_pairs:
    f.write(line + "\n")
f.close()

print("\nDataset Generation Complete.\n")





