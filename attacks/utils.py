import os

def create_folders(file_path):
    # Split the file path into individual folder names
    folders = file_path.split("/")
    # Remove the filename (last element) from the list
    folders = folders[1:-1]

    # Iterate over each folder and create it if it doesn't exist
    path_so_far = "/"
    for folder in folders:
        path_so_far = os.path.join(path_so_far, folder)
        if not os.path.exists(path_so_far):
            os.mkdir(path_so_far)