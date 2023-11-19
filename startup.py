import os
from natsort import natsorted, ns
folder_path = os.path.join(os.getcwd(), 'images')
list_files = os.listdir(folder_path)
print(list_files)