"""
Author: Markus Hausleitner

Usage example: python transform_image_inputs.py "G:\\Matcode\\101\\data" "/training/"

"""
import collections
import json
import string
import sys
from os import walk
import os.path
from os import path
import numpy as np

# we use json files as input-container for scene coordinates
file_extension = "json"

# training file name
train_file = 'train.txt'

# max height of our input images
max_height = 640

# max width of our input-images
max_width = 512

"""
Start processing root path 
"""


def main(root_path, input_txt_files_folder):
    scan_dir_for_files(root_path, input_txt_files_folder)


"""
Finally, perform a lookup for the input-image file
"""


def find_file_in_dir(fname, search_path) -> string:
    result = []
    for root, dir, files in os.walk(search_path):
        if fname in files:
            fpath = os.path.join(root, fname)
            result.append(fpath)
    # filenames are not unique by themself, but by their path, so a list as a  result is possible
    return '\n'.join(result)


"""
Finally create the train.txt file and create and fill training txt files
"""


def create_train_file(matched_fnames, input_txt_files_folder, root_dir):
    input_txt_files_folder = root_dir + input_txt_files_folder
    if not path.exists(input_txt_files_folder):
        os.mkdir(input_txt_files_folder)
    input_images = []
    for f in matched_fnames:
        input_images.append(find_file_in_dir(f[0], root_dir))
        file = open(input_txt_files_folder + f[0].replace(".tiff", ".txt"), "w+")  # write mode clears file if not empty
        file.write(f[1])
        file.close()
    tfile = open(input_txt_files_folder+train_file, "w+")
    tfile.write('\n'.join(input_images))
    tfile.close()


"""
Scan from root-directory recursive downwards for label files in form of json files in a Labels folder
"""


def scan_dir_for_files(root_dir, input_txt_files_folder):
    matched_fnames = []
    # iterate through all  dirs/subdirs in root folder to reach all labels
    for (dirpath, dirnames, filenames) in walk(root_dir):
        if 'Labels' in dirpath:  # we only want to get the labels folder
            for f in filenames:
                if f.endswith('.' + file_extension):
                    with open(dirpath + '\\' + f) as json_file:
                        data = json.load(json_file)
                        data_dict = json.loads(json.dumps(data))
                        labels = data_dict['Labels']
                        if not labels:
                            # if labels is empty we have no objects in the image
                            # -> skip further processing
                            continue
                        process_jsondata(labels, matched_fnames)
    create_train_file(matched_fnames, input_txt_files_folder, root_dir)


"""
polygons: 
first value is first Y 1, and X 2 
second value is first second X 1 and X2 
third value is first X 1 and Y2 
fourth value is Y 1 und Y2
"""


def get_bounding_box(polygons):
    dims = len(polygons)
    aab_xyz = np.zeros((2, 2))
    aab_xyz[0, :] = np.min(polygons, axis=0)
    aab_xyz[1, :] = np.max(polygons, axis=0)
    return aab_xyz, (aab_xyz[1, :] - aab_xyz[0, :])

"""
Fist we calculate the bounding box for given coordinates, then we transform them into a format usable by yolo
"""


def normalize_coords(polygons) -> string:
    bb_with_range = get_bounding_box(polygons)
    min_val = np.minimum(max_height, bb_with_range[0][:, 0])  # min returns mimal entry
    bb_with_range[0][:, 0] = np.maximum(1, min_val)  # minimum compares by element(matrizes)

    min_val = np.minimum(max_width, bb_with_range[0][:, 1])
    bb_with_range[0][:, 1] = np.maximum(1, min_val)
    res = calculate_yolobb(bb_with_range[0])
    return ' '.join([str(elem) for elem in res])


"""
Implementation of bbtoYolo.m from provided source by institute
Process JSON file to extract coordinates and filename
Returns normalized yolo-compatible array of relative coodinates (X,Y, W, H) 
"""


def calculate_yolobb(box):
    dw = 1. / max_height
    dh = 1. / max_width

    x = box[0, 0] + box[1, 0] / 2.0
    y = box[0, 1] + box[1, 1] / 2.0
    w = box[1, 0] - box[0, 0]
    h = box[1, 1] - box[0, 1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


"""
parse JSON file for images and polygons

"""


def process_jsondata(labels, matched_fnames):
    if isinstance(labels, collections.Mapping):  # if image contains only one label we have a dict
        try:
            matched_fnames.append((labels.get("imagefile", ""), str(labels.get('class')) + " " +
                                   normalize_coords(labels.get('poly'))))
        except TypeError as err:
            print("Type error for labels parsing imageFiles")
        except AttributeError:
            print("Attribute error for labels parsing imageFiles")
    else:  # we have multiple objects in a scene -> we have a list
        fname = ""
        polygons = []
        for label in labels:
            if fname is "":
                fname = label.get("imagefile", "")
            try:
                polys = label.get('poly')
                polygons.append(str(label.get('class')) + " " + normalize_coords(polys))
            except TypeError as err:
                print("Type error for label parsing imageFile: ")
            except AttributeError:
                print("Attribute error for label parsing imageFile: ")
        matched_fnames.append((fname, '\n'.join([str(elem) for elem in polygons])))


"""
Script takes two arguments: first one for the root directory containing all the data, second one the path relative to 
the first one containing all the training data after execution
"""


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
