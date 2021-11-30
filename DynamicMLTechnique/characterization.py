import os
import sys
import math
import json
import csv
from enum import Enum
from statistics import median
from collections import deque, namedtuple
from typing import Tuple, Iterable

import numpy as np
from PIL import Image
from plantcv import plantcv as pcv


DPI = 120
"""DPI for images analyzed"""
DEBUG = True
"""If True, output images and print messages will appear"""

class OutputColumn(Enum):
    """Enum for CSV column names"""
    FILE_NAME = 'File name'
    ROOT_ID = 'Root no.'
    LENGTH = 'Root length (in.)'
    ANGLE = 'Root angle (deg)'
    MAX_DIAMETER = 'Max root diameter (in.)'
    MED_DIAMETER = 'Median root diameter (in.)'
    NUM_REGIONS = 'No. of regions'

class OutputRow(namedtuple('OutputRow', [col.name for col in OutputColumn])):
    """Named tuple representing a row entry in the output CSV"""
    __slots__ = ()

def count_num_regions(mask: np.ndarray) -> int:
    """Counts the number of continuous regions in a mask"""
    D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    R, C = mask.shape
    visited = set()
    num_regions = 0
    for i, row in enumerate(mask):
        for j, cell in enumerate(row):
            if not cell or (i, j) in visited:
                continue
            num_regions += 1
            visited.add((i, j))
            q = deque([(i, j)])
            while len(q):
                r0, c0 = q.popleft()
                for dr, dc in D:
                    r = r0 + dr
                    c = c0 + dc
                    if r < 0 or r >= R or c < 0 or c >= C:
                        continue
                    if not mask[r, c] or (r, c) in visited:
                        continue
                    visited.add((r, c))
                    q.append((r, c))
    return num_regions

def isolate_mask_region(mask: np.ndarray) -> np.ndarray:
    """Creates a mask of only the largest continuous region in the mask"""
    D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    R, C = mask.shape
    visited = set()
    best_mask = mask
    best_mask_size = 0
    for i, row in enumerate(mask):
        for j, cell in enumerate(row):
            if not cell or (i, j) in visited:
                continue
            filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
            visited.add((i, j))
            q = deque([(i, j)])
            filtered_mask[i, j] = 1
            mask_size = 1
            while len(q):
                r0, c0 = q.popleft()
                for dr, dc in D:
                    r = r0 + dr
                    c = c0 + dc
                    if r < 0 or r >= R or c < 0 or c >= C:
                        continue
                    if not mask[r, c] or (r, c) in visited:
                        continue
                    visited.add((r, c))
                    q.append((r, c))
                    filtered_mask[r, c] = 1
                    mask_size += 1
            if mask_size > best_mask_size:
                best_mask_size = mask_size
                best_mask = filtered_mask
    return best_mask

def max_index(iter: Iterable) -> int:
    """Returns the index of the maximum element"""
    return iter.index(max(iter))

def euclid_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculates Euclidean distance between two points"""
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)

def calc_radii(mask: np.ndarray, skeleton_path) -> float:
    """Calculates the distance from each point in a skeleton path to the edge of the mask"""
    edge_mask = generate_edge_mask(mask)
    edge = np.transpose(edge_mask.nonzero())
    return np.array([min(
        euclid_dist(path_point, edge_point) for edge_point in edge
    ) for path_point in skeleton_path])

def generate_edge_mask(mask: np.ndarray) -> np.ndarray:
    """Generates a binary mask consisting of only edge pixels"""
    D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    R, C = mask.shape

    edge_mask = np.zeros(mask.shape)
    
    for i, row in enumerate(mask):
        for j, cell in enumerate(row):
            if not cell:
                continue
            if cell and (i == 0 or i == R - 1 or j == 0 or j == C - 1):
                edge_mask[i, j] = 1
                continue
            for dr, dc in D:
                if not mask[i + dr, j + dc]:
                    edge_mask[i, j] = 1
                    break
    return edge_mask

def characterize(filename: str, root_no: int, mask: np.ndarray, debug_path=None, dpi=DPI) -> dict:
    """Performs characterization on a binary mask"""
    num_regions = count_num_regions(mask)
    if debug_path:
        print(f'Characterization for {filename} - Root #{root_no}')
        # Image.fromarray(mask * 255, mode='L').save(debug_path)
        if num_regions != 1:
            print(f'Warning: Expected 1 region in mask but got {num_regions}')
        print()
    if num_regions == 0:
        return OutputRow(filename, root_no, 0., 0., 0., 0., 0)
    if num_regions > 1:
        mask = isolate_mask_region(mask)

    skeleton = pcv.morphology.skeletonize(mask=mask)
    img1, seg_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=10, mask=mask)
    if debug_path:
        Image.fromarray(seg_img).save(debug_path)

    pcv.morphology.segment_path_length(skeleton, segment_objects)
    pcv.morphology.segment_angle(skeleton, segment_objects)

    output = pcv.outputs.observations['default']
    longest_index = max_index(output['segment_path_length']['value'])

    path_length = output['segment_path_length']['value'][longest_index] / dpi
    path_angle = output['segment_angle']['value'][longest_index]
    if debug_path:
        print(f'Root length (in.):\n\t{path_length}')
        print(f'Root angle (deg):\n\t{path_angle}')

    path = np.array([edge[0] for edge in segment_objects[longest_index]])
    diameters = 2 / dpi * np.array(calc_radii(np.transpose(mask), path))

    if debug_path:
        print(f'Max root diameter (in.):\n\t{max(diameters)}')
        print(f'Median root diameter (in.):\n\t{median(diameters)}')
        print('------------')

    return OutputRow(
        filename,
        root_no,
        path_length,
        path_angle,
        max(diameters),
        median(diameters),
        num_regions
    )

def get_image_name(filepath: str):
    """Retrieves the image name from its file path"""
    return os.path.basename(filepath)

def characterize_image(image_path: str, debug_dir=None):
    """Characterizes a mask file in image format (e.g. PNG, JPG)"""
    img, _, _ = pcv.readimage(filename=image_path, mode='gray')
    image_name = get_image_name(image_path)
    debug_path = f'./{debug_dir}/{image_name}_{0}.png' if debug_dir else None
    return characterize(image_name, 0, img, debug_path=debug_path)

def characterize_json(json_path: str, debug_dir=None):
    """Characterizes a mask file in JSON format"""
    with open(json_path) as f:
        overall_mask = json.load(f)
    image_name = get_image_name(json_path)
    num_roots = len(overall_mask[0][0])
    results = []
    for i in range(num_roots):
        mask = np.array([[cell[i] for cell in row] for row in overall_mask], dtype=np.uint8)
        debug_path = f'./{debug_dir}/{image_name}_{i}.png' if debug_dir else None
        results.append(characterize(image_name, i, mask, debug_path=debug_path))
    return results

def characterize_dir(dir_path='./masks', output_path='./characterization.csv', debug_dir=None):
    """Characterizes mask files from a directory and outputs to a CSV"""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([col.value for col in OutputColumn])
        num_files = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                num_files += 1
                print(f'Progress for {root}: {num_files}/{len(files)}')
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext == '.json':
                    for output_row in characterize_json(filepath, debug_dir):
                        writer.writerow(list(output_row))
                elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    writer.writerow(list(characterize_image(filepath, debug_dir)))


if __name__ == '__main__':
    mask_dir = sys.argv[1] if len(sys.argv) >= 2 else './masks'
    if DEBUG:
        DEBUG_DIR = './debug'
        """Directory to output debugging images to"""
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        characterize_dir(mask_dir, debug_dir=DEBUG_DIR)
    else:
        characterize_dir(mask_dir)
