import cv2
import os

### directory names for origin and destination
origin_dir     = '/c_negatives'
destin_dir     = '/g_negatives'

### create properly-named directory
if not os.path.exists(destin_dir):
    os.makedirs(destin_dir)

### turn directory names into absolute paths
origin_dir_abs = os.path.abspath('.' + origin_dir)
destin_dir_abs = os.path.abspath('.' + destin_dir)

### get absolute paths and file names for all colored images
walks = os.walk(origin_dir_abs)
_, _, filenames = walks.next()
img_paths_abs = [(origin_dir_abs + '/' + filename) for filename in filenames]
var = zip(img_paths_abs, filenames) # zip file names and paths for easy mapping

# convert all images to gray_scale format
for (img_path_abs, filename) in var:
    gray_scale = cv2.imread(img_path_abs, 0)
    cv2.imwrite(destin_dir_abs + '/' + filename + '.png', gray_scale)
