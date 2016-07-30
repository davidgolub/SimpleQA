require('..')

local img_path = '../datasets/ImageClassification/cifar10/raw/image_paths.txt'
local tot_img_paths = datasets.read_line_data(img_path)

local mean, std = dmn.image_functions.compute_mean_std_image_list(tot_img_paths, 10000)
print(mean)
print(std)