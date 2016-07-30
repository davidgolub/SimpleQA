--[[
Tests image loader test. Makes sure grayscale images are loaded correctly
Assumes it's executed from the python directory
]]

require('..')
black_and_white_im_path = 'tests/grayscale.jpg'

local first_img = image.load(black_and_white_im_path)
local second_img = dmn.image_functions.python_load_image(black_and_white_im_path)

assert(first_img:size() == second_img:size(), "First image and second image dim must match")

-- then test super super large images
local large_img_path = '../datasets/Captioning/context/images/Indianapolis_in_1831.png'
large_img = dmn.image_functions.python_load_image(large_img_path)


-- Python code
import PIL.Image as Image 
import numpy

tmp = Image.open('../datasets/Captioning/context/images/Indianapolis_in_1831.png')
arr = numpy.array(tmp.size)
tmp.putdata(arr)

 
