softmax = {}

http = require("socket.http")

-- ML APIs
include('AttentionAPI.lua')

-- Entity Linkers
include('EntityLinkerAPI.lua')


-- ml apis
softmax.qa_api = softmax.AttentionAPI{}

-- entity linking api
softmax.entity_linker_api = softmax.EntityLinkerAPI{}


-- profilers
-- softmax.profiler = require('../apis/profilers/ProFi.lua')


--local tmp_path = '/Users/David/Desktop/test.%s'
--local image_url = 'https://static.elie.net/image/blog/2011/07/301-redirect1.png'
--local im_path = softmax.image_api:download(image_url, tmp_path)
--local cur_img = image.load(im_path)

print("Done loading image captioning modules")