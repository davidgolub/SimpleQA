datasets = {}
http = require("socket.http")
https = require 'ssl.https'
utf8 = require('utf8')
require('json')
require('lfs')
cjson = require('cjson')

-- Math helper functions
include('../datasets/util/math.lua')

-- For reading word embeddings, image features, and captions
include('../datasets/util/Vocab.lua')
include('../datasets/util/HashVocab.lua')
include('../datasets/util/SparseHashVocab.lua')


-- For reading simple questions data
include('../datasets/util/read_sq_data.lua')

-- For processing the qa data/generating vocab
include('../datasets/util/qa_processing_util.lua')

include('../datasets/Freebase/Freebase_EL_API.lua')

datasets.freebase_api = datasets.Freebase_EL_API()

