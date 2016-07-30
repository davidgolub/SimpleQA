require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('io')
require('json')
require('gnuplot')

package.path = package.path .. ';/Users/david/.luarocks/share/lua/5.1/?.lua;/Users/david/.luarocks/share/lua/5.1/?/init.lua;/Users/david/torch/install/share/lua/5.1/?.lua;/Users/david/torch/install/share/lua/5.1/?/init.lua;./?.lua;/Users/david/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
.. ";/Users/David/Desktop/nlp/deeplearning/softmax/?/init.lua"
.. ";/Users/David/Desktop/softmax/?/init.lua"
.. ";/home/ubuntu/softmax/?/init.lua;"
.. "/softmax/?/init.lua"
.. ";/home/david/Desktop/softmax/?/init.lua;"

-- image processing and async libraries
lua_json = include('../opensource/json.lua')
async = require('async')
require('image')
gm = require('graphicsmagick')

print("Loading all DMN models")
package.path = package.path .. ';../dmn/?'

local num_threads = 6
torch.setnumthreads(num_threads)
-- for dataset processing utils
dmn = {}

dmn.dummy_path = 'data/QA/vocab.txt'
dmn.models_dir = 'trained_models/'
dmn.predictions_dir = 'predictions/'

-- Utility functions for networks (Composite patterns)
include('util/functions.lua')

-- Utility constants
include('util/constants.lua')

-- Utility IO functions (loading images etc)
include('util/io_functions.lua')

-- Utility math functions (calculating mean, etc)
include('util/math_functions.lua')

-- Utility evaluate functions (for lua)
include('util/eval_functions.lua')

-- Logging functions for networks (Logging on local/cloud)
include('util/logger.lua')
include('util/print_logger.lua')

dmn.logger = dmn.Logger()
dmn.logger:add_logger(dmn.PrintLogger())

-- For squeeze unit
include('models/nn_units/Squeeze.lua')

-- For optimizer
include('models/nn_units/Optim.lua')


-- For attention units
include('models/nn_units/CRowAddTable.lua')
include('models/nn_units/SmoothCosineSimilarity.lua')
include('models/nn_units/JoinTable.lua')
include('models/nn_units/PaddedJoinTable.lua')
include('models/nn_units/SpatialCrossLRN.lua')
include('models/nn_units/Linear.lua')

-- change linear to point to correct table
--nn.Linear = dmn.Linear

-- For all the rnn units
include('models/rnn_units/units.lua')

-- Utility functions for lstm units
include('models/RNN_Utils.lua')

-- Recurrent models
include('models/Attention_LSTM_Decoder.lua')
include('models/LSTM_Decoder.lua')

-- Deep semantic similarity network
include('models/DSSM_Layer.lua')

-- Input models
include('input_module/input_layers/InputLayer.lua')
include('input_module/input_layers/BOWLayer.lua')
include('input_module/input_layers/EmbedLayer.lua')
include('input_module/input_layers/HashLayer.lua')
include('input_module/input_layers/SparseHashLayer.lua')
include('input_module/input_layers/FastHashLayer.lua')

-- Hidden models
include('input_module/hidden_layers/HiddenLayer.lua') 
include('input_module/hidden_layers/HiddenDummyLayer.lua')
include('input_module/hidden_layers/HiddenIdentityLayer.lua')
include('input_module/hidden_layers/HiddenProjLayer.lua')
include('input_module/hidden_layers/HiddenGRUProjLayer.lua')

-- Answer reranking module
include('answer_module/AnswerRerankModule.lua')

-- Semantic memory module
include('semantic_memory_module/WordEmbedModule.lua')

-- Attention Network with DSSMs
include('dmn_network/Attention_Network.lua')

printf = utils.printf

-- For data
require('../datasets')

-- For trainers
require('../apis')

-- Check python if it exists
if dmn.functions.module_exists('fb.python') then 
  python = {}

  python.py = require('fb.python')
  python.np = python.py.import("numpy")
  python.pil_image = python.py.import("PIL.Image")

  -- To load any image
  python.py.exec([=[
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
]=])

end

-- share parameters of nngraph gModule instances
function share_params(cell, src, ...)
  for i = 1, #cell.forwardnodes do
    local node = cell.forwardnodes[i]
    if node.data.module then
    	--print(node.data.module)
      node.data.module:share(src.forwardnodes[i].data.module, ...)
    end
  end
end

function header(s)
  dmn.logger:print(string.rep('-', 80))
  dmn.logger:print(s)
  dmn.logger:print(string.rep('-', 80))
end

print("Done loading modules for dynamic memory network")

