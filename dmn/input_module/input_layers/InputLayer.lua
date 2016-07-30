--[[

  Hidden Layer base class

--]]

local InputLayer = torch.class('dmn.InputLayer')

function InputLayer:__init(config)
  assert(config.emb_dim ~= nil, "Must specify embed dimensions")
  assert(config.num_classes ~= nil, "Must specify number of classes")
  assert(config.dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.dropout ~= nil, "Must specify whether to use dropout or not")

  self.config = dmn.functions.deepcopy(config)
  self.gpu_mode = config.gpu_mode
  self.emb_dim = config.emb_dim
  self.emb_vecs = config.emb_vecs
  self.dropout = config.dropout
  self.vocab_size = config.num_classes
  self.dropout_prob = config.dropout_prob
  
  if config.emb_vecs ~= nil then
    self.vocab_size = config.emb_vecs:size(1)
  end
end

-- Returns all of the weights of this module
function InputLayer:getWeights()
	error("Get weights not implemented!")
end

-- Sets gpu mode
function InputLayer:set_gpu_mode()
	error("Set gpu mode not implemented!")
end

function InputLayer:set_cpu_mode()
	error("Set cpu mode not implemented!")
end

-- Enable Dropouts
function InputLayer:enable_dropouts()
	error("Enable dropouts not implemented!")
end

-- Disable Dropouts
function InputLayer:disable_dropouts()
	error("Disable dropouts not implemented!")
end


-- Does a single forward step of concat layer, concatenating
-- Input 
function InputLayer:forward(word_indices, gpu_mode)
   assert(word_indices ~= nil) 
   --print("Gpu mode for forward step parent", gpu_mode)
  local word_type = gpu_mode and 'torch.CudaTensor' or 'torch.IntTensor'
  check_type(word_indices, word_type)
end

function InputLayer:backward(word_indices, err, gpu_mode)
  assert(word_indices ~= nil, "Word indices are null!")
  assert(err ~= nil, "Error is null!")

  local word_type = gpu_mode and 'torch.CudaTensor' or 'torch.IntTensor'
	check_type(word_indices, word_type)
end

-- Returns size of outputs of this combine module
function InputLayer:getOutputSize()
	error("Get output size not implemented!")
end

function InputLayer:getParameters()
	error("Get parameters not implemented!")
end

-- zeros out the gradients
function InputLayer:zeroGradParameters() 
	error("Zero grad parameters not implemented!")
end

function InputLayer:getModules() 
  error("Get modules not implemented!")
end

function InputLayer:share(other, ...) 
  error("Get modules not implemented!")
end

function InputLayer:normalizeGrads(batch_size)
	error("Normalize grads not implemented!")
end




