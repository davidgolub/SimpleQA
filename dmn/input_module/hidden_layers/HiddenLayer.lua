--[[

  Hidden Layer base class

--]]

local HiddenLayer = torch.class('dmn.HiddenLayer')

function HiddenLayer:__init(config)
   assert(config.mem_dim ~= nil, "Must specify memory dimensions")
   assert(config.input_dim ~= nil, "Must specify input dimensions")
   assert(config.dropout_prob ~= nil, "Must specify dropout probability")
   assert(config.num_layers ~= nil, "Must specify number of layers")
   assert(config.dropout ~= nil, "Must specify dropout")

   self.gpu_mode = config.gpu_mode
   self.input_dim = config.input_dim
   self.proj_dim = config.mem_dim
   self.dropout_prob = config.dropout_prob
   self.dropout = config.dropout
   self.num_layers = config.num_layers

   --print("Hidden layer dropout probability ", self.dropout_prob)
end

-- Returns all of the weights of this module
function HiddenLayer:getWeights()
   error("Get weights not implemented!")
end

-- Returns all the nn modules of this layer as an array
function HiddenLayer:getModules() 
   error("Get modules not implemented!")
end

-- Sets gpu mode
function HiddenLayer:set_gpu_mode()
   error("Set gpu mode not implemented!")
end

function HiddenLayer:set_cpu_mode()
   error("Set cpu mode not implemented!")
end

-- Enable Dropouts
function HiddenLayer:enable_dropouts()
   error("Enable dropouts not implemented!")
end

-- Disable Dropouts
function HiddenLayer:disable_dropouts()
   error("Disable dropouts not implemented!")
end

-- Does a single forward step of hidden layer, which
-- projects inputs into hidden state for lstm. Returns an array
-- Where first state corresponds to cell state, second state
-- corresponds to first hidden state
function HiddenLayer:forward(inputs, gpu_mode)
   assert(inputs ~= nil)
   assert(gpu_mode ~= nil)
   local cuda_type = gpu_mode and 'torch.CudaTensor' or 'torch.DoubleTensor'
   check_type(inputs, cuda_type)
end

-- Does a single backward step of hidden layer
-- Cell errors is an array where first input is error with respect to 
-- cell inputs of lstm, second input is error with respect to hidden inputs
-- of lstm
function HiddenLayer:backward(inputs, cell_errors, gpu_mode)
   assert(inputs ~= nil)
   assert(gpu_mode ~= nil)
   local cuda_type = gpu_mode and 'torch.CudaTensor' or 'torch.DoubleTensor'
   check_type(inputs, cuda_type)
end

-- Returns size of outputs of this hidden module
function HiddenLayer:getOutputSize()
   error("Get output size not implemented!")
end

-- Returns parameters of this model: parameters and gradients
function HiddenLayer:getParameters()
   error("Get parameters not implemented!")
end

-- zeros out the gradients
function HiddenLayer:zeroGradParameters() 
   error("Zero grad parameters not implemented!")
end

function HiddenLayer:normalizeGrads(batch_size)
   error("Normalize gradients not implemented!")
end

