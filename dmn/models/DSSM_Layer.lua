--[[
 A Deep Semantic Similarity Model over text that has temporal convolutions and max pooling layers,
 returns a latent semantic representation of the text
--]]

local DSSM_Layer, parent = torch.class('dmn.DSSM_Layer', 'nn.Module')


function DSSM_Layer:__init(config)
   parent.__init(self)

   assert(config.gpu_mode ~= nil, "Must specify whether to use gpu mode or not")
   assert(config.in_dim ~= nil, "must specify input dimensions")
   assert(config.hidden_dim ~= nil, "must specify hidden input dimension")
   assert(config.out_dim ~= nil, "must specify output dimensions")
   assert(config.in_stride ~= nil, "must specify stride for convolution")
   assert(config.in_kernel_width ~= nil, "must specify kernel width")
   assert(config.hidden_stride ~= nil, "must specify output stride for convolution")
   assert(config.hidden_kernel_width ~= nil, "must specify output kernel width for convolution")
   assert(config.out_stride ~= nil, "must specify output stride for convolution")
   assert(config.out_kernel_width ~= nil, "must specify output kernel width for convolution")
   assert(config.dssm_type ~= nil, "Must specify dssm type")

   -- configuration parameters
   self.config = dmn.functions.deepcopy(config)
   self.gpu_mode = config.gpu_mode
   self.in_dim = config.in_dim
   self.out_dim = config.out_dim
   self.hidden_dim = config.hidden_dim
   self.in_stride = config.in_stride
   self.in_kernel_width = config.in_kernel_width
   self.hidden_stride = config.hidden_stride
   self.hidden_kernel_width = config.hidden_kernel_width
   self.out_stride = config.out_stride
   self.out_kernel_width = config.out_kernel_width
   self.out_dim = config.out_dim

   self.dssm_type = config.dssm_type

   if self.dssm_type == 'hidden_network' then 
      self.master_network = self:new_hidden_network()
   elseif self.dssm_type == 'shallow_network' then 
      self.master_network = self:new_shallow_network()
   else
      error("Invalid network given " .. self.dssm_type)
   end
end

-- network with hidden layer
function DSSM_Layer:new_hidden_network()
   dmn.logger:print("Creating new hidden network for dssm")
   local network = nn.Sequential()
      :add(nn.TemporalConvolution(self.in_dim, self.hidden_dim, 
        self.in_kernel_width, self.in_stride))
      :add(nn.Linear(self.hidden_dim, self.hidden_dim))   
      :add(nn.Tanh())
      :add(nn.TemporalConvolution(self.hidden_dim, self.out_dim, 
      self.hidden_kernel_width, self.hidden_stride))
      :add(nn.Linear(self.out_dim, self.out_dim))   
      --:add(nn.Tanh())
      :add(nn.Transpose({2, 1}))
      :add(nn.Max(2)) --note: max() is not supported on Cuda devices
      :add(nn.Linear(self.out_dim, self.out_dim))
      :add(nn.Tanh()) -- Projection layer
  if self.gpu_mode then 
    dmn.logger:print("Setting DSSM_Layer's network to cuda ")
    network:cuda()
  end
  return network
end

-- network with hidden layer
function DSSM_Layer:new_temporal_hidden_network()
   dmn.logger:print("Creating new temporal network for dssm")
   local pooling_size = 3
   local network = nn.Sequential()
      :add(nn.TemporalConvolution(self.in_dim, self.hidden_dim, 
        self.in_kernel_width, self.in_stride))
      :add(nn.TemporalMaxPooling(pooling_size))
      :add(nn.TemporalConvolution(self.hidden_dim, self.out_dim, 
      self.hidden_kernel_width, self.hidden_stride))
      :add(nn.Tanh())
      :add(nn.Transpose({3, 2}))
      :add(nn.Max(3)) --note: max() is not supported on Cuda devices
      :add(nn.Linear(self.out_dim, self.out_dim))
      :add(nn.Tanh()) -- Projection layer
  if self.gpu_mode then 
    dmn.logger:print("Setting DSSM_Layer's network to cuda ")
    network:cuda()
  end
  return network
end

function DSSM_Layer:new_shallow_network()
   dmn.logger:print("Creating regular normal layer")
   local network = nn.Sequential()
      :add(nn.TemporalConvolution(self.in_dim, self.out_dim, 
        self.in_kernel_width, self.in_stride))
      :add(nn.Transpose({2, 1}))
      :add(nn.Max(2)) --note: max() is not supported on Cuda devices so must use transpose
      :add(nn.Linear(self.out_dim, self.out_dim)) -- Projection layer
      :add(nn.Tanh())

  if self.gpu_mode then 
    dmn.logger:print("Setting DSSM_Layer's network to cuda ")
    network:cuda()
  end

  return network
end

function DSSM_Layer:new_network_sequential()
   self.tmp = nn.TemporalConvolution(self.in_dim, self.hidden_dim, 
        self.in_kernel_width, self.in_stride)
   local network = nn.Sequential()
      :add(nn.TemporalConvolution(self.in_dim, self.hidden_dim, 
        self.in_kernel_width, self.in_stride))
   if self.gpu_mode then 
     network:cuda()
     self.tmp:cuda()
   end
   return network
end

-- enables dropouts
function DSSM_Layer:enable_dropouts()
end

-- disables dropouts
function DSSM_Layer:disable_dropouts()
end

-- Set all of the network parameters to gpu mode
function DSSM_Layer:set_gpu_mode()
  self.gpu_mode = true
  self.master_network:cuda()
end

-- Transfers network to cpu mode
function DSSM_Layer:set_cpu_mode()
  self.gpu_mode = false
  self.master_network:double()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- returns output of dssm
function DSSM_Layer:forward(inputs)
  assert(inputs ~= nil, "Inputs must not be null")
  check_valid_gpu_inputs(inputs, self.gpu_mode)

  -- make sure inputs are of the right type
  -- must make a copy as calling forward twice changes the inputs for some reason
  local outputs = self.master_network:forward(inputs)

  --local tmp_out = self.tmp:forward(outputs)
  return outputs
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: Derivative with respect to DSSM network
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function DSSM_Layer:backward(inputs, grad_outputs)
  assert(inputs ~= nil, "Inputs must not be null")
  assert(grad_outputs ~= nil, "Grad outputs must not be null")

  local inputGrads = self.master_network:backward(inputs, grad_outputs)
  return inputGrads
end

-- Shares the parameters of this DSSM model with another network
function DSSM_Layer:share(dssm, ...)
  assert(dssm ~= nil, "must specify dssm to share with")
  dmn.logger:print("Sharing DSSM layer")
  if self.in_dim ~= dssm.in_dim then error("DSSM input dimension mismatch") end
  if self.out_dim ~= dssm.out_dim then error("DSSM output dimension mismatch") end
  self.master_network:share(dssm.master_network, ...)

  dmn.logger:print("Sanity checking that dssm sharing worked")
 
  if self.sanity_check then 
       -- sanity check: make sure forwarding same input gives same output
    local res1 = torch.rand(50, self.in_dim)

    -- for gpu mode
    if self.gpu_mode then 
      res1 = res1:cuda()
    end

    local test1 = self:forward(res1):clone()
    local test2 = dssm:forward(res1):clone()

    local testres = test1 - test2
    assert(torch.sum(testres) == 0, "DSSM sharing failed, result doesn't sum to 0")
  end
end

function DSSM_Layer:zeroGradParameters()
  self.master_network:zeroGradParameters()
end

function DSSM_Layer:getModules()
  return {self.master_network}
end

function DSSM_Layer:parameters()
  return self.master_network:parameters()
end

-- Clear saved gradients
function DSSM_Layer:forget()
  --print("DSSM Layer doesn't have gradients")
end

function DSSM_Layer:print_config()
  printf('%-25s = %d\n', 'input dimension', self.config.in_dim)
  printf('%-25s = %d\n', 'hidden dimension', self.config.hidden_dim)
  printf('%-25s = %d\n', 'output dimension', self.config.out_dim)
  printf('%-25s = %d\n', 'input stride', self.config.in_stride)
  printf('%-25s = %d\n', 'hidden stride', self.config.hidden_stride)
  printf('%-25s = %d\n', 'output stride', self.config.out_stride)
  printf('%-25s = %d\n', 'input kernel size', self.config.in_kernel_width)
  printf('%-25s = %d\n', 'hidden kernel size', self.config.hidden_kernel_width)
  printf('%-25s = %d\n', 'output kernel width', self.config.out_kernel_width)
end




