--[[

  Embed layer: Simple word embedding layer for input into lstm

--]]

local EmbedLayer, parent = torch.class('dmn.EmbedLayer', 'dmn.InputLayer')

function EmbedLayer:__init(config)
  parent.__init(self, config)
  self.config = dmn.functions.deepcopy(config)
  self.gpu_mode = self.config.gpu_mode
  self.emb_table = nn.LookupTable(self.vocab_size, self.emb_dim)
    -- Copy embedding weights
  if config.emb_vecs ~= nil then
    print("Initializing embeddings from config ")
    self.emb_table.weight:copy(config.emb_vecs)
  end

  self.emb = nn.Sequential()
            :add(self.emb_table)


  if self.dropout then
    print("Adding dropout to embed layer")
    self.emb:add(nn.Dropout(self.dropout_prob, false))
  end

  if self.gpu_mode then
    self:set_gpu_mode()
  end
end

-- Returns all of the weights of this module
function EmbedLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function EmbedLayer:set_gpu_mode()
  dmn.logger:print("Setting GPU mode on embed layer")
  self.emb:cuda()
  self.emb_table:cuda()
  self.gpu_mode = true
  print("GPU MODE FOR EMBED LAYER IS " .. tostring(self.gpu_mode))
end

function EmbedLayer:set_cpu_mode()
  dmn.logger:print("Setting CPU mode on embed layer")
  self.emb:double()
  self.emb_table:double()
  self.gpu_mode = false
end

-- Enable Dropouts
function EmbedLayer:enable_dropouts()
   enable_sequential_dropouts(self.emb)
end

-- Disable Dropouts
function EmbedLayer:disable_dropouts()
   disable_sequential_dropouts(self.emb)
end

-- Does a single forward step of concat layer, concatenating
-- Input 
function EmbedLayer:forward(word_indeces)
   assert(word_indeces ~= nil, "Must specify word indeces to forward")
   parent:forward(word_indeces, self.gpu_mode)
   self.word_proj = self.emb:forward(word_indeces)
   return self.word_proj
end

function EmbedLayer:backward(word_indices, err)

   parent:backward(word_indices, err, self.gpu_mode)
   local emb_err = self.emb:backward(word_indices, err)
   return emb_err
end

function EmbedLayer:share_params(other, ...)
  assert(other ~= nil, "Must specify other layer to share params with")

  self.emb_table:share(other.emb_table, ...)
 
  -- sanity check: make sure you get same outputs on forwarding
  local input = self.gpu_mode and torch.CudaTensor{1, 2} or torch.IntTensor{1, 2}
  local test = self:forward(input)
  local test1 = other:forward(input)
  local diff = test - test1
  assert(torch.sum(diff) == 0, "Parameters aren't shared")
end
-- Returns size of outputs of this combine module
function EmbedLayer:getOutputSize()
  return self.emb_dim
end

function EmbedLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function EmbedLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function EmbedLayer:getModules() 
  return {self.emb}
end

function EmbedLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

function EmbedLayer:print_config()
  printf('%-25s = %s\n', 'gpu mode', self.config.gpu_mode)
  printf('%-25s = %s\n', 'use dropout', self.config.dropout)
  printf('%-25s = %d\n', 'embed dimension', self.config.emb_dim)
  printf('%-25s = %d\n', 'number of classes', self.config.num_classes)
  printf('%-25s = %f\n', 'input dropout probability', self.config.dropout_prob)
end



