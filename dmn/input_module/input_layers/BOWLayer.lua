--[[

  BOW Layer: Represents a sentence as a sum of bag of words of its embeddings

--]]

local BOWLayer, parent = torch.class('dmn.BOWLayer', 'dmn.InputLayer')

function BOWLayer:__init(config)
  parent.__init(self, config)
  
  -- create master embed cell
  self.emb = self:new_emb_table()

  -- transfer them to gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end

  -- get their parameters
end

function BOWLayer:forget()
end

function BOWLayer:new_emb_table()
  local emb_table = nn.LookupTable(self.vocab_size, self.emb_dim)
  
  -- Copy embedding weights
  if self.emb_vecs ~= nil then
    print("Initializing embeddings from config ")
    emb_table.weight:copy(config.emb_vecs)
  end

  local emb = nn.Sequential()
            :add(emb_table)
            :add(nn.Sum(1))

  if self.dropout then
    print("Adding dropout to embed layer")
    emb:add(nn.Dropout(self.dropout_prob, false))
  end

  -- share parameters if needed
  if self.gpu_mode then 
    print("CUDA BABY")
    emb:cuda()
  end

  if self.emb ~= nil then
    emb:share(self.emb, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return emb
end



-- Returns all of the weights of this module
function BOWLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function BOWLayer:set_gpu_mode()
  self.emb:cuda()
  self.gpu_mode = true
end

function BOWLayer:set_cpu_mode()
  self.emb:double()
  self.gpu_mode = false
end

-- Enable Dropouts
function BOWLayer:enable_dropouts()
  enable_sequential_dropouts(self.emb)
end

-- Disable Dropouts
function BOWLayer:disable_dropouts()
  disable_sequential_dropouts(self.emb)
end


-- Does a single forward step of hashing layer, projecting hashed word vectors
-- Into lower dimensional latent semantic space
function BOWLayer:forward(sentence_rep)
   assert(sentence_rep ~= nil, "Must specify word tokens")

   self.word_proj = self.emb:forward(sentence_rep)

   return self.word_proj
end

function BOWLayer:backward(sentence_rep, err)
   assert(sentence_rep ~= nil, "Must specify word tokens")
   assert(err ~= nil, "Must specify error with respect to gradient output")

   self.errors = self.emb:backward(sentence_rep, err)
   
   return self.errors
 end

function BOWLayer:share_params(other, ...)
  assert(other ~= nil, "Must specify other BOWLayer to share params with")
  print("Sharing BOW HashLayer")
  self.emb:share(other.emb, ...)

  local cur_type = self.gpu_mode and torch.CudaTensor or torch.IntTensor

  local input = cur_type{1, 2}
  local test = self:forward(input)
  local test1 = other:forward(input)
  local diff = test - test1
  assert(torch.sum(diff) == 0, "Parameters aren't shared")
end

-- Returns size of outputs of this combine module
function BOWLayer:getOutputSize()
  return self.emb_dim
end

function BOWLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function BOWLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function BOWLayer:getModules() 
  return {self.emb}
end

-- Shares parameters between this embed layer and other layer
function BOWLayer:share(other, ...)
  self:share_params(other, ...)
end


function BOWLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end

function BOWLayer:print_config()
  printf('%-25s = %s\n', 'gpu mode', self.config.gpu_mode)
  printf('%-25s = %s\n', 'use dropout', self.config.dropout)
  printf('%-25s = %d\n', 'embed dimension', self.config.emb_dim)
  printf('%-25s = %d\n', 'number of classes', self.config.num_classes)
  printf('%-25s = %f\n', 'input dropout probability', self.config.dropout_prob)
end




