--[[

  Hash layer: Trigram word hashing layer to create word embeddings

--]]

local HashLayer, parent = torch.class('dmn.HashLayer', 'dmn.InputLayer')

function HashLayer:__init(config)
  parent.__init(self, config)
  self.emb_table = nn.Linear(self.vocab_size, self.emb_dim)
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
function HashLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function HashLayer:set_gpu_mode()
  self.emb:cuda()
  self.gpu_mode = true
end

function HashLayer:set_cpu_mode()
  self.emb:double()
  self.gpu_mode = false
end

-- Enable Dropouts
function HashLayer:enable_dropouts()
   enable_sequential_dropouts(self.emb)
end

-- Disable Dropouts
function HashLayer:disable_dropouts()
   disable_sequential_dropouts(self.emb)
end


-- Does a single forward step of hashing layer, projecting hashed word vectors
-- Into lower dimensional latent semantic space
function HashLayer:forward(hashed_rep)
   assert(hashed_rep ~= nil, "Must specify word tokens")
   self.word_proj = self.emb:forward(hashed_rep)
   return self.word_proj
end

function HashLayer:backward(hashed_rep, err)
   assert(word_tokens ~= nil, "Must specify word tokens")
   assert(err ~= nil, "Must specify error with respect to gradient output")
   local emb_err = self.emb:backward(hashed_rep, err)
   return emb_err
end

function HashLayer:share(other, ...)
  share_params(self.emb, other.emb, ...)
end

-- Returns size of outputs of this combine module
function HashLayer:getOutputSize()
  return self.emb_dim
end

function HashLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function HashLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function HashLayer:getModules() 
  return {self.emb}
end

-- Shares parameters between this embed layer and other layer
function HashLayer:share(other)
end


function HashLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end


