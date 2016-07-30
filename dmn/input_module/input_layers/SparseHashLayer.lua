--[[

  Hash layer: Trigram word hashing layer to create word embeddings but in sparse representation
  to save memory: represent words as indices in the word hashing layer and use a lookup table
  plus sum layer

--]]

local SparseHashLayer, parent = torch.class('dmn.SparseHashLayer', 'dmn.InputLayer')

function SparseHashLayer:__init(config)
  parent.__init(self, config)
  
  -- create master embed cell
  self.emb = self:new_emb_table()

  -- create array of children cells
  self.emb_arr = {}

  -- transfer them to gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end

  -- get their parameters
end

function SparseHashLayer:new_emb_table()
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
function SparseHashLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function SparseHashLayer:set_gpu_mode()
  self.emb:cuda()
  for i = 1, #self.emb_arr do
    self.emb_arr[i]:cuda()
  end
  self.gpu_mode = true
end

function SparseHashLayer:set_cpu_mode()
  self.emb:double()
  for i = 1, #self.emb_arr do
    self.emb_arr[i]:double()
  end
  self.gpu_mode = false
end

-- Enable Dropouts
function SparseHashLayer:enable_dropouts()
  enable_sequential_dropouts(self.emb)
  for i = 1, #self.emb_arr do
    enable_sequential_dropouts(self.emb_arr[i])
  end
end

-- Disable Dropouts
function SparseHashLayer:disable_dropouts()
  disable_sequential_dropouts(self.emb)
  for i = 1, #self.emb_arr do
    disable_sequential_dropouts(self.emb_arr[i])
  end
end


-- Does a single forward step of hashing layer, projecting hashed word vectors
-- Into lower dimensional latent semantic space
function SparseHashLayer:forward(hashed_rep)
   assert(hashed_rep ~= nil, "Must specify word tokens")

   -- get number of tokens in the sentence
   local sentence_length = #hashed_rep
   self.word_proj = self.gpu_mode and torch.CudaTensor(sentence_length, self.emb_dim)
                    or torch.DoubleTensor(sentence_length, self.emb_dim)

   for i = 1, sentence_length do 
    local cur_rep = hashed_rep[i]
    if self.emb_arr[i] == nil then 
      print("Creating a new sparsehash word encoder for vector"..i)
      self.emb_arr[i] = self:new_emb_table()
    end
    self.word_proj[i] = self.emb_arr[i]:forward(cur_rep)
   end

   return self.word_proj
end

function SparseHashLayer:backward(hashed_rep, err)
   assert(hashed_rep ~= nil, "Must specify word tokens")
   assert(err ~= nil, "Must specify error with respect to gradient output")

   -- get number of tokens in the sentence
   local sentence_length = #hashed_rep

   self.errors = {}

   for i = sentence_length, 1, -1 do 
    local cur_rep = hashed_rep[i]
    if self.emb_arr[i] == nil then 
      print("Creating a word sentence encoder")
      self.emb_arr[i] = self:new_emb_table()
    end

    self.errors[i] = self.emb_arr[i]:backward(cur_rep, err[i])
   end
   
   return self.errors
 end

function SparseHashLayer:share_params(other, ...)
  assert(other ~= nil, "Must specify other SparseHashLayer to share params with")
  print("Sharing Sparse HashLayer")
  self.emb:share(other.emb, ...)

  local cur_type = self.gpu_mode and torch.CudaTensor or torch.IntTensor

  local input = {cur_type{1, 2}, cur_type{3, 4}}
  local test = self:forward(input)
  local test1 = other:forward(input)
  local diff = test - test1
  assert(torch.sum(diff) == 0, "Parameters aren't shared")
end

-- Returns size of outputs of this combine module
function SparseHashLayer:getOutputSize()
  return self.emb_dim
end

function SparseHashLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function SparseHashLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function SparseHashLayer:getModules() 
  return {self.emb}
end

-- Shares parameters between this embed layer and other layer
function SparseHashLayer:share(other)
end


function SparseHashLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end





