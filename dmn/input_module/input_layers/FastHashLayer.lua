--[[

  Fast Hash layer: Trigram word hashing layer to create word embeddings but in sparse representation
  to save memory: represent words as indices in the word hashing layer and use a lookup table
  plus sum layer

--]]

local FastHashLayer, parent = torch.class('dmn.FastHashLayer', 'dmn.InputLayer')

function FastHashLayer:__init(config)
  parent.__init(self, config)
  
  -- create master embed cell
  self.emb = self:new_emb_table()
  self.join_layer = nn.JoinTable(1)

  -- create array of children cells
  self.emb_arr = {}

  -- transfer them to gpu mode
  if self.gpu_mode then
    self:set_gpu_mode()
  end

  -- get their parameters
end

function FastHashLayer:new_emb_table()
  local emb_table = nn.LookupTable(self.vocab_size, self.emb_dim)
  
  -- Copy embedding weights
  if self.emb_vecs ~= nil then
    dmn.logger:print("Initializing embeddings from config ")
    emb_table.weight:copy(config.emb_vecs)
  end

  local emb = nn.Sequential()
            :add(emb_table)

  if self.dropout then
    dmn.logger:print("Adding dropout to embed layer")
    emb:add(nn.Dropout(self.dropout_prob, false))
  end

  -- share parameters if needed
  if self.gpu_mode then 
    dmn.logger:print("CUDA BABY")
    emb:cuda()
  end

  if self.emb ~= nil then
    emb:share(self.emb, 'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return emb
end



-- Returns all of the weights of this module
function FastHashLayer:getWeights()
  return self.params
end

-- Sets gpu mode
function FastHashLayer:set_gpu_mode()
  self.emb:cuda()
  self.gpu_mode = true
end

function FastHashLayer:set_cpu_mode()
  self.emb:double()
  self.gpu_mode = false
end

-- Enable Dropouts
function FastHashLayer:enable_dropouts()
  enable_sequential_dropouts(self.emb)
end

-- Disable Dropouts
function FastHashLayer:disable_dropouts()
  disable_sequential_dropouts(self.emb)
end


-- Does a single forward step of hashing layer, projecting hashed word vectors
-- Into lower dimensional latent semantic space
function FastHashLayer:forward(hashed_rep)
   assert(hashed_rep ~= nil, "Must specify word tokens")

   local char_indeces, word_lengths = unpack(hashed_rep)

   -- get number of tokens in the sentence
   local sentence_length = word_lengths:size(1)
   self.word_proj = self.gpu_mode and torch.CudaTensor(sentence_length, self.emb_dim)
                    or torch.DoubleTensor(sentence_length, self.emb_dim)

   self.joined_vectors = char_indeces
   -- first get character/hashed embeddings
   self.char_embeddings = self.emb:forward(self.joined_vectors)

   local cur_index = 1

   for i = 1, sentence_length do
    local cur_word_length = word_lengths[i]
    local cur_embeddings = self.char_embeddings[{{cur_index, cur_index + cur_word_length - 1}}]

    -- sum up embeddings == bag of words representation
    self.word_proj[i] =  torch.squeeze(cur_embeddings:sum(1))
    cur_index = cur_index + cur_word_length
   end

   return self.word_proj
end

function FastHashLayer:backward(hashed_rep, err)
   assert(hashed_rep ~= nil, "Must specify word tokens")
   assert(err ~= nil, "Must specify error with respect to gradient output")


   local char_indeces, word_lengths = unpack(hashed_rep)

   -- get number of tokens in the sentence
   local sentence_length = word_lengths:size(1)
   
   -- first join the tables into one table
   self.errors = self.char_embeddings:new()
   local cur_index = 1
   for i = 1, sentence_length do 
    -- since we sum up our gradient of input is just gradient of output
    local cur_word_length = word_lengths[i]

    for j = cur_index, cur_index + cur_word_length - 1 do 
      self.errors[j] = err[i]
    end

    -- update word index
    cur_index = cur_index + cur_word_length
   end
   
   local errs = self.emb:backward(self.joined_vectors, self.errors)
   return errs
 end

function FastHashLayer:share_params(other, ...)
  assert(other ~= nil, "Must specify other FastHashLayer to share params with")
  dmn.logger:print("Sharing Sparse HashLayer")
  self.emb:share(other.emb, ...)

  local cur_type = self.gpu_mode and torch.CudaTensor or torch.IntTensor

  local input = {cur_type{1, 2}, cur_type{1}}
  local test = self:forward(input)
  local test1 = other:forward(input)
  local diff = test - test1
  assert(torch.sum(diff) == 0, "Parameters aren't shared")
end

-- Returns size of outputs of this combine module
function FastHashLayer:getOutputSize()
  return self.emb_dim
end

function FastHashLayer:getParameters()
  return self.params, self.grad_params
end

-- zeros out the gradients
function FastHashLayer:zeroGradParameters() 
  self.emb:zeroGradParameters()
end

function FastHashLayer:getModules() 
  return {self.emb}
end

-- Shares parameters between this embed layer and other layer
function FastHashLayer:share(other)
end


function FastHashLayer:normalizeGrads(batch_size)
  self.emb.gradWeight:div(batch_size)
end





