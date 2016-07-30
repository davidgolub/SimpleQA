--[[

A vocabulary object. Initialized from a file with one vocabulary token per line.
Maps between vocabulary tokens and indices. If an UNK token is defined in the
vocabulary, returns the index to this token if queried for an out-of-vocabulary
token.

--]]

local Vocab = torch.class('datasets.Vocab')

function Vocab:__init(path, add_unk)
  assert(add_unk ~= nil)
  print("Loading vocabulary from path " .. path)

  self.hashed = false
  self.size = 0
  self._index = {}
  self._tokens = {}

  -- Include special start symbol and end symbol
  local file = io.open(path, 'r')
  if file == nil then error("Error opening file " .. path .. "\n") end

  while true do
    local line = file:read()
    if line == nil then break end
    self.size = self.size + 1
    self._tokens[self.size] = line
    self._index[line] = self.size
  end
  file:close()

  local unks = {'<unk>', '<UNK>', 'UUUNKKK'}
  for _, tok in pairs(unks) do
    self.unk_index = self.unk_index or self._index[tok]
    if self.unk_index ~= nil then
      self.unk_token = tok
      break
    end
  end

  local starts = {'<s>', '<S>'}
  for _, tok in pairs(starts) do
    self.start_index = self.start_index or self._index[tok]
    if self.start_index ~= nil then
      self.start_token = tok
      break
    end
  end

  local ends = {'</s>', '</S>'}
  for _, tok in pairs(ends) do
    self.end_index = self.end_index or self._index[tok]
    if self.end_index ~= nil then
      self.end_token = tok
      break
    end
  end

  if add_unk then
      self:add_start_token()
      self:add_end_token()
      self:add_pad_token()
      self:add_unk_token()
  end

  print("Loaded all the vocabulary from " .. path .. " size is " .. self.size)
end

function Vocab:contains(w)
  if not self._index[w] then return false end
  return true
end

function Vocab:add(w)
  if self._index[w] ~= nil then
    return self._index[w]
  end
  self.size = self.size + 1
  self._tokens[self.size] = w
  self._index[w] = self.size
  return self.size
end

function Vocab:index(w)
  local index = self._index[w]
  if index == nil then
    if self.unk_index == nil then
      error('Token not in vocabulary and no UNK token defined: ' .. w)
    end
    return self.unk_index
  end
  return index
end

function Vocab:token(i)
  if i < 1 or i > self.size then
    error('Index ' .. i .. ' out of bounds')
  end
  return self._tokens[i]
end

function Vocab:tokens(indeces)
  assert(indeces ~= nil, "Must specify indeces to predict on")
  local output = {}

  -- get length: first case is when it's a table, second when an IntTensor
  local len = (torch.typename(indeces) == 'torch.IntTensor' or 
               torch.typename(indeces) == 'torch.CudaTensor')
               and indeces:size(1) or #indeces
  for i = 1, len do
    output[i] = self:token(indeces[i])
  end
  return output
end

-- Converts tokens including unknowns to IntTensor
function Vocab:map(tokens, gpu_mode)
  assert(tokens ~= nil, "Must specify tokens to map")
  assert(gpu_mode ~= nil, "Must specify gpu mode to use for mapping tokens")
  --local use_gpu = (gpu_mode == nil) and false or gpu_mode

  --local len = #tokens
  --local output = use_gpu and torch.CudaTensor(len) or torch.IntTensor(len)
  --for i = 1, len do
  --  output[i] = self:index(tokens[i])
  --end
  --return output
  return self:map_no_unk(tokens, gpu_mode)
end

-- Converts tokens excluding unknowns to int indeces
function Vocab:map_no_unk(tokens, gpu_mode)
  assert(tokens ~= nil, "Must specify tokens to map")
  assert(gpu_mode ~= nil, "Must specify gpu mode")
  
  local use_gpu = gpu_mode
  
  local len = #tokens
  local has_unk = false
  for i = 1, #tokens do
    local index = self:index(tokens[i])
    if #tokens > 30 then 
      --print(tokens[i])
    end
    if index == self.unk_index then
      --print(tokens[i])
      --print(index)
      len = len - 1
    end
  end
  
  if has_unk then num_unk_sentences = num_unk_sentences + 1 end
  local output = use_gpu and torch.CudaTensor(len) or torch.IntTensor(len)
  local curr_index = 1
  for i = 1, #tokens do
    local index = self:index(tokens[i])
    if index ~= self.unk_index then
      output[curr_index] = index
      curr_index = curr_index + 1
    end
  end
  return output
end

function Vocab:add_pad_token()
  if self.unk_token ~= nil then return end
  self.pad_index = self:add('<pad>')
end

function Vocab:add_unk_token()
  if self.unk_token ~= nil then return end
  self.unk_index = self:add('<unk>')
end

function Vocab:add_start_token()
  if self.start_token ~= nil then return end
  self.start_index = self:add('<s>')
end

function Vocab:add_end_token()
  if self.end_token ~= nil then return end
  self.end_index = self:add('</s>')
end
