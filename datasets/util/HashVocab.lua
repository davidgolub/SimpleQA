--[[

A HashVocabulary object. Initialized from a file with one HashVocabulary token per line.
Maps between HashVocabulary tokens and indices. If an UNK token is defined in the
HashVocabulary, returns the index to this token if queried for an out-of-HashVocabulary
token.

--]]

local HashVocab = torch.class('datasets.HashVocab')

function HashVocab:__init(path, add_unk)
  assert(add_unk ~= nil, "Must determine whether to add unknown token or not")
  print("Loading HashVocabulary from path " .. path)

  self.hashed = true
  self.size = 0
  self._index = {}
  self._tokens = {}

  -- Include special start symbol and end symbol
  local file = io.open(path)
  if file == nil then error("Error opening file " .. path .. "\n") end

  -- Gets all tokens from line via hashing and adds
  local function add_line(line)
    local hashed_items = self:hash(line)
    for i = 1, #hashed_items do
      local hashed_item = hashed_items[i]
      self:add(hashed_item)
    end
  end

  local num_lines = 0
  while true do
    local line = file:read()
    num_lines = num_lines + 1
    if line == nil then break end
    add_line(line)
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
      self:add_pad_token()
      self:add_end_token()
      self:add_unk_token()
  end

  print("Loaded all the HashVocabulary from " .. path .. " size is " .. self.size)
end

-- Hashes word into word vector
function HashVocab:hash(w)
  hashed_items = {}
  padded_word = "#" .. w .. "#"
  for i = 1, #padded_word - 2 do
    table.insert(hashed_items, padded_word:sub(i, i + 2))
  end
  return hashed_items
end

function HashVocab:contains(w)
  if not self._index[w] then return false end
  return true
end

function HashVocab:add(w)
  if self._index[w] ~= nil then
    return self._index[w]
  end
  self.size = self.size + 1
  self._tokens[self.size] = w
  self._index[w] = self.size
  return self.size
end

-- returns index/hashed vocab representation
-- if gpu_mode is null then assume that we want IntTensor
function HashVocab:index(word, gpu_mode)
  assert(word ~= nil, "Must specify word to index")
  local tensor = torch.DoubleTensor(self.size):zero()
  local hashed_items = self:hash(word)

  for i = 1, #hashed_items do
    local token = hashed_items[i]
    local index = self._index[token]
    if index == nil then
      if self.unk_index == nil then
        error('Token not in HashVocabulary and no UNK token defined: ' .. w)
      else
        index = self.unk_index
      end
    end
    tensor[index] = tensor[index] + 1
  end
  return tensor
end

function HashVocab:token(i)
  if i < 1 or i > self.size then
    error('Index ' .. i .. ' out of bounds')
  end
  return self._tokens[i]
end

function HashVocab:tokens(indeces)
  local output = {}
  local len = #indeces
  for i = 1, len do
    output[i] = self:token(indeces[i])
  end
  return output
end

-- Converts tokens including unknowns to IntTensor
-- tokens: a table of strings
function HashVocab:map(tokens)
  assert(tokens ~= nil, "Tokens must not be null")
  local len = #tokens
  local output = torch.DoubleTensor(len, self.size)
  for i = 1, len do
    output[i] = self:index(tokens[i])
  end
  return output
end

-- Converts tokens excluding unknowns to int indeces
function HashVocab:map_no_unk(tokens)
  assert(tokens ~= nil, "Tokens must not be null")
   self:map(tokens)
end

function HashVocab:add_pad_token()
  if self.unk_token ~= nil then return end
  self.pad_index = self:add('<pad>')
end

function HashVocab:add_unk_token()
  if self.unk_token ~= nil then return end
  self.unk_index = self:add('<unk>')
end

function HashVocab:add_start_token()
  if self.start_token ~= nil then return end
  self.start_index = self:add('<s>')
end

function HashVocab:add_end_token()
  if self.end_token ~= nil then return end
  self.end_index = self:add('</s>')
end
