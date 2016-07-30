--[[

A SparseHashVocabulary object. Initialized from a file with one SparseHashVocabulary token per line.
Maps between SparseHashVocabulary tokens and indices. If an UNK token is defined in the
HashVocabulary, returns the index to this token if queried for an out-of-HashVocabulary
token.

--]]

local SparseHashVocab = torch.class('datasets.SparseHashVocab')

function SparseHashVocab:__init(path, add_unk)
  assert(add_unk ~= nil, "Must determine whether to add unknown token or not")
  print("Loading SparseHashVocabulary from path " .. path)

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
      self:add_end_token()
      self:add_unk_token()
  end

  print("Loaded all the SparseHashVocabulary from " .. path .. " size is " .. self.size)
end

-- Hashes word into word vector
function SparseHashVocab:hash(w)
  hashed_items = {}
  padded_word = "#" .. w .. "#"
  for i = 1, #padded_word - 2 do
    table.insert(hashed_items, padded_word:sub(i, i + 2))
  end
  return hashed_items
end

function SparseHashVocab:contains(w)
  if not self._index[w] then return false end
  return true
end

function SparseHashVocab:add(w)
  if self._index[w] ~= nil then
    return self._index[w]
  end
  self.size = self.size + 1
  self._tokens[self.size] = w
  self._index[w] = self.size
  return self.size
end

-- returns index/hashed vocab representation
function SparseHashVocab:index(token, gpu_mode)
  assert(token ~= nil, "Must specify token SparseHashVocab needs to index")
  assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")

  local index = self._index[token]
  if index == nil then
    if self.unk_index == nil then
      error('Token not in SparseHashVocabulary and no UNK token defined: ' .. w)
    else
      index = self.unk_index
    end
  end
 return index
end

function SparseHashVocab:token(i)
  if i < 1 or i > self.size then
    error('Index ' .. i .. ' out of bounds')
  end
  return self._tokens[i]
end

function SparseHashVocab:tokens(indeces)
  local output = {}
  local len = #indeces
  for i = 1, len do
    output[i] = self:token(indeces[i])
  end
  return output
end

-- Converts tokens including unknowns to IntTensor if gpu_mode is null or false, CudaTensor if true
-- tokens: a table of strings
-- returns: IntTensor with indeces, and IntTensor with word lengths
function SparseHashVocab:map(tokens, gpu_mode)
  assert(tokens ~= nil, "Tokens must not be null")
  local cur_type = gpu_mode and torch.CudaTensor or torch.IntTensor
  local len = #tokens

  local num_chars = 0
  local word_lengths = cur_type(len)

  -- compute number of characters
  for i = 1, #tokens do
    num_chars = num_chars + #tokens[i]
    word_lengths[i] = #self:hash(tokens[i])
  end 
  
  local chars = cur_type(num_chars)

  local cur_index = 1
  for i = 1, #tokens do 
    local word = tokens[i]
    local hashed_items = self:hash(word)
    for j = 1, #hashed_items do
      local cur_char = hashed_items[j]
      local index = self:index(cur_char, gpu_mode)
      chars[cur_index] = index
      cur_index = cur_index + 1
    end
  end

  assert(chars:size(1) == word_lengths:sum(), "Number of chars and sum of word lengths must match up")

  local res = {chars, word_lengths}
  return res
end

-- Converts tokens excluding unknowns to int indeces
function SparseHashVocab:map_no_unk(tokens)
  assert(tokens ~= nil, "Tokens must not be null")
   self:map(tokens)
end

function SparseHashVocab:add_unk_token()
  if self.unk_token ~= nil then return end
  self.unk_index = self:add('<unk>')
end

function SparseHashVocab:add_start_token()
  if self.start_token ~= nil then return end
  self.start_index = self:add('<s>')
end

function SparseHashVocab:add_end_token()
  if self.end_token ~= nil then return end
  self.end_index = self:add('</s>')
end
