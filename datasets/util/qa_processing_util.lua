--[[

  Functions for preprocessing question, input, answer data.
  This involves tokenizing the data, generating vocabulary for inputs/outputs/questions,
  etc.

--]]

-- creates image features for further use
-- saves image paths to image_dir..image_paths
function datasets.download_images(image_dir, image_path_file)
  assert(image_dir ~= nil, "Must specify images to download for future use")
  assert(image_path_file ~= nil, "Must specify image path file to read image features")

  local image_paths = datasets.read_line_data(image_path_file)
  local local_image_paths = {}
  for i = 1, #image_paths do
    local cur_path = image_paths[i]

    -- save path depends on type of image png, etc.
    local save_path = image_dir .. i .. ".%s"
    local act_save_path = softmax.image_api:download(cur_path, save_path)
    table.insert(local_image_paths, act_save_path)
  end

  datasets.save_line_data(local_image_paths, image_dir .. "image_paths.txt")
end

-- generates negative samples from positive sample set and number of samples.
-- returns an array of negative samples, one for each of items_to_avoid. They are INDECES to original array
function datasets.generate_negative_samples(all_samples, items_to_avoid, num_samples)
  assert(all_samples ~= nil, "Must specify all samples to generate data from")
  assert(items_to_avoid ~= nil, "Must specify items to avoid")
  assert(num_samples ~= nil, "Must specify number of samples to generate")

  local sample_table = {}
  for i = 1, #items_to_avoid do
    if i % 1000 == 0 then 
      dmn.logger:print("Generating negative samples on index " .. i .. " of " .. #items_to_avoid)
    end
    if i % 100000 == 0 then 
      dmn.logger:print("Collecting garbage")
      dmn.logger:print(collectgarbage("count")*1024)
      collectgarbage()
    end

    local curr_item_to_avoid = items_to_avoid[i]
    local negative_samples = datasets.generate_negative_sample(all_samples, 
                              curr_item_to_avoid,
                              num_samples)
    sample_table[i] = negative_samples
  end
  return sample_table
end

--Returns a list of negative samples. Samples are an array of INDECES to the all_samples data structure
function datasets.generate_negative_sample(all_samples, item_to_avoid, num_samples)
  assert(all_samples ~= nil, "Must specify all samples to generate data from")
  assert(item_to_avoid ~= nil, "Must specify item to avoid")
  assert(num_samples ~= nil, "Must specify number of samples to generate")

  local negative_samples = {}
  local num_generated = 0
  
  while num_generated < num_samples do 
    local num_items = #all_samples
    local index = torch.random() % num_items + 1
    local sample = all_samples[index]
    if not dmn.math_functions.equals(sample, item_to_avoid) then
      table.insert(negative_samples, index)
      num_generated = num_generated + 1
    end
  end
  return negative_samples
end

-- Saves the entire dataset to the specified directory
-- Has questions one per line in basedir/questions.txt
-- Has inputs tab separated per question then per line in basedir/inputs.txt
-- Has answers one per line in basedir/answers.txt
function datasets.save_qa_dataset(dataset, threshold, base_dir)
  assert(dataset ~= nil, "Must specify the dataset to save")
  assert(base_dir ~= nil, "Must specify the directory to save it to")
  assert(threshold ~= nil, "Must specify threshold to use for directory")
  assert(#dataset.questions == #dataset.inputs, "Number of questions must match num inputs")
  assert(#dataset.questions == #dataset.answers, "Number of questions must match num ans")
  
  if lfs.attributes(base_dir) == nil then
  	print("Directory not found, making new directory at " .. base_dir)
  	lfs.mkdir(base_dir)
  end
  -- write vocabulary items for dataset
  datasets.generate_and_save_vocab(dataset, threshold, base_dir)

  local question_path = base_dir .. "questions.txt"
  local input_path = base_dir .. "inputs.txt"
  local answer_path = base_dir .. "outputs.txt"
   
  datasets.save_line_data(dataset.questions, question_path)
  datasets.save_tsv_data(dataset.inputs, input_path)
  datasets.save_line_data(dataset.answers, answer_path)
end

function datasets.read_line_data(input_path)
  assert(input_path ~= nil, "Must specify input path to read line data from")

  local file = io.open(input_path, "r")
  if file == nil then error("Error opening file " .. input_path) end

  dmn.logger:print("Reading raw line data from " .. input_path)

  local inputs = {}

  local curr_index = 1
  while true do
    if curr_index % 10000 == 0 then 
      dmn.logger:print("On " .. curr_index)
    end

    line = file:read()
    if line == nil then break end
    table.insert(inputs, line)

    curr_index = curr_index + 1
  end

  file:close()
  return inputs
end

-- reads tsv data into an array of arrays
function datasets.read_tabbed_data(input_path, delimiter)
  local delimiter = (delimiter == nil) and "\t" or delimiter

  local file = io.open(input_path, "r")
  if file == nil then error("Error opening file " .. input_path) end

  local data = {}

  local curr_index = 1
  while true do
    if curr_index % 1000 == 0 then 
      dmn.logger:print("On " .. curr_index)
    end

    line = file:read()
    if line == nil then break end
    local items = stringx.split(line, delimiter)
    table.insert(data, items)

    curr_index = curr_index + 1
  end

  file:close()
  return data
end

-- Reads tsv data from file, either reads to max_amount or reads all of the data
function datasets.read_tsv_data(input_path, max_amount)
  local max_read_size = (max_amount == nil) and 100000000 or max_amount
  dmn.logger:print("Reading tsv data from " .. input_path)

  local file = io.open(input_path, "r")
  if file == nil then error("Error opening file " .. input_path) end

  local questions = {}
  local positive_samples = {}

  local curr_index = 1
  while true do
    line = file:read()
    curr_index = curr_index + 1

    if curr_index % 1000 == 0 then 
      dmn.logger:print("On " .. curr_index)
    end

    if curr_index > max_read_size then break end
    if line == nil then break end
    local items = line:split("\t")
    local question = items[1]
    local target = items[2]
    table.insert(questions, question)
    table.insert(positive_samples, target)
  end

  file:close()
  return questions, positive_samples
end

-- saves a table of strings into a line separated file
function datasets.save_line_data(items, save_path)
  assert(items ~= nil, "Must specify table to save")
  assert(save_path ~= nil, "Must specify save_path to save to")

  dmn.logger:print("Saving line data to " .. save_path)
  make_dir(save_path)

  local file = io.open(save_path, "w")
  if file == nil then
    error("Error opening file " .. save_path .. "\n") 
  end

  -- write questions
  for i = 1, #items do
    if i % 1000 == 0 then
      dmn.logger:print("Writing line " .. i)
    end
    local curr_item = items[i]
    file:write(curr_item)
    if i ~= #items then
      file:write("\n")
    end
  end

  file:close()
  dmn.logger:print("Done saving line data to " .. save_path)
end

-- saves a table of a table of strings into a tsv file
function datasets.save_tsv_data(table, save_path)
  assert(table ~= nil, "Must specify table to save")
  assert(save_path ~= nil, "Must specify save_path to save to")

  make_dir(save_path)

  local file = io.open(save_path, "w")
  if file == nil then
    error("Error opening file " .. save_path .. "\n") 
  end

  for i = 1, #table do
    if i % 1000 == 0 then
      dmn.logger:print("Writing line " .. i)
    end
    local curr_inputs = table[i]
    for j = 1, #curr_inputs do
      local curr_input = curr_inputs[j]
      file:write(curr_input)
      if j ~= #curr_inputs then
        file:write("\t")
      end
    end
    if i ~= #table then
      file:write("\n")
    end
  end

  file:close()
  dmn.logger:print("Done saving tsv data to path " .. save_path)
end

function datasets.save_tsv_data_indeces(table, indeces_to_table, save_path)
  assert(table ~= nil, "Must specify table to save")
  assert(indeces_to_table ~= nil, "Must specify indeces to table")
  assert(save_path ~= nil, "Must specify save_path to save to")

  make_dir(save_path)

  local file = io.open(save_path, "w")
  if file == nil then
    error("Error opening file " .. save_path .. "\n") 
  end

  for i = 1, #indeces_to_table do
    if i % 1000 == 0 then
      dmn.logger:log("Writing line " .. i)
    end
    local curr_inputs = indeces_to_table[i]
    for j = 1, #curr_inputs do
      local curr_index = curr_inputs[j]
      local curr_input = table[curr_index]
      file:write(curr_input)
      if j ~= #curr_inputs then
        file:write("\t")
      end
    end
    if i ~= #table then
      file:write("\n")
    end
  end

  file:close()
end

-- Saves the vocab tokens, places one token per line
function datasets.save_vocab(vocab_list, threshold, save_path)
  assert(vocab_list ~= nil, "Must specify vocab list to save to file")
  assert(threshold ~= nil, "Must specify threshold to use to save to file")
  assert(save_path ~= nil, "Must specify save path to use")

  dmn.logger:log("Saving vocabulary to path " .. save_path)
	-- Include special start symbol and end symbol
  local file = torch.DiskFile(save_path, 'w')
  assert(file:isWritable(), "Must be able to write to file")

  if file == nil then error("Error opening file " .. save_path .. "\n") end

  for i = 1, #vocab_list.words do
  	local curr_word = vocab_list.words[i]
  	local curr_count = vocab_list.counts[curr_word]

  	if curr_count >= threshold then
      --dmn.logger:print("Saving word " .. curr_word .. " with count " .. curr_count)
  		file:writeString(curr_word)
      local err = file:hasError()
      if i ~= #vocab_list.words then
      file:writeString("\n")
      end
  	end
  end

  -- MUST CLOSE FILE AT THE END OR THERE IS AN ERROR READING
  file:close()
  dmn.logger:print("Done saving vocab to path " .. save_path)
end


-- Generates vocab tokens for questions/inputs/answers.
-- By default uses same vocab for questions, inputs, and answers
-- Questions and answers are an array of sentences
-- Inputs are an array of arrays of sentences
function datasets.generate_vocab(questions, inputs, answers, char_level)
	assert(questions ~= nil, "Must specify questions to extract tokens from")
	assert(inputs ~= nil, "Must specify inputs to extract tokens from")
	assert(answers ~= nil, "Must specify answers to extract tokens from")
  assert(char_level ~= nil, "Must specify whether to use character level tokenization or not")

	local size = 1
	local words = {}
	local counts = {}

    -- add in questions
	for i = 1, #questions do
    if i % 10000 == 0 then 
      dmn.logger:print("On question index " .. i .. " of " .. #questions)
      dmn.functions.collect_garbage()
    end

		local tokens = datasets.tokenize_text(questions[i], char_level)
		for j = 1, #tokens do
			local token = tokens[j]
			local added_token = datasets.add_token(token, counts, words, size)
			if added_token then
				size = size + 1
			end
		end
	end

    -- add in inputs
	for i = 1, #inputs do
    if i % 10000 == 0 then 
      dmn.logger:print("On question index " .. i .. " of " .. #inputs)
      dmn.functions.collect_garbage()
    end

		local curr_input = inputs[i]
		for k = 1, #curr_input do
			local tokens = datasets.tokenize_text(curr_input[k], char_level)
			for j = 1, #tokens do
				local token = tokens[j]
				local added_token = datasets.add_token(token, counts, words, size)
				if added_token then
					size = size + 1
				end
			end
		end
	end

	-- add in answers
	for i = 1, #answers do
    if i % 10000 == 0 then 
      dmn.logger:print("On question index " .. i .. " of " .. #answers)
      dmn.functions.collect_garbage()
    end

		local tokens = datasets.tokenize_text(answers[i], char_level)
		for j = 1, #tokens do
			local token = tokens[j]
			local added_token = datasets.add_token(token, counts, words, size)
			if added_token then
				size = size + 1
			end
		end
	end

	local vocab = {}
	vocab.words = words
	vocab.counts = counts

  dmn.functions.collect_garbage()
	return vocab
end

function datasets.generate_single_vocab(sentences, use_char)
  assert(sentences ~= nil, "Must specify sentences to extract tokens from")
  assert(use_char ~= nil, "Must specify whether to use character level tokenization or not")

  local size = 1
  local words = {}
  local counts = {}

    -- add in questions
  for i = 1, #sentences do
    local tokens = datasets.tokenize_text(sentences[i], use_char)
    dmn.logger:print("Reading " .. i .. " of " .. #sentences)
    for j = 1, #tokens do
      local token = tokens[j]
      local added_token = datasets.add_token(token, counts, words, size)
      if added_token then
        size = size + 1
      end
    end
  end

  local vocab = {}
  vocab.words = words
  vocab.counts = counts
  return vocab
end

function datasets.add_token(token, curr_counts, curr_words, size)
	assert(token ~= nil, "Must specify token to add to vocab")
	assert(curr_counts ~= nil, "Must specify current counts to add token to")
	assert(curr_words ~= nil, "Must specify current words in dictionary")

	if curr_counts[token] == nil then
		curr_counts[token] = 1
		curr_words[size] = token
		return true
	else
		curr_counts[token] = curr_counts[token] + 1
		return false
	end
	
end

function datasets.read_embedding(vocab_path, emb_path)
  -- Reads vocabulary from vocab_path, 
  -- Reads word embeddings from embd_path
  local vocab = datasets.Vocab(vocab_path, false)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

-- creates tokens based on word counts in path
function datasets.get_vocab(path, dummy_path)
  -- Reads sentences from specified path
  -- Reads vocab from vocab path
  local vocab = datasets.Vocab(dummy_path, true)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      vocab:add(token)
    end
  end

  file:close()
  return vocab
end

-- creates tokens based on word counts in path
function datasets.get_hashed_vocab(path, dummy_path)
  -- Reads sentences from specified path
  -- Reads vocab from vocab path
  local vocab = datasets.HashVocab(dummy_path, true)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    for i = 1, len do
      local token = tokens[i]
      vocab:add(token)
    end
  end

  file:close()
  return vocab
end

function datasets.read_sentences(path, vocab, gpu_mode, char_level, pad_size)
  assert(path ~= nil, "Must specify path to read sentences from")
  assert(vocab ~= nil, "Must specify vocab to read sentences with")
  assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")
  assert(char_level ~= nil, "Must specify whether to read with character or not")
  assert(pad_size ~= nil, "Must specify whether to read with pad size or not")
  -- Reads sentences from specified path
  -- Reads vocab from vocab path
  local sentences = {}
  local file = io.open(path, 'r')
  if file == nil then error("Cannot open file " .. path) end
  local line
  local curr_index = 1
  while true do
    if curr_index % 1000 == 0 then 
      collectgarbage()
      dmn.logger:print("Reading line ", curr_index)
    end
    curr_index = curr_index + 1
    line = file:read()
    if line == nil then break end
    local sent = datasets.get_input_tokens(line, vocab, gpu_mode, char_level, pad_size)
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

-- gets input tokens from sentence depending on whether it's gpu or cpu mode
function datasets.get_input_tokens(sentence, vocab, gpu_mode, tokenization_type, pad_size)
  assert(sentence ~= nil, "Sentence must be not null")
  assert(vocab ~= nil, "Vocab must not be null")
  assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")
  assert(tokenization_type ~= nil, "Must specify whether to read with character or not")
  assert(pad_size ~= nil, "Must specify whether to read with pad size or not")

  tokens = datasets.tokenize_text(sentence, tokenization_type)

  -- only add symbols if there is no tokenization
  if tokenization_type ~= dmn.constants.NO_TOKENIZATION then 
    for i = 1, pad_size do 
      table.insert(tokens, '<pad>')
    end

    table.insert(tokens, '</s>')

    for i = 1, pad_size do 
      table.insert(tokens, 1, '<pad>')
    end
    table.insert(tokens, 1, '<s>')
  end
 
  -- also we need to check if it's cuda or not
  local sent = vocab:map(tokens, gpu_mode)
  return sent
end

-- Reads tab-separated sentences from specified path.
-- Returns an array of an array of sentences
-- All sentences corresponding to question are on a single line
-- All sentences on a single line are tab separated
function datasets.read_multisentences(path, vocab, gpu_mode, char_level, pad_size)
  assert(path ~= nil, "Must specify path to read vocab from")
  assert(vocab ~= nil, "Must specify vocab object to map words to tokens/hashed reps")
  assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")
  assert(char_level ~= nil, "Must specify whether to use character level tokenization or not")
  assert(pad_size ~= nil, "Must specify pad size to use for reading in sentences")

  -- Reads vocab from vocab path
  local tot_sentences = {}
  local file = io.open(path, 'r')
  if file == nil then error("Cannot open file " .. path) end
  local line
  local tot_num_lines_read = 0
  while true do
    line = file:read()
    tot_num_lines_read = tot_num_lines_read + 1

    if tot_num_lines_read % 1000 == 0 then 
      collectgarbage()
      dmn.logger:log("Reading multisentence line " .. tot_num_lines_read)
    end

    if line == nil then break end
    local cur_sent_arr = {}
    local cur_sentences = line:split("\t")
    for i = 1, #cur_sentences do
      local sentence = cur_sentences[i]
      tokens = datasets.get_input_tokens(sentence, vocab, gpu_mode, char_level, pad_size)
      cur_sent_arr[#cur_sent_arr + 1] = tokens
    end
    tot_sentences[#tot_sentences + 1] = cur_sent_arr
  end

  file:close()
  return tot_sentences
end

-- Reads sentences from specified path but adds special <s>
-- token to input sentence and </s> to end of output sentence.
-- so that datasets can learn to predict next token.
function datasets.read_predict_sentences(path, vocab, gpu_mode, char_level)
   assert(path ~= nil, "Must specify path to read vocab from")
   assert(vocab ~= nil, "Must specify vocab object to map words to tokens/hashed reps")
   assert(gpu_mode ~= nil, "Must specify whether to use gpu mode or not")
   assert(char_level ~= nil, "Must specify whether to use character level tokenization or not")

    local in_sentences = {}
    local out_sentences = {}
    local file = io.open(path, 'r')
    local line
    while true do
      line = file:read()
      if line == nil then break end
      tokens = datasets.tokenize_text(line, char_level)
        -- first get labels for sentence
      table.insert(tokens, '</s>')
      local out_ids = vocab:map_no_unk(tokens, gpu_mode)
      table.insert(tokens, 1, "<s>")
      table.remove(tokens)
      local in_ids = vocab:map_no_unk(tokens, gpu_mode)
      
      in_sentences[#in_sentences + 1] = gpu_mode and in_ids:cuda() or in_ids 
      out_sentences[#out_sentences + 1] = gpu_mode and out_ids:cuda() or out_ids
    end

    file:close()
    return in_sentences, out_sentences
end

-- Tokenizes text. Returns an array of characters/tokens.
-- Uses tokens by default.
function datasets.tokenize_text(input_text, tokenization_type)
  assert(input_text ~= nil, "Must specify input text to tokenize")
  assert(tokenization_type ~= nil, "Must specify whether to use input tokenization or not")
  local tokens
  if tokenization_type == dmn.constants.CHAR_LEVEL then 
    tokens = datasets.tokenize_text_char(input_text)
    --tokens = datasets.tokenize_text_reg(input_text)
  elseif tokenization_type == dmn.constants.WORD_LEVEL then
    tokens = datasets.tokenize_text_reg(input_text)
  elseif tokenization_type == dmn.constants.NO_TOKENIZATION then
    tokens = input_text:split('\t')
  else
    error("Invalid character type given")
  end
  return tokens
end


-- extracts all characters from the string
function datasets.tokenize_text_char(input_text)
  assert(input_text ~= nil, "Must specify input text to tokenize")
  local tokens = {}
  for i = 1, #input_text do
    local c = input_text:sub(i,i)
    table.insert(tokens, c)
    -- do something with c
  end
  return tokens
end

function datasets.tokenize_text_reg(input_text)
	assert(input_text ~= nil, "Must specify input text to tokenize")
	function replace_tokens(curr_line, tokens_list, replace_list)
    for i = 1, #tokens_list do
      curr_line = stringx.replace(curr_line, tokens_list[i], replace_list[i])
    end
    return trim(curr_line)
  end

  function replace_regex(curr_line, tokens_list, replace_list)
    for i = 1, #tokens_list do
      curr_line = string.gsub(curr_line, tokens_list[i], replace_list[i])
    end
    return trim(curr_line)
  end

    local regex_list = {'%p', '%s+'}
    local new_regex_list = {' ', ' '}

    local tokens_list = {'?', 
                         ',', 
                         '(', 
                         ')', 
                         '\n', 
                         '.', 
                         '\r', 
                         '\r\n', 
                         '?', 
                         '.', 
                         '-', 
                         '"',
                         '/'}
    local new_list = {' ?',
                      ' , ', 
                      ' (', 
                      ') ', 
                      '', 
                      ' .', 
                      '', 
                      '', 
                      ' ?', 
                      ' .', 
                      ' - ', 
                      '"',
                      ' '
                    }

    local trimmed_line = replace_tokens(string.lower(input_text), tokens_list, new_list)
    local regexed_line = replace_regex(trimmed_line, regex_list, new_regex_list)
    local tokens = stringx.split(regexed_line)

    return tokens
end
