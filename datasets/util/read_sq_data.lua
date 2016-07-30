--[[

  Functions for loading sq dataset from disk

--]]

-- loads original sq dataset from file
-- sq dataset is of the form
-- Subject-entity [tab] relationship [tab] Object-entity [tab] question per line
function datasets.read_sq_dataset_helper(dataset_path)
  assert(dataset_path ~= nil, "Must specify dataset path for reading")
  print('Reading sq dataset from ' .. dataset_path)

  local sq_dataset = {}
  local subject_entities = {}
  local relationships = {}
  local object_entities = {}
  local questions = {}

  -- Include special start symbol and end symbol
  local file = io.open(dataset_path)
  if file == nil then error("Error opening file " .. dataset_path .. "\n") end

  local curr_index = 1
  while true do
    if curr_index % 10000 == 0 then 
      dmn.logger:print("On index " .. curr_index)
    end
    local line = file:read()
    if line == nil then break end
    local items = line:split('\t')
    local subject_entity = stringx.replace(items[1], 'www.freebase.com', '')
    local relationship = stringx.replace(items[2], 'www.freebase.com', '')
    local object_entity = stringx.replace(items[3], 'www.freebase.com', '')
    local question = items[4]

    subject_entities[curr_index] = subject_entity
    relationships[curr_index] = relationship
    object_entities[curr_index] = object_entity
    questions[curr_index] = question

    curr_index = curr_index + 1
  end

  file:close()

  sq_dataset.subject_entities = subject_entities
  sq_dataset.predicates = relationships
  sq_dataset.object_entities = object_entities
  sq_dataset.questions = questions
  sq_dataset.size = #questions
  return sq_dataset
end

-- loads simplequestion dataset dataset in addition to 
-- returns: imagelstm.Vocab, train, val, test datasets in that order
function datasets.sq_read_dataset_raw(base_path)
    -- directory containing dataset files
  local data_dir = base_path

  -- load train dataset
  local train_dataset_path = data_dir .. "annotated_fb_data_train.txt"
  local train_dataset = datasets.read_sq_dataset_helper(train_dataset_path)

   -- load validation dataset
  local val_dataset_path = data_dir .. "annotated_fb_data_valid.txt"
  local val_dataset = datasets.read_sq_dataset_helper(val_dataset_path)

   -- load test dataset
  local test_dataset_path = data_dir .. "annotated_fb_data_test.txt"
  local test_dataset = datasets.read_sq_dataset_helper(test_dataset_path)

  -- add names to the datasets
  local train_subject_names = datasets.read_line_data(data_dir .. "train/subject_names.txt")
  local test_subject_names = datasets.read_line_data(data_dir .. "test/subject_names.txt")
  local val_subject_names = datasets.read_line_data(data_dir .. "val/subject_names.txt")

  train_dataset.subject_entity_names = train_subject_names
  test_dataset.subject_entity_names = test_subject_names
  val_dataset.subject_entity_names = val_subject_names

  return train_dataset, val_dataset, test_dataset
end

function datasets.sq_read_dataset_parsed(base_dir, gpu_mode, 
  use_question_hashing, use_predicate_hashing, use_entity_hashing,
  char_level, pad_size)
  assert(dir ~= nil, "Directory is null")
  assert(gpu_mode ~= nil, "Must specify gpu mode for reading dataset")
  assert(use_question_hashing ~= nil, "Must specify whether to use hashing for questions or not")
  assert(use_predicate_hashing ~= nil, "Must specify whether to use hashing for predicates or not")
  assert(use_entity_hashing ~= nil, "Must specify whether to use hashing for entities or not")
  assert(char_level ~= nil, "Must specify whether to use char level or not")
  assert(pad_size ~= nil, "Must specify pad size to use")

  -- get vocab
  local predicate_vocab_type = use_predicate_hashing and datasets.SparseHashVocab or datasets.Vocab
  local entity_vocab_type = use_entity_hashing and datasets.SparseHashVocab or datasets.Vocab
  local question_vocab_type = use_question_hashing and datasets.SparseHashVocab or datasets.Vocab
  
  local vocab_prefix = char_level and "_char" or ""
  local entity_vocab = entity_vocab_type(base_dir .. 
    string.format('entity_vocab%s.txt', vocab_prefix),
    true)
  local predicate_vocab = predicate_vocab_type(base_dir .. 
    string.format('predicate_vocab%s.txt', vocab_prefix), true)
  local question_vocab = question_vocab_type(base_dir .. 
    string.format('question_vocab%s.txt', vocab_prefix), true)

  -- get train paths
  local train_path = base_dir .. "train/"
  local val_path = base_dir .. "val/"
  local test_path = base_dir .. "test/"

  local train_dataset = datasets.sq_read_dataset_parsed_helper(train_path,
                     gpu_mode, question_vocab, predicate_vocab, entity_vocab,
                     char_level, pad_size)
  local val_dataset = datasets.sq_read_dataset_parsed_helper(val_path,
                        gpu_mode, question_vocab, predicate_vocab, entity_vocab,
                        char_level, pad_size)
  local test_dataset = datasets.sq_read_dataset_parsed_helper(test_path,
                        gpu_mode, question_vocab, predicate_vocab, entity_vocab,
                        char_level, pad_size)

  train_dataset.question_vocab = question_vocab
  train_dataset.entity_vocab = entity_vocab
  train_dataset.predicate_vocab = predicate_vocab
  
  return train_dataset, val_dataset, test_dataset 
end

-- reads question answer dataset with single inputs
-- dir: where sentences are
-- input_vocab: Vocab for input
-- question_vocab: Vocab for questions
function datasets.sq_read_dataset_parsed_helper(dir, gpu_mode, question_vocab, 
  predicate_vocab, entity_vocab, char_level, pad_size)
  assert(dir ~= nil, "Directory is null")
  assert(gpu_mode ~= nil, "Must specify gpu mode for reading dataset")
  assert(question_vocab ~= nil, "Must specify question vocab")
  assert(predicate_vocab ~= nil, "Must specify predicate vocab")
  assert(entity_vocab ~= nil, "Must specify entity vocab")
  assert(char_level ~= nil, "Must specify whether to use char level or not")
  assert(pad_size ~= nil, "Must specify pad size to use")

  dmn.logger:print("Reading sq dataset from " .. dir)

  local dataset = {}

  dmn.logger:print("Reading questions")
  -- first read questions
  dataset.questions = datasets.read_sentences(dir .. 'questions.txt', question_vocab, gpu_mode, char_level, pad_size)
  dataset.raw_questions = datasets.read_line_data(dir .. 'questions.txt')

  -- then read predicates
  dmn.logger:print("Reading predicates")
  --dataset.candidate_predicates = datasets.read_sentences(dir .. 'all_predicates.txt', predicate_vocab, gpu_mode)
  dataset.positive_predicates = datasets.read_sentences(dir .. 'positive_predicates.txt', predicate_vocab, gpu_mode,
    char_level, pad_size)
  dataset.raw_positive_predicates = datasets.read_line_data(dir .. 'positive_predicates.txt')
  dataset.positive_predicate_probabilities = dmn.functions.count_values(dataset.raw_positive_predicates)

  dmn.logger:print("Reading entities")
  dataset.positive_entities = datasets.read_sentences(dir .. 'positive_entities.txt', entity_vocab, gpu_mode, 
    char_level, pad_size)
  dataset.raw_positive_entities = datasets.read_line_data(dir .. 'positive_entities.txt')
  dataset.raw_positive_inputs = raw_positive_inputs
  dataset.positive_entity_probabilities = dmn.functions.count_values(dataset.raw_positive_entities)

  dataset.question_vocab = question_vocab
  dataset.predicate_vocab = predicate_vocab
  dataset.entity_vocab = entity_vocab

  -- size
  dataset.size = #dataset.questions

  collectgarbage()
  return dataset
end