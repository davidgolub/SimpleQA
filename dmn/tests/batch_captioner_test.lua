require('..')

local dataset_path, model_params, dir_params, data_params = unpack(require('opts/context_captioning/caption_opts_piano.lua'))

-- Tests that batch captioner works
train_dataset, val_dataset, test_dataset
 = datasets.read_caption_data(dataset_path, 
 	model_params.use_gpu_mode, 
 	model_params.char_level,
 	model_params.only_feats)

vocab = train_dataset.vocab

local pred_beam_size = 1
local test_beam_size = 1

	-- initialize model
local model = dmn.Captioner_Network{
  char_level = model_params.char_level,
  vocab = vocab,
  only_feats = model_params.only_feats,
  load_weights = model_params.load_weights,
  batch_size = model_params.batch_size,
  optim_method_string = model_params.optim_method_string,
  optim_state = model_params.optim_state,
  network_type = model_params.network_type,
  gpu_mode = model_params.use_gpu_mode,
  num_classes = vocab.size,
  emb_dim = model_params.emb_dim,
  mem_dim = model_params.mem_dim,
  image_dim = model_params.image_dim,
  num_layers = model_params.num_layers,
  dropout = model_params.dropout,
  in_dropout_prob = model_params.in_dropout_prob,
  hidden_dropout_prob = model_params.hidden_dropout_prob,
  rnn_type = model_params.rnn_type,
  cell_type = model_params.cell_type,
  tune_image_features = model_params.tune_image_features
}

local images = torch.rand(4, 3, 224, 224)
local input_sentences = torch.IntTensor{{1, 2, 3, 4}, 
										{1, 4, 5, 6}, 
										{1, 7, 8, 9}}

-- input_sentences[i] = batch of indices corresponding to word i.
local output_sentences = torch.IntTensor{{1, 2, 3, 4}, {1, 4, 5, 6}, {1, 7, 8, 9}}

local input_masks = torch.IntTensor(4, 2)
input_masks[1][1] = 5
input_masks[1][2] = 3
input_masks[2][1] = 5
input_masks[2][2] = 2
input_masks[3][1] = 5
input_masks[3][2] = 3
input_masks[4][1] = 5
input_masks[4][2] = 3

model.grad_params:zero()
local batch_loss, predictions = model:forward(images, input_sentences, output_sentences, false, input_masks)
local input_errs = model:backward(images, input_sentences, output_sentences, false, input_masks)
local batch_sum = model.grad_params:sum()

model.grad_params:zero()

local cur_loss = 0
-- should be the same as batch forwarding
for i = 1, images:size(1) do 
  local loss, predictions = model:forward(images[i], input_sentences[{{},i}], output_sentences[{{},i}], false, input_masks)
  model:backward(images[i], input_sentences[{{},i}], output_sentences[{{},i}], false, input_masks)
  cur_loss = cur_loss + loss
end

print(cur_loss / images:size(1))
print(batch_loss)

local sequential_sum = model.grad_params:sum() / images:size(1)

print("Batch sum, sequential sum ", batch_sum, sequential_sum)
assert(batch_sum == sequential_sum, "Two must be equal")