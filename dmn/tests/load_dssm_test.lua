--[[

  DSSM testing scripts: makes sure that loaded model gives exactly same predictions as trained model

--]]


require('..')

dmn.logger:header("Testing loading/saving DSSM Network")

local dataset_path, model_params, dir_params = unpack(require('opts/dssm_opts.lua'))

if model_params.use_gpu_mode then 
  dmn.logger:print("Loading gpu modules")
  require('cutorch') -- uncomment for GPU mode
  require('cunn') -- uncomment for GPU mode
end

-- Create random model and 
local dssm_trainer = softmax.DSSM_Trainer()
local predicate_vocab = {size = 2000, hashed = false}
local entity_vocab = {size = 2000, hashed = false}
local model = dssm_trainer:load_model(model_params, predicate_vocab, entity_vocab)

local model_save_path = 'dummy_model.th'
model:save(model_save_path, model_params.epochs)

local loaded_model = dmn.DSSM_Network.load(model_save_path)
local question = torch.IntTensor{1, 2, 3, 6, 7, 2, 3, 120, 293, 120, 239, 491, 230, 129, 203, 45, 345,}
local predicates = {torch.IntTensor{1, 2, 623, 102, 239, 2, 3, 120, 293, 120, 239, 491}, 
torch.IntTensor{5, 6, 8, 9, 10,  491, 230, 129, 203, 45, 345}, torch.IntTensor{45, 233, 121, 1234,
3, 6, 7, 2, 3, 120, 293, }}

local results = model:predict_tokenized(question, predicates, #predicates)
local loaded_results = loaded_model:predict_tokenized(question, predicates, #predicates)

for i = 1, #results do 
	local model_result = results[i][1]
	local loaded_model_result = loaded_results[i][1]
	dmn.logger:print(model_result .. " " .. loaded_model_result)
	assert(model_result == loaded_model_result, "Model results must match")
end







