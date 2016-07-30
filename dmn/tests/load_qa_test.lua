--[[
Run loading QA model tests
]]

require('..')

-- Tests gpus
gpu_test = {}

tester = torch.Tester()
answer_vocab = {}
question_vocab = {}
input_vocab = {}

answer_vocab.hashed = false
answer_vocab.size = 25000
question_vocab.hashed = false
question_vocab.size = 25000
input_vocab.hashed = false
input_vocab.size = 25000

function gpu_test.Test_Attention()

local config = {
          optim_state = {learningRate = 1e-4},
          optim_method_string = "rmsprop",
          gpu_mode = false,
          e_vocab = input_vocab,
          p_vocab = answer_vocab,
          q_vocab = question_vocab,
          batch_size = 100,
          attention_mem_dim = 100,
          attention_num_layers = 1,
          question_emb_dim = 200,
          question_mem_dim = 100,
          question_num_layers = 2,
          question_num_classes = question_vocab.size,
          question_dropout_prob = 0.5,
          question_dropout = false,
          entity_emb_dim = 300,--00,
          entity_out_dim = 100,--0,
          entity_hidden_dim = 200,--0,
          entity_in_stride = 1,
          entity_in_kernel_width = 2,
          entity_hidden_kernel_width = 2,
          entity_hidden_stride = 1,
          entity_out_kernel_width = 1,
          entity_out_stride = 1,
          entity_num_classes = input_vocab.size,
          entity_dropout_prob = 0.5,
          entity_dropout = false,
          predicate_emb_dim = 300,
          predicate_out_dim = 100,
          predicate_hidden_dim = 200,
          predicate_in_stride = 1,
          predicate_in_kernel_width = 3,
          predicate_hidden_kernel_width = 2,
          predicate_hidden_stride = 1,
          predicate_out_kernel_width = 2,
          predicate_out_stride = 1,
          predicate_num_classes = question_vocab.size,
          predicate_dropout_prob = 0.5,
          predicate_dropout = false}


	-- creates our model
    local cpu_model = dmn.Attention_Network(config)
    cpu_model:save("test.th")
    local loaded_model = dmn.Attention_Network.load("test.th")
	
	local function forward_inputs(model, question_indices, predicate_indices, entity_indices, corr_entity_index, corr_predicate_index)
	    output = model:forward(question_indices, predicate_indices, entity_indices, corr_entity_index, corr_predicate_index)
	   	return output
	end

	-- Test gpu inputs
	-- Create inputs
	local question_indices = torch.IntTensor(30):random(5, 5000)

	local table_word_indices = {}
		for i = 1, 10 do
			local word_indices = torch.IntTensor(100):random(5, 5000)
			table.insert(table_word_indices, word_indices)
		end
	local correct_index = 1

	-- Load/save model
	local original_model_res = forward_inputs(cpu_model, question_indices, table_word_indices, table_word_indices, correct_index, correct_index)
    local loaded_model_res = forward_inputs(loaded_model, question_indices, table_word_indices, table_word_indices, correct_index, correct_index)

    print(original_model_res)
    print(loaded_model_res)

  	tester:assertlt(torch.abs(original_model_res - loaded_model_res), 1e-7, 'Difference between outputs must be less than 1e-7')
end


tester:add(gpu_test)
tester:run()

