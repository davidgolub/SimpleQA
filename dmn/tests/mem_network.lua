require('.')

local vocab_size = 30

local dmn_network = dmn.DMN_Network{
	 				gpu_mode = false,
	 				question_num_classes = vocab_size,
	 				question_emb_dim = 2,
	 				question_in_dim = 2,
	 				question_mem_dim = 5,
	 				question_num_layers = 1,
	 				question_dropout_prob = 0.5,
	 				question_dropout = false,
	 				answer_num_classes = vocab_size,
	 				answer_emb_dim = 2,
	 				answer_input_dim = 2,
	 				answer_mem_dim = 5,
	 				answer_num_layers = 1,
	 				answer_in_dropout_prob = 0.5,
	 				answer_hidden_dropout_prob = 0.5,
	 				answer_dropout = false,
	 				episodic_mem_dim = 5,
	 				episodic_gate_size = 5,
	 				episodic_num_episodes = 5,
	 				semantic_num_classes = vocab_size,
	 				semantic_emb_dim = 2,
	 				semantic_in_dim = 2,
	 				semantic_mem_dim = 5,
	 				semantic_num_layers = 1,
	 				semantic_dropout_prob = 0.5,
	 				semantic_dropout = false
					}

local question_indices = torch.IntTensor{1, 2, 3, 4, 8, 9, 10, 11, 12, 15, 29}
local word_indices = torch.IntTensor{1, 2, 3, 4, 8, 9, 10, 11, 12, 15, 18}
local input_indices = torch.IntTensor{1, 2, 3, 4, 8, 9, 10, 11, 12, 15, 27}
local output_indices = torch.IntTensor{1, 2, 3, 4, 8, 9, 10, 11, 12, 15, 20}

print("Forwarding network")
local start_time = sys.clock()
local err = dmn_network:forward(question_indices, word_indices, input_indices, output_indices)
local err1 = dmn_network:backward(question_indices, word_indices, input_indices, output_indices)
local end_time = sys.clock()
print(start_time - end_time)
print("Done forwarding network")

dmn_network:grad_check()

