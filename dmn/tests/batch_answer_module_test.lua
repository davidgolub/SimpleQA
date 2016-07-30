require('..')

local network = dmn.AnswerModule{
 gpu_mode = false,
 num_classes = 10,
 emb_dim = 100,
 mem_dim = 100,
 num_layers = 1,
 in_dropout_prob = 0.0,
 hidden_dropout_prob = 0.0,
 dropout = false,
 rnn_type = 'gf_lstm',
 cell_type = 'gf_lstm'
}

local test_network = function(network)
	local inputs = torch.rand(10, 100)
	local input_indices = torch.IntTensor{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
	local desired_indices = torch.IntTensor{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
	local mask = torch.IntTensor(10, 2)

	for i = 1, mask:size(1) do
		mask[i][1] = 5
		mask[i][2] = 4
	end

	local batch_loss, predictions, batch_lstm_output = network:forward(inputs, input_indices, desired_indices, mask)
	local memory_errors = network:backward(inputs, input_indices, desired_indices, mask)

	print("PREDICTIONS ARE")
	print(predictions)

	print(memory_errors:size())

	local tot_loss = 0
	for i = 1, inputs:size(1) do
		local cur_input = inputs[i]
		local cur_input_indices = torch.squeeze(input_indices[{{},i}])
		local cur_desired_indices = torch.squeeze(desired_indices[{{}, i}])
		local curr_loss, curr_predictions, cur_lstm_output = network:forward(
			cur_input,
			cur_input_indices, 
			cur_desired_indices, 
			mask)
		local cur_memory_errors = network:backward(
			cur_input,
			cur_input_indices, 
			cur_desired_indices, 
			mask)

		print("Prediction difference")
		local diff = curr_predictions - predictions[{{},i}]
		local input_mem_diff = cur_memory_errors - memory_errors[i]
		local cur_lstm_output = batch_lstm_output[{{}, i}]
		print("Input memory difference " .. torch.abs(input_mem_diff):sum())
		print(torch.abs(diff):sum())
		tot_loss = tot_loss + curr_loss
	end

	print(tot_loss / inputs:size(1))
	print(batch_loss)
end

test_network(network)

--network.rnn_type = 'gru'
--test_network(network)