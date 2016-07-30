--[[
	Batch forwarding LSTMs test
]]

require('..')

local config = {
	in_dim = 5,
	mem_dim = 10,
	num_layers = 1,
	gpu_mode = false,
	dropout = false,
	lstm_type = 'lstm',
	gru_type = 'gru'
}

local test_network = function(network, network_type)
	local hidden_batch_inputs = network_type == 'lstm' and {torch.rand(8, 10), torch.rand(8, 10)} 
														or torch.rand(8, 10)
	local batch_inputs = torch.rand(3, 8, 5)



	local t1 = sys.clock()
	results = network:forward(batch_inputs, hidden_batch_inputs, false)
	local rand_outputs = results:clone():fill(0.5)
	
	results1, results2 = network:backward(batch_inputs, hidden_batch_inputs, false, rand_outputs)
	local t2 = sys.clock()

	--print(results:size())
	print((t2 - t1) / 100)

	local t1 = sys.clock()
	for i = 1, batch_inputs:size(2) do 
		local cur_input = torch.squeeze(batch_inputs[{{},i}])
		local cur_hidden_input = network_type == 'lstm' and {hidden_batch_inputs[1][i], hidden_batch_inputs[2][i]}
														or hidden_batch_inputs[i]
		local cur_outputs = torch.squeeze(rand_outputs[{{}, i}])
		results_single = network:forward(cur_input, cur_hidden_input, false)
		results1_single, results2_single = network:backward(cur_input, cur_hidden_input, false, cur_outputs)
		local diff = results_single - torch.squeeze(results[{{},i}])
		local grad_diff = results1_single - torch.squeeze(results1[{{}, i}])
		
		
		local hidden_grad_diff1, hidden_grad_diff2
		if network_type == 'lstm' then 
			hidden_grad_diff1 = results2_single[1]
				- torch.squeeze(results2[1][i])
			hidden_grad_diff2 = results2_single[2]
				- torch.squeeze(results2[2][i])
		else 
			hidden_grad_diff1 = results2_single - results2[i]
			hidden_grad_diff2 = hidden_grad_diff1
		end

		print("Difference in results " .. i .. " " .. torch.abs(diff):sum())
		print("Difference in grads " .. i .. " " .. torch.abs(grad_diff):sum())
		print("Difference in cell grads " .. i .. " " .. torch.abs(hidden_grad_diff1):sum())
		print("Difference in hidden grads " .. i .. " " .. torch.abs(hidden_grad_diff2):sum())
	end
end

print("Testing lstms")
-- lstm
local lstm_decoder_network = dmn.LSTM_Decoder(config)
local gru_decoder_network = dmn.GRU_Decoder(config)

print("Testing lstm decoder with regular lstm")
test_network(lstm_decoder_network, 'lstm')

print("Testing gru decoder with regular gru")
test_network(gru_decoder_network, 'gru')

config.lstm_type = 'gf_lstm'
config.gru_type = 'gf_gru'
local lstm_decoder_network = dmn.LSTM_Decoder(config)
local gru_decoder_network = dmn.GRU_Decoder(config)

print("Testing lstm decoder with gf lstm")
test_network(lstm_decoder_network, 'lstm')

print("Testing gru decoder with gf gru")
test_network(gru_decoder_network, 'gru')