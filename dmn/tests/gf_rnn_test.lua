require('..')

-- code to test basic forwarding of rnn units
function test_unit(input_size, mem_size, num_layers)
	test_lstm_unit = dmn.rnn_units.gf_lstm(input_size, mem_size, num_layers)
	test_gru_unit = dmn.rnn_units.gf_gru_unit(input_size, mem_size, num_layers)

	graph.dot(test_lstm_unit.fg, 'MLP', 'myMLP')
	
	local input = torch.rand(input_size)

	local h_prev
	local c_prev

	if num_layers == 1 then 
		h_prev = torch.rand(mem_size)
		c_prev = torch.rand(mem_size)
	else
		h_prev = {}
		c_prev = {}
		for i = 1, num_layers do
			table.insert(h_prev, torch.rand(mem_size))
			table.insert(c_prev, torch.rand(mem_size))
		end
	end
	local gru_res = test_gru_unit:forward({input, h_prev})
	local lstm_res = test_lstm_unit:forward({input, c_prev, h_prev})
	
	print("LSTM RESULT")
	print(lstm_res)
	print("GRU RESULT")
	print(gru_res)
end

-- try one layer
test_unit(50, 10, 1)

-- try two layers
test_unit(50, 10, 2)

-- try 10 layers
test_unit(50, 10, 5)


