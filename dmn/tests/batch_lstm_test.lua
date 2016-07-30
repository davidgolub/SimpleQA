--[[
	Batch forwarding LSTMs test
]]

require('..')

local config = {
	in_dim = 100,
	mem_dim = 300,
	num_layers = 1,
	gpu_mode = false,
	dropout = false,
	lstm_type = 'lstm',
	gru_type = 'gru'
}

local test_network = function(network, network_type)
	local hidden_batch_inputs = network_type == 'lstm' and {torch.rand(50, 300), torch.rand(50, 300)} 
														or torch.rand(50, 300)
	local hidden_single_inputs = network_type == 'lstm' and {torch.rand(300), torch.rand(300)}
														or torch.rand(300)

	-- batch of length 200, 50 x 100 inputs.

	local t1 = sys.clock()
	results = network:forward(torch.rand(200, 2, 100), hidden_batch_inputs, false)
	results1 = network:backward(torch.rand(200, 2, 100), hidden_batch_inputs, false, results)
	local t2 = sys.clock()

	--print(results:size())
	print((t2 - t1) / 100)

	local t1 = sys.clock()
	results_single = network:forward(torch.rand(200, 100), hidden_single_inputs, false)
	results1_single = network:backward(torch.rand(200, 100), hidden_single_inputs, false, results)


end

print("Testing lstms")
-- lstm
local lstm_encoder_network = dmn.LSTM_Encoder(config)
local lstm_decoder_network = dmn.LSTM_Decoder(config)

print("Testing lstm encoder with regular lstm")
test_network(lstm_encoder_network, 'lstm')

print("Testing lstm decoder with regular lstm")
test_network(lstm_decoder_network, 'lstm')

-- gf_lstm
config.lstm_type = 'gf_lstm'

local lstm_decoder_network = dmn.LSTM_Decoder(config)
local lstm_encoder_network = dmn.LSTM_Encoder(config)

print("Testing encoder LSTM network with gated-feedback lstms")
--test_network(lstm_encoder_network, 'lstm')

print("Testing decoder LSTM network with gated-feedback lstms")
--test_network(lstm_decoder_network, 'lstm')

print("Testing grus")
-- gru
local gru_decoder_network = dmn.GRU_Decoder(config)
local gru_encoder_network = dmn.GRU_Encoder(config)

print("Testing encoder GRU network with regular gru")
--test_network(gru_encoder_network, 'gru')

print("Testing decoder GRU network with regular gru")
--test_network(gru_decoder_network, 'gru')

config.gru_type = 'gf_gru'

-- gru
local gru_decoder_network = dmn.GRU_Decoder(config)
local gru_encoder_network = dmn.GRU_Encoder(config)

print("Testing encoder GRU network with gated-feedback gru")
test_network(gru_encoder_network, 'gru')

print("Testing decoder GRU network with gated-feedback gru")
test_network(gru_decoder_network, 'gru')
