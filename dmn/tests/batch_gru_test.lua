--[[
	Batch forwarding LSTMs test
]]

require('..')

local config = {
	in_dim = 100,
	mem_dim = 100,
	num_layers = 1,
	gpu_mode = false,
	dropout = false,
	gru_type = 'gru'
}

local network = dmn.GRU_Decoder(config)

local t1 = sys.clock()
results = network:forward(torch.rand(200, 100, 100), torch.rand(100, 100), false)
results1 = network:backward(torch.rand(200, 100, 100), torch.rand(100, 100), false, results)
local t2 = sys.clock()

print((t2 - t1) / 100)

results = network:forward(torch.rand(5, 100), torch.rand(100), false)
results1 = network:forward(torch.rand(5, 100), torch.rand(100), false, results1)