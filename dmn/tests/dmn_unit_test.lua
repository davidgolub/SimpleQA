require('..')

-- Testing new dmn unit
local input_size = 100
local gate_size = 50
local mem_size = 100
test_unit = dmn.rnn_units.dmn_unit_new(input_size, mem_size, gate_size)

local input = torch.rand(mem_size)
local h_prev = torch.rand(mem_size)
local mem = torch.rand(mem_size)
local question = torch.rand(mem_size)

res = test_unit:forward({input, mem, question})
print(res)

-- Testing previous attention units
local input_size = 100
local gate_size = 50
local mem_size = 100
test_unit = dmn.rnn_units.dmn_unit(input_size, mem_size, gate_size)

local input = torch.rand(mem_size)
local h_prev = torch.rand(mem_size)
local mem = torch.rand(mem_size)
local question = torch.rand(mem_size)

res = test_unit:forward({input, h_prev, mem, question})

print(res)
local err = test_unit:backward({input, h_prev, mem, question}, res)
print("PRINTING ERROR")
print(err)
memory_module = dmn.EpisodicMemory{  
  				mem_dim = 100,
 				num_episodes = 10,
  				gpu_mode = false,
  				gate_size = 50
			  }

-- Fact candidates
local mem_state = torch.rand(100)
local inputs = torch.rand(5, 100)
local question_state = torch.rand(100)
local reverse = false

local memory = memory_module:forward(inputs, mem_state, question_state, reverse)

local input_err, mem_err, question_err = 
memory_module:backward(inputs, mem_state, question_state, reverse, memory)