require('..')

-- testing basic attention unit
tmp = dmn.attention_units.attention_unit(5, 2)
inputs = torch.rand(10, 5)
print(inputs)
results = tmp:forward(inputs)

print(results)

-- testing basic gru unit
local gru_unit = dmn.rnn_units.gru_unit(5, 3, 2)
local res = gru_unit:forward({torch.rand(5), {torch.rand(3), torch.rand(3)}})

-- testing basic lstm attention unit
local in_dim = 5
local mem_dim = 7
local num_layers = 2
local attention_lstm_unit = dmn.rnn_units.attention_lstm_unit(in_dim, mem_dim, mem_dim, num_layers, 'coarse_fixed')

local input = torch.rand(in_dim)
local context = torch.rand(10, mem_dim)
local context_2 = torch.zeros(15, mem_dim)
local prev_hidden_state = {torch.rand(mem_dim), torch.rand(mem_dim)}
local prev_cell_state = {torch.rand(mem_dim), torch.rand(mem_dim)}

local res = attention_lstm_unit:forward({input, context, prev_cell_state, prev_hidden_state})

local res1 = attention_lstm_unit:forward({input, context, prev_cell_state, prev_hidden_state})
local errs = attention_lstm_unit:backward({input, context, prev_cell_state, prev_hidden_state}, res1)

local img = dmn.image_functions.visualize_tensor(res1[3], 25)
image.save("tmp.jpg", img)

print("Forwrd backward DONE")
for i,node in ipairs(attention_lstm_unit.forwardnodes) do
	  local gmnode = attention_lstm_unit.forwardnodes[i]
	  assert(gmnode, 'trying to map another gModule with a different structure')
	  if node.data.annotations._debugLabel == '[.../NLP/DeepLearning/softmax/dmn/models/rnn_units/units.lua]:19' then
	  end
end


-- testing gradient of attention network
local lstm_unit = dmn.Attention_LSTM_Decoder{
	in_dim = 5,
	context_dim = 2,
	mem_dim = 2,
	num_layers = 1,
	gpu_mode = false,
	attention_type = 'coarse'
}
lstm_unit:disable_dropouts()
lstm_unit:grad_check()



