--[[
	Batch forwarding LSTMs test
]]

require('..')


 num_layers = 1
 input_size = 100
 mem_dim = 100
  local input = nn.Identity()()
  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer LSTM
  local htable, ctable = {}, {}
  for layer = 1, num_layers do
    local h_p = (num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)
    local c_p = (num_layers == 1) and ctable_p or nn.SelectTable(layer)(ctable_p)

    local new_gate = function()
      local in_module = (layer == 1)
        and nn.Linear(input_size, mem_dim)(input)
        or  nn.Linear(mem_dim, mem_dim)(htable[layer - 1])
      return nn.CAddTable(){
        in_module,
        nn.Linear(mem_dim , mem_dim)(h_p)
      }
    end

    local gf_gate = function()
      -- get input module
      -- U_i->j*h^i_t-1
      local gf_layer = {}
      local concatenated_features = (num_layers == 1) and htable_p or nn.JoinTable(1){htable_p}
      for j = 1, num_layers do
        local in_module = (layer == 1)
        and nn.Linear(input_size, mem_dim)(input)
        or  nn.Linear(mem_dim, mem_dim)(htable[layer - 1])

        local hidden_concat = nn.Linear(num_layers * mem_dim, 1)(concatenated_features)
        local reset_gate = nn.Sigmoid()(nn.CAddTable(){hidden_concat, nn.Linear(input_size, 1)(input)})
        
        -- replicate reset gate
        local replicated_reset_gate = dmn.Squeeze()(nn.Replicate(mem_dim)(reset_gate))

        -- g_i->j * U_i->j * h^i_t-1
        local curr_sum = nn.CMulTable(){replicated_reset_gate, in_module}

        -- quirks for single-layered things
        if num_layers > 1 then 
          table.insert(gf_layer, curr_sum)
        else
          gf_layer = curr_sum
        end
      end

      -- quirks for single-layered things
      local summed_gf = (num_layers == 1) and nn.Identity()(gf_layer) or nn.CAddTable()(gf_layer)
      return summed_gf
    end
    
    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())

    -- gated feedback update
    local update = nn.Tanh()(gf_gate())

    -- update the state of the LSTM cell
    ctable[layer] = nn.CAddTable(){
      nn.CMulTable(){f, c_p},
      nn.CMulTable(){i, update}
    }

    -- output gate
    local o = nn.Sigmoid()(new_gate())
    htable[layer] = nn.CMulTable(){o, nn.Tanh()(ctable[layer])}
  end

  -- if LSTM is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)
  local cell = nn.gModule({input, ctable_p, htable_p}, {ctable, htable})

inputs = torch.rand(3, 100)
local new_inputs = {inputs, torch.rand(3, 100), torch.rand(3, 100)}
results = cell:forward(new_inputs)
print(results)
