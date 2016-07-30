 require('..')
 num_layers = 1
 in_dim = 100
 mem_dim = 300
 local input = nn.Identity()()
  local htable_p = nn.Identity()()

  -- multilayer GRU
  local htable = {}
  for layer = 1, num_layers do
    -- get current inputs for the layer
    local curr_input_size = (layer == 1) and in_dim or mem_dim
    -- assert(false)
    local curr_input = (layer == 1) and input or nn.Identity()(htable[layer - 1])

    -- get previous hidden state for layers
    local h_p = (num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)

    local new_gate = function()
      local i2h = (layer == 1)
        and nn.Linear(in_dim, mem_dim)(input)
        or  nn.Linear(mem_dim, mem_dim)(htable[layer - 1])
      local h2h = nn.Linear(mem_dim, mem_dim)(h_p)
      return nn.CAddTable()({i2h, h2h})
    end

    local gf_gate = function()
      -- get input module
      -- U_i->j*h^i_t-1
      local gf_layer = {}
      local concatenated_features = (num_layers == 1) and htable_p or nn.JoinTable(1){htable_p}
      for j = 1, num_layers do
        local gf_input_size = (layer == 1) and in_dim or mem_dim
        local in_module = (layer == 1)
        and nn.Linear(in_dim, mem_dim)(input)
        or  nn.Linear(mem_dim, mem_dim)(htable[layer - 1])

        local hidden_concat = nn.Linear(num_layers * mem_dim, 1)(concatenated_features)
        local reset_gate = nn.Sigmoid()(nn.CAddTable(){hidden_concat, nn.Linear(in_dim, 1)(input)})
        
        -- replicate reset gate.
        local replicated_reset_gate = nn.Replicate(mem_dim)(reset_gate)

        -- now wrong dimensions, so transpose
        local transposed_replicated_reset_gate = dmn.Squeeze()(nn.Transpose({2, 1})(replicated_reset_gate))

        -- g_i->j * U_i->j * h^i_t-1
        local curr_sum = nn.CMulTable(){transposed_replicated_reset_gate, in_module}

        -- quirks for single-layered things
        if num_layers > 1 then 
          table.insert(gf_layer, curr_sum)
        else
          gf_layer = curr_sum
        end
      end

      -- quirks for single-layered things
      local summed_gf = (num_layers == 1) and gf_layer or nn.CAddTable()(gf_layer)
      return summed_gf
    end

    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(gf_gate())
    local reset_gate = nn.Sigmoid()(new_gate())
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, h_p})
    local p2 = nn.Linear(mem_dim, mem_dim)(gated_hidden)
    local p1 = nn.Linear(curr_input_size, mem_dim)(curr_input)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), h_p})
    local next_h = nn.CAddTable()({zh, zhm1})
    htable[layer] = update_gate
  end

  -- if GRU is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable = nn.Identity()(htable)
  local cell = nn.gModule({input, htable_p}, {htable})

 local res = cell:forward({torch.rand(50, 100), torch.rand(50, 300)})

 print(res)