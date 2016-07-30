local rnn_units = torch.class('dmn.rnn_units')

-- Creates a fine-grained, coarse, coarse_fixed (mean pooling) or dropout attention lstm unit. 
-- It's an rnn where c_i, si = g(s_i-1, ci-1, yi, att_i).
-- input_size: size of input vecotrs
-- context_size: size of context vectors
-- rnn_size: size of memory of lstm
-- num_layers: number of layers of lstm
function rnn_units.attention_lstm_unit(input_size, context_size, rnn_size, num_layers, attention_type)
  assert(input_size ~= nil, "Must specify input size to use")
  assert(context_size ~= nil, "Must specify context size to use")
  assert(rnn_size ~= nil, "Must specify rnn_size to use")
  assert(num_layers ~= nil, "Must specify number of layers to use")
  assert(attention_type ~= nil, "Must specify attention type to use [coarse, fine, dropout]")

  local input = nn.Identity()()

  local ctable_p = nn.Identity()()
  local htable_p = nn.Identity()()
  
  local context = nn.Identity()()
  local first_h_layer = (num_layers == 1) and htable_p or nn.SelectTable(1)(htable_p)

  -- Attention is softmax(e_ij) where e_ij = va^T * tanh(Wa*S_i-1 + U_a * h_j)
  local perceptroned_context = nn.Linear(context_size, rnn_size)(context)
  local added_context = nn.Tanh()(dmn.CRowAddTable(){perceptroned_context, nn.Linear(rnn_size, rnn_size)(first_h_layer)})

  local soft_attention, summed_context, coarse_attention, weights

  if attention_type == 'fine' then 
    soft_attention = nn.SoftMax()(nn.Linear(rnn_size, context_size)(added_context))
  elseif attention_type == 'coarse' then
    weights = nn.Linear(rnn_size, 1)(added_context)
    coarse_attention = nn.SoftMax()(weights)
    local replicated_attention = nn.Replicate(context_size, 2)(coarse_attention)
    soft_attention = dmn.Squeeze()(replicated_attention)
  elseif attention_type == 'coarse_fixed' then 
    weights = dmn.Squeeze()(nn.Linear(rnn_size, 1)(added_context))
    coarse_attention = nn.SoftMax()(weights)
    local replicated_attention = nn.Replicate(context_size, 2)(coarse_attention)
    soft_attention = replicated_attention
  elseif attention_type == 'dropout' then
    soft_attention = nn.SoftMax()(nn.Dropout(0.3, false)(nn.Linear(rnn_size, context_size)(added_context)))
  else 
    error("Invalid attention type given")
  end

  summed_context = nn.Sum()(nn.CMulTable(){soft_attention, context})

  -- multilayer LSTM
  local htable, ctable = {}, {}
  for layer = 1, num_layers do
    local h_p = (num_layers == 1) and htable_p or nn.SelectTable(layer)(htable_p)
    local c_p = (num_layers == 1) and ctable_p or nn.SelectTable(layer)(ctable_p)

    local new_gate = function()
       -- only apply attention on first layer
      if layer == 1 then 
        return nn.CAddTable(){
          nn.Linear(input_size, 4 * rnn_size)(input),
          nn.Linear(rnn_size, 4 * rnn_size)(h_p),
          nn.Linear(context_size, 4 * rnn_size)(summed_context)
        }
      else 
        return nn.CAddTable(){
          nn.Linear(rnn_size, 4 * rnn_size)(htable[layer - 1]),
          nn.Linear(rnn_size, 4 * rnn_size)(h_p)
        }
      end
    end

    local all_input_sums = new_gate()
    local sigmoid_chunk = nn.Narrow(1, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(1, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(1, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(1, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
 
    local in_transform = nn.Narrow(1, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
 
    ctable[layer] = nn.CAddTable()({
      nn.CMulTable()({forget_gate, c_p}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    htable[layer] = nn.CMulTable()({out_gate, nn.Tanh()(ctable[layer])})
  end

  -- if LSTM is single-layered, this makes htable/ctable Tensors (instead of tables).
  -- this avoids some quirks with nngraph involving tables of size 1.
  htable, ctable = nn.Identity()(htable), nn.Identity()(ctable)

  -- Return soft attention for visualization
  local cell = nn.gModule({input, context, ctable_p, htable_p}, {ctable, htable, soft_attention})
  return cell
end


--[[ 
Gated-feedback LSTM implementation
--]]
function rnn_units.gf_lstm(input_size, mem_dim, num_layers)
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

  return cell
end