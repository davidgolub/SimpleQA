--[[
 Long Short-Term Memory that returns hidden states on ALL input, not just final state.
 You're responsible for putting in input to hidden states. Uses attention + context
--]]

local LSTM, parent = torch.class('dmn.Attention_LSTM_Decoder', 'nn.Module')

function LSTM:__init(config)
  parent.__init(self)

  assert(config.in_dim ~= nil, "Input dim to attention lstm must be specified")
  assert(config.context_dim ~= nil, "Context dim to attention lstm must be specified")
  assert(config.mem_dim ~= nil, "Memory dim to attention lstm must be specified")
  assert(config.num_layers ~= nil, "Number of layers to attention lstm must be specified")
  assert(config.gpu_mode ~= nil, "Gpu mode of attention lstm must be specified")
  assert(config.attention_type ~= nil, "Must specify attention type")
  
  self.config = dmn.functions.deepcopy(config)
  self.in_dim = config.in_dim
  self.context_dim = config.context_dim
  self.mem_dim = config.mem_dim
  self.num_layers = config.num_layers
  self.gate_output = true
  
  self.master_cell = self:new_cell()
  self:init_values()
end

function LSTM:init_values()
  dmn.rnn_utils.init_values_attention(self)
end

function LSTM:new_initial_values()
  return dmn.rnn_utils.new_initial_values(self)
end

-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function LSTM:new_cell()
 local cell = dmn.rnn_units.attention_lstm_unit(self.config.in_dim, 
    self.config.context_dim,
    self.config.mem_dim,
    self.config.num_layers,
    self.config.attention_type)
 if self.config.gpu_mode then
    cell:cuda()
 end
  
 -- share parameters
 if self.master_cell then
    share_params(cell, self.master_cell, 'weight', 'bias', 'gradWeight', 'gradBias')
 end
 return cell
end


-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- context: A T_1 x context_dim tensor of context vectors, where T is the number of time steps
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns T x mem_dim tensor, all the intermediate hidden states of the LSTM. If multilayered,
-- returns output only of last memory layer
function LSTM:forward(inputs, context, hidden_inputs, reverse)
  assert(inputs ~= nil, "Inputs must not be null")
  assert(context ~= nil, "Context must not be null")
  assert(hidden_inputs ~= nil, "Hidden inputs must not be null")
  assert(reverse ~= nil, "Must specify whether inputs should be in reverse or not")

  local size = inputs:size(1)
  self.output = self.tensors[size]
  if self.output == nil then
    self.tensors[size] = self.tensor_type(size, self.mem_dim):zero()
    self.output = self.tensors[size]
  end

  for t = 1, size do
    if self.depth > 100 then 
      dmn.logger:print("Warning: attention depth is getting beyond 100, " .. self.depth)
    end
    local input = reverse and inputs[size - t + 1] or inputs[t]
    self.depth = self.depth + 1
    local cell = self.cells[self.depth]

    if cell == nil then
      --print("Cells are null at depth ", self.depth)
      cell = self:new_cell()
      self.cells[self.depth] = cell
    end

    local prev_output
    if self.depth > 1 then
      prev_output = self.cells[self.depth - 1].output
    else
      prev_output = hidden_inputs
    end

    local cell_inputs = {input, context, prev_output[1], prev_output[2]}
    local outputs = cell:forward(cell_inputs)
    local htable = outputs[2]
    if self.num_layers == 1 then
      self.output[t] = htable
    else
      self.output[t] = htable[self.num_layers]
    end
  end
  return self.output
end

-- Does a single tick of lstm layer, used in beam search
-- input: in_dim tensor, in_dim is input to the LSTM.
-- prev_states: previous states of the lstm (cell_state, hidden array)
-- Returns cell_state, hidden_state of LSTM, both mem_dim tensors
function LSTM:tick(input, context, prev_outputs)
  assert(input ~= nil, "Input must not be null")
  assert(context ~= nil, "Context must be not null")
  assert(prev_outputs ~= nil, "Previous outputs must not be null")

  --local in_val = {input, copied_prev_outputs[1], copied_prev_outputs[2]}
  local in_val = {input, context, prev_outputs[1], prev_outputs[2]}
  local cell = self.master_cell

  local outputs = cell:forward(in_val)
  return outputs
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs, contexts, and initial state of lstm
function LSTM:backward(inputs, context, hidden_inputs, reverse, grad_outputs)
  assert(inputs ~= nil, "Must specify inputs to lstm")
  assert(context ~= nil, "Must specify context into lstm")
  assert(hidden_inputs ~= nil, "Must specify hidden inputs into lstm")
  assert(reverse ~= nil, "Must specify whether to feed lstm in reverse or not")
  assert(grad_outputs ~= nil, "Must specify grad outputs")

  local context_size = context:size(1)
  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end
  
  local input_grads = self.back_tensors[size]
  local context_grads = self.tensor_type(context_size, self.context_dim):zero()

  if input_grads == nil then
    self.back_tensors[size] = self.tensor_type(inputs:size()):zero()
    input_grads = self.back_tensors[size]
  end

  -- dummy grads for probability
  local prob_grads = self.tensor_type(context_size, self.config.context_dim):zero()
  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]

    local grads = {self.gradInput[3], 
                   self.gradInput[4], 
                   prob_grads} 

    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      grads[2][self.num_layers]:add(grad_output)
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or hidden_inputs


    self.gradInput = cell:backward({input, context, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end

    -- add gradient with respec to context to grad input
    context_grads:add(self.gradInput[2])

    if self.depth == 1 then
      if self.num_layers == 1 then
        self.initial_values[1]:copy(self.gradInput[3])
        self.initial_values[2]:copy(self.gradInput[4])
      else 
        for i = 1, self.num_layers do 
          self.initial_values[1][i]:copy(self.gradInput[3][i])
          self.initial_values[2][i]:copy(self.gradInput[4][i])
        end
      end
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads, context_grads, self.initial_values
end

function LSTM:share(lstm, ...)
  assert(lstm ~= nil, "Must include lstm or not")
  
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:grad_check()
  self.params, self.grad_params = self.master_cell:getParameters()
  local input_indices = torch.rand(5, self.in_dim)
  local context_indices = torch.rand(6, self.context_dim)
  local desired_state = torch.rand(5, self.mem_dim)
  local hidden_state = {torch.zeros(self.mem_dim), torch.zeros(self.mem_dim)}
  local criterion = nn.MSECriterion()

  local currIndex = 0
  local feval = function(x)
      self.grad_params:zero()
      local lstm_output = self:forward(input_indices, context_indices, hidden_state, false)
      local loss = criterion:forward(lstm_output, desired_state)
      local errors = criterion:backward(lstm_output, desired_state)
      self:backward(input_indices, context_indices, hidden_state, false, errors)
      currIndex = currIndex + 1
      print(loss)
      print(currIndex, " of ", self.params:size(1))    
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    print("Gradient error for attention network is")
    print(diff)
    assert(diff < 1e-5, "Gradient is greater than tolerance")
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:getModules()
  return {self.master_cell}
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

-- Clear saved gradients
function LSTM:forget()
  self.depth = 0
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end

function LSTM:enable_dropouts()
  dmn.logger:print("Enabling dropouts for attention lstm encoder")
  self.master_cell:training()
  for i = 1, #self.cells do 
    self.cells[i]:training()
  end
end

function LSTM:disable_dropouts()
  dmn.logger:print("Disabling dropouts for attention lstm decoder")
  self.master_cell:evaluate()
  for i = 1, #self.cells do 
    self.cells[i]:evaluate()
  end
end

function LSTM:set_gpu_mode()
  dmn.logger:print("Setting gpu mode for LSTM encoder")
  self.master_cell:cuda()
  self.config.gpu_mode = true
  self:init_values()
end

function LSTM:set_cpu_mode()
  dmn.logger:print("Setting cpu mode for LSTM encoder")
  self.master_cell:double()
  self.config.gpu_mode = false
  self:init_values()
end

function LSTM:print_config()
  printf('%-25s = %d\n', 'input dimension', self.config.in_dim)
  printf('%-25s = %d\n', 'memory dimension', self.config.mem_dim)
  printf('%-25s = %d\n', 'context dimension', self.config.context_dim)
  printf('%-25s = %d\n', 'number of layers', self.config.num_layers)
  printf('%-25s = %s\n', 'gpu_mode', self.config.gpu_mode)
end

