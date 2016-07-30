--[[
 Long Short-Term Memory that returns hidden states on ALL input, not just final state.
 You're responsible for putting in input to hidden states
--]]

local LSTM, parent = torch.class('dmn.LSTM_Decoder', 'nn.Module')

function LSTM:__init(config)
  parent.__init(self)

  assert(config.in_dim ~= nil, "Input dim to lstm must be specified")
  assert(config.mem_dim ~= nil, "Memory dim to lstm must be specified")
  assert(config.num_layers ~= nil, "Number of layers to lstm must be specified")
  assert(config.gpu_mode ~= nil, "Gpu mode of lstm must be specified")
  assert(config.lstm_type ~= nil, "Must specify lstm type")

  self.config = config
  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim
  self.num_layers = config.num_layers
  self.gate_output = config.gate_output or true
  self.gpu_mode = config.gpu_mode
  self.lstm_type = config.lstm_type

  self.master_cell = self:new_cell()
  self:init_values() 
end

function LSTM:init_values()
  dmn.rnn_utils.init_values(self)
end

function LSTM:new_initial_values()
  return dmn.rnn_utils.new_initial_values(self)
end

-- Returns cell type
function LSTM:cell_type()
  local cell_type
  if self.lstm_type == 'gf_lstm' then 
    cell_type = dmn.rnn_units.gf_lstm 
  elseif self.lstm_type == 'fast_lstm' then
    cell_type = dmn.rnn_units.fast_lstm
  elseif self.lstm_type == 'lstm' then
    cell_type = dmn.rnn_units.old_lstm 
  else
    error("Unknown cell type to use")
  end
  return cell_type
end
-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function LSTM:new_cell()
 local cell_type = self:cell_type()
 local cell = cell_type(self.in_dim, self.mem_dim, self.num_layers)
 if self.gpu_mode then
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
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns T x mem_dim tensor, all the intermediate hidden states of the LSTM. If multilayered,
-- returns output only of last memory layer
function LSTM:forward(inputs, hidden_inputs, reverse)
  assert(inputs ~= nil, "Must specify inputs into lstm decoder")
  assert(hidden_inputs ~= nil, "Must specify hidden inputs into lstm decoder")
  assert(reverse ~= nil, "Must specify whether to feed inputs in reverse or not")

  local size = inputs:size(1)
  if inputs:dim() == 3 then 
    -- batch forwarding
    self.output = self.tensor_type(inputs:size(1), inputs:size(2), self.mem_dim)
  else
    self.output = self.tensors[size]
  end
  if self.output == nil then
      dmn.logger:print("Creating new input tensors")
    if self.gpu_mode then
      self.tensors[size] = torch.FloatTensor(size, self.mem_dim):zero():cuda()
    else
      self.tensors[size] = torch.DoubleTensor(size, self.mem_dim):zero()
    end
    self.output = self.tensors[size]
  end

  for t = 1, size do
    if self.depth > 100 then 
      dmn.logger:print("Warning: LSTM Decoder depth is getting beyond 100, " .. self.depth)
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

    local cell_inputs = {input, prev_output[1], prev_output[2]}
    --print("ON time " .. t)
    --print(self.depth)
    --print(hidden_inputs)
    --print(cell_inputs)
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
function LSTM:tick(input, prev_outputs)
  assert(input ~= nil)
  assert(prev_outputs ~= nil)

  --local in_val = {input, copied_prev_outputs[1], copied_prev_outputs[2]}
  local cell = self.master_cell
  --if cell == nil then
  --    dmn.logger:print("Cells are null at depth ", self.depth)
  --    cell = self:new_cell()
  --    self.cells[self.depth] = cell
  -- end

  -- self.depth = self.depth + 1

  local in_val = {input, prev_outputs[1], prev_outputs[2]}

  local outputs = cell:forward(in_val)
  return outputs
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function LSTM:backward(inputs, hidden_inputs, reverse, grad_outputs)
  assert(inputs ~= nil, "Must specify inputs into lstm decoder")
  assert(hidden_inputs ~= nil, "Must specify hidden inputs into lstm decoder")
  assert(reverse ~= nil, "Must specify whether to feed inputs in reverse or not")
  assert(grad_outputs ~= nil)

  local size = inputs:size(1)
  if self.depth == 0 then
    error("No cells to backpropagate through")
  end

  local input_grads = (inputs:dim() == 3) and self.tensor_type(inputs:size()) or self.back_tensors[size]
  local new_inputs = dmn.math_functions.copy_lstm_units(hidden_inputs, self.num_layers)

  local in_grad = (inputs:dim() == 3) and self.tensor_type(inputs:size(2), inputs:size(3)):zero()
                  or self.tensor_type(inputs:size(2)):zero()
  self.gradInput = {
      in_grad, -- grad with respect to input
      new_inputs[1],
      new_inputs[2] -- grad with respect to hidden state of lstm
  }

  if input_grads == nil then
    dmn.logger:print("Creating new input grads")
    if self.gpu_mode then
      self.back_tensors[size] = torch.FloatTensor(inputs:size()):cuda()
    else
      self.back_tensors[size] = torch.DoubleTensor(inputs:size())
    end
    input_grads = self.back_tensors[size]
  end

  for t = size, 1, -1 do
    local input = reverse and inputs[size - t + 1] or inputs[t]
    local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
    local cell = self.cells[self.depth]
    local grads = {self.gradInput[2], self.gradInput[3]}
    if self.num_layers == 1 then
      grads[2]:add(grad_output)
    else
      grads[2][self.num_layers]:add(grad_output)
    end

    local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                                         or hidden_inputs                        
    self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
    if reverse then
      input_grads[size - t + 1] = self.gradInput[1]
    else
      input_grads[t] = self.gradInput[1]
    end
    if self.depth == 1 then
      if self.num_layers == 1 then
        self.initial_values = {self.gradInput[2]:clone(), 
        self.gradInput[3]:clone()}
      else 
        for i = 1, self.num_layers do 
          self.initial_values = {{}, {}}
          table.insert(self.initial_values[1], self.gradInput[2][i]:clone())
          table.insert(self.initial_values[2], self.gradInput[3][i]:clone())
        end
      end
    end
    self.depth = self.depth - 1
  end
  self:forget() -- important to clear out state
  return input_grads, self.initial_values
end

function LSTM:share(lstm, ...)
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.num_layers ~= lstm.num_layers then error("LSTM layer count mismatch") end
  if self.gate_output ~= lstm.gate_output then error("LSTM output gating mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
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

function LSTM:enable_dropouts()
  dmn.logger:print("No dropouts to enable for lstm encoder")
end

function LSTM:disable_dropouts()
  dmn.logger:print("No dropouts to disable for lstm decoder")
end

function LSTM:set_gpu_mode()
  dmn.logger:print("Setting gpu mode for LSTM decoder")
  self.gpu_mode = true
  self.master_cell:cuda()
  self:init_values()
end

function LSTM:set_cpu_mode()
  dmn.logger:print("Setting cpu mode for LSTM decoder")
  self.gpu_mode = false
  self.master_cell:double()
  self:init_values()
end

function LSTM:reset_depth()
  self.depth = 0
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

function LSTM:print_config()
  dmn.logger:printf('%-25s = %d\n', 'input dimension', self.config.in_dim)
  dmn.logger:printf('%-25s = %d\n', 'memory dimension', self.config.mem_dim)
  dmn.logger:printf('%-25s = %d\n', 'number of layers', self.config.num_layers)
  dmn.logger:printf('%-25s = %s\n', 'gpu_mode', self.config.gpu_mode)
end



