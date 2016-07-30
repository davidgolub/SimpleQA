--[[
Class to factor out common code from all of the lstms/rnns/grus etc. This was becoming repetitive
]]
local rnn_utils = torch.class('dmn.rnn_utils')

-- Initializes values from lstm
function rnn_utils.init_values(lstm)
  assert(lstm ~= nil, "Must specify lstm to init values for")
  lstm.tensor_type = lstm.gpu_mode and torch.CudaTensor or torch.DoubleTensor
  lstm.depth = 0
  lstm.cells = {}  -- table of cells in a roll-out
  lstm.tensors = {}  -- table of tensors for faster lookup
  lstm.back_tensors = {} -- table of tensors for backprop

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local ctable_init, ctable_grad, htable_init, htable_grad
  if lstm.num_layers == 1 then
    ctable_init = lstm.tensor_type(lstm.mem_dim):zero()
    htable_init = lstm.tensor_type(lstm.mem_dim):zero()
    ctable_grad = lstm.tensor_type(lstm.mem_dim):zero()
    htable_grad = lstm.tensor_type(lstm.mem_dim):zero()
  else
    ctable_init, ctable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, lstm.num_layers do
      ctable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
      htable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
      ctable_grad[i] = lstm.tensor_type(lstm.mem_dim):zero()
      htable_grad[i] = lstm.tensor_type(lstm.mem_dim):zero()
    end
  end

  lstm.dummy_values = {dmn.functions.deepcopy(ctable_init), dmn.functions.deepcopy(htable_init)}
  lstm.initial_values = {ctable_init, htable_init}
  lstm.gradInput = {
    lstm.tensor_type(lstm.in_dim):zero(), -- grad with respect to input
    ctable_grad,
    htable_grad, -- grad with respect to hidden state of lstm
  }


  -- precreate outputs for faster performance
  for i = 1, 100 do
    lstm.tensors[i] = lstm.tensor_type(i, lstm.mem_dim):zero()
    lstm.back_tensors[i] = lstm.tensor_type(i, lstm.in_dim):zero()
  end
end

-- Initializes values from lstm
function rnn_utils.init_values_attention(lstm)
  assert(lstm ~= nil, "Must specify lstm to init values for")
  lstm.tensor_type = lstm.config.gpu_mode and torch.CudaTensor or torch.DoubleTensor
  lstm.depth = 0
  lstm.cells = {}  -- table of cells in a roll-out
  lstm.tensors = {}  -- table of tensors for faster lookup
  lstm.back_tensors = {} -- table of tensors for backprop

  -- initial (t = 0) states for forward propagation and initial error signals
  -- for backpropagation
  local ctable_init, ctable_grad, htable_init, htable_grad
  if lstm.num_layers == 1 then
    ctable_init = lstm.tensor_type(lstm.mem_dim):zero()
    htable_init = lstm.tensor_type(lstm.mem_dim):zero()
    ctable_grad = lstm.tensor_type(lstm.mem_dim):zero()
    htable_grad = lstm.tensor_type(lstm.mem_dim):zero()
  else
    ctable_init, ctable_grad, htable_init, htable_grad = {}, {}, {}, {}
    for i = 1, lstm.num_layers do
      ctable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
      htable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
      ctable_grad[i] = lstm.tensor_type(lstm.mem_dim):zero()
      htable_grad[i] = lstm.tensor_type(lstm.mem_dim):zero()
    end
  end

  lstm.dummy_values = {dmn.functions.deepcopy(ctable_init), dmn.functions.deepcopy(htable_init)}
  lstm.initial_values = {ctable_init, htable_init}
  lstm.gradInput = {
    lstm.tensor_type(lstm.in_dim):zero(), -- grad with respect to input
    lstm.tensor_type(lstm.in_dim):zero(), -- dummy
    ctable_grad, -- grad with respect to cell state of lstm
    htable_grad, -- grad with respect to hidden state of lstm
  }


  -- precreate outputs for faster performance
  for i = 1, 100 do
    lstm.tensors[i] = lstm.tensor_type(i, lstm.mem_dim):zero()
    lstm.back_tensors[i] = lstm.tensor_type(i, lstm.in_dim):zero()
  end
end

function rnn_utils.new_initial_values(lstm)
  assert(lstm ~= nil, "Must specify lstm to init values for")
  local ctable_init, htable_init
  if lstm.num_layers == 1 then
    ctable_init = lstm.tensor_type(lstm.mem_dim):zero()
    htable_init = lstm.tensor_type(lstm.mem_dim):zero()
  else
    htable_init, ctable_init = {}, {}
    for i = 1, lstm.num_layers do
      ctable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
      htable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
    end
  end

  return {ctable_init, htable_init}
end

function rnn_utils.new_initial_values_gru(lstm)
  assert(lstm ~= nil, "Must specify lstm to init values for")
  local ctable_init, htable_init
  if lstm.num_layers == 1 then
      htable_init = lstm.tensor_type(lstm.mem_dim):zero()
  else
    htable_init, ctable_init = {}
    for i = 1, lstm.num_layers do
      htable_init[i] = lstm.tensor_type(lstm.mem_dim):zero()
    end
  end

  return htable_init
end