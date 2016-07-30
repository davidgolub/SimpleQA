--[[

  A QuestionModule takes two things as input: 
  1) a lookup-table for word embeddings
  1) an LSTM cell network for encoding
  
  It first encodes word indices to embeddings and then 
  It encodes the word embeddings into the memory states of the lstm.
  Returns final memory state of lstm at the end

--]]

local QuestionModule = torch.class('dmn.QuestionModule')

function QuestionModule:__init(config)
  -- parameters for lstm cell
  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.emb_dim ~= nil, "Must specify embed dimensions")
  assert(config.num_classes ~= nil, "Must specify number of classes")
  assert(config.dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.dropout ~= nil, "Must specify dropout")
  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.mem_dim ~= nil, "Must specify memory dimension of lstm")
  assert(config.num_layers ~= nil, "Must specify number of layers to lstm")
  assert(config.hashing ~= nil, "Must specify whether to hash word tokens or not")

  self.config = config
  self.hashing = config.hashing
  self.gpu_mode = config.gpu_mode
  self.mem_dim = config.mem_dim
  self.num_layers = config.num_layers
  self.reverse = false;

  local embed_type = self.hashing and dmn.SparseHashLayer or dmn.EmbedLayer
  self.embed_layer = embed_type{
                      gpu_mode = config.gpu_mode,
                      emb_dim = config.emb_dim,
                      num_classes = config.num_classes,
                      dropout_prob = config.dropout_prob,
                      gpu_mode = config.gpu_mode,
                      dropout = config.dropout
                    }
  self.lstm_layer =  dmn.LSTM_Encoder{
                    in_dim = config.emb_dim,
                    mem_dim = config.mem_dim,
                    num_layers = config.num_layers,
                    gpu_mode = config.gpu_mode
                    }

  self.hidden_inputs = new_hidden_activations_lstm(self.gpu_mode, self.mem_dim, self.num_layers)

  self.tot_modules = {}
  insert_modules_to_table(self.tot_modules, self.lstm_layer:getModules())
  insert_modules_to_table(self.tot_modules, self.embed_layer:getModules())

  self.modules = nn.Parallel()
  add_modules(self.modules, self.lstm_layer:getModules())
  add_modules(self.modules, self.embed_layer:getModules())
    
  if self.gpu_mode then
    self:set_gpu_mode()
  end

  print("Modules we're optimizing for question module")
  print(self.modules)
end

-- Enable Dropouts
function QuestionModule:enable_dropouts()
  self.embed_layer:enable_dropouts()
end

-- Disable Dropouts
function QuestionModule:disable_dropouts()
  self.embed_layer:disable_dropouts()
end



-- Resets depth to 1
function QuestionModule:reset_depth()
  self.lstm_layer:forget()
end


function QuestionModule:zeroGradParameters()
  self.grad_params:zero()
  self.lstm_layer:zeroGradParameters()
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- states: hidden, cell states of LSTM if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns lstm output, class predictions, and error if train, else not error 
function QuestionModule:forward(inputs)
    assert(inputs ~= nil, "Must specify inputs to forward")
    self.word_embeds = self.embed_layer:forward(inputs)
    local lstm_output = self.lstm_layer:forward(self.word_embeds, self.hidden_inputs, self.reverse)
    return lstm_output
end


-- Backpropagate: forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- hidden_inputs: {hidden_dim, hidden_tim} tensors
-- reverse: True if reverse input, false otherwise
-- errors: T x num_layers x num_hidden tensor
-- class_predictions: T x 1 tensor of predictions
-- labels: actual labels
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function QuestionModule:backward(inputs, errors)
  assert(inputs ~= nil, "Must put in gru regular inputs")
  assert(errors ~= nil, "must put in lstm outputs")
  local lstm_input_derivs, hidden_derivs = 
  self.lstm_layer:backward(self.word_embeds, self.hidden_inputs, self.reverse, errors)
  local emb_errors = self.embed_layer:backward(inputs, lstm_input_derivs)
  return lstm_input_derivs, hidden_derivs
end

function QuestionModule:grad_check()
  self.params, self.grad_params = self.modules:getParameters()
  local input_indices = torch.IntTensor{1, 2, 3, 2, 3, 4, 2, 3, 4, 2, 3, 4}
  local criterion = nn.MSECriterion()
  local desired_state = torch.rand(self.lstm_layer.mem_dim)

  local currIndex = 0
  local feval = function(x)
      self.grad_params:zero()
      local lstm_output = self:forward(input_indices)
      local loss = criterion:forward(lstm_output, desired_state)
      local errors = criterion:backward(lstm_output, desired_state)
      self:backward(input_indices, errors)
      currIndex = currIndex + 1
      print(currIndex, " of ", self.params:size(1))    
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    print("Gradient error for question module network is")
    print(diff)
    assert(diff < 1e-5, "Gradient is greater than tolerance")
end

-- Sets all networks to gpu mode
function QuestionModule:set_gpu_mode()
  self.lstm_layer:set_gpu_mode()
  self.embed_layer:set_gpu_mode()
end

-- Sets all networks to cpu mode
function QuestionModule:set_cpu_mode()
  self.lstm_layer:set_cpu_mode()
  self.embed_layer:set_cpu_mode()
end

function QuestionModule:getModules() 
  return self.tot_modules
end

function QuestionModule:getParameters()
  return self.params, self.grad_params
end

function QuestionModule:getWeights()
  return self.params
end

-- Resets depths for lstm
function QuestionModule:forget()
  self.lstm_layer:forget()
end

function QuestionModule:print_config()
  printf('%-25s = %d\n', 'embed dimension', self.config.emb_dim)
  printf('%-25s = %d\n', 'input dimension', self.config.in_dim)
  printf('%-25s = %s\n', 'use dropout', self.config.dropout)
  printf('%-25s = %f\n', 'dropout probability', self.config.dropout_prob)
  printf('%-25s = %d\n', 'number of classes', self.config.num_classes)
  printf('%-25s = %d\n', 'memory dimension', self.config.mem_dim)
  printf('%-25s = %d\n', 'number of layers', self.config.num_layers)
end



