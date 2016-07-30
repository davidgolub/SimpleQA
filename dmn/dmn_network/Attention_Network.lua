local Attention_Network = torch.class('dmn.Attention_Network')

function Attention_Network:__init(config) 
  self:check_config(config)

  -- make sure there is no rep exposure
  self.config = dmn.functions.deepcopy(config)

  self.char_level = config.char_level
  self.pad_size = config.pad_size

  self.predicate_vocab = config.p_vocab
  self.entity_vocab = config.e_vocab
  self.question_vocab = config.q_vocab
  self.gpu_mode = config.gpu_mode
  self.model_epoch  = config.model_epoch or 0;

  self:init_layers(config)

  self.batch_size = config.batch_size or 100
  self.optim_method_string = config.optim_method_string or "rmsprop"
  if self.optim_method_string == 'adagrad' then
    self.optim_method = optim.adagrad
  elseif self.optim_method_string == 'rmsprop' then
    self.optim_method = optim.rmsprop
  else
    self.optim_method = optim.sgd
  end
  self:init_values()

  self.optim_state = config.optim_state
  --self.reg = 1e-4;

  self.reverse = false

  local modules = nn.Parallel()
                  :add(self.projection_layer)

  for i = 1, #self.layers do
    dmn.logger:print(self.layers[i])
    local curr_modules = self.layers[i]:getModules()
    add_modules(modules, curr_modules)
  end

  if self.config.version == 'v2' then 
    -- add constant layer
    modules:add(self.attention_constant_layer)
  end

  if self.gpu_mode then 
    self:set_gpu_mode()
  end

  dmn.logger:print("==== Modules we're optimizing for entire Attention Network ====")
  dmn.logger:print(modules)

  self.params, self.grad_params = modules:getParameters()
  self:print_config()
end

function Attention_Network:reset_depth()
  self.attention_layer:reset_depth()
end

function Attention_Network:forget()
  --Resets depth (needed for LSTMs)
  print("Forgetting");
  self.question_lstm_layer:forget()
  self.attention_layer:forget()
end

function Attention_Network:sample_inputs(inputs, value_to_avoid, num_inputs)
  assert(inputs ~= nil, "Inputs to evaluate must not be null")
  assert(value_to_avoid ~= nil, "Must specify the value to avoid")
  assert(num_inputs ~= nil, "Must specify number of inputs to the dataset")

  local neg_sample = datasets.generate_negative_sample(inputs, value_to_avoid, num_inputs)
  assert(#neg_sample == num_inputs, "Number of negative samples must equal number of inputs specified")
  return neg_sample
end

-- Makes sure that all parameters are specified for this dmn network
function Attention_Network:check_config(config)
  assert(config.version ~= nil, "Must specify correct version for attention network")
  assert(config.gpu_mode ~= nil, "Must specify gpu mode")
  assert(config.char_level ~= nil, "Must specify char level or not")
  assert(config.pad_size ~= nil, "Must specify pad size to use")

  assert(config.e_vocab ~= nil, "Must specify entity vocab")
  assert(config.q_vocab ~= nil, "Must specify question vocab")
  assert(config.p_vocab ~= nil, "Must specify predicate vocab")

  assert(config.optim_state ~= nil, "Must specify optim state")

  -- answer layer parameters
  assert(config.answer_rerank_criterion ~= nil, "Must specify attention rerank type")
  
  -- attention layer parameters
  assert(config.attention_mem_dim ~= nil, "Must specify memory dimension of attention layer")
  assert(config.attention_num_layers ~= nil, "Must specify number of layers for attention layer")
  assert(config.attention_type ~= nil, "Must specify attention type to use")

  -- question parameters
  assert(config.question_emb_dim ~= nil, "Must specify question embed dimensions")
  assert(config.question_mem_dim ~= nil, "Must specify question memory dimensions")
  assert(config.question_num_layers ~= nil, "Must specify number of layers for question")
  assert(config.question_num_classes ~= nil, "Must specify number of classes/unique tokens for question")
  assert(config.question_dropout_prob ~= nil, "Must specify question dropout probability")
  assert(config.question_dropout ~= nil, "Must specify whether to use question dropout or not")
  assert(config.question_lstm_type ~= nil, "Must specify question lstm type to use")

  -- predicate parameters
  assert(config.predicate_emb_dim ~= nil, "Must specify question embed dimensions")
  assert(config.predicate_hidden_dim ~= nil, "Must specify input hidden dimensions")
  assert(config.predicate_out_dim ~= nil, "Must specify output dimension for question embeddings (of dssm)")
  assert(config.predicate_in_stride ~= nil, "Must specify question input stride")
  assert(config.predicate_in_kernel_width ~= nil, "Must specify question input kernel width")
  assert(config.predicate_hidden_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.predicate_hidden_stride ~= nil, "Must specify question hidden stride")
  assert(config.predicate_out_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.predicate_out_stride ~= nil, "Must specify question hidden stride")
  assert(config.predicate_num_classes ~= nil, "Must specify number of classes")
  assert(config.predicate_dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.predicate_dropout ~= nil, "Must specify whether to use dropout or not")

  -- entity parameters
  assert(config.entity_emb_dim ~= nil, "Must specify question embed dimensions")
  assert(config.entity_hidden_dim ~= nil, "Must specify input hidden dimensions")
  assert(config.entity_out_dim ~= nil, "Must specify output dimension for question embeddings (of dssm)")
  assert(config.entity_in_stride ~= nil, "Must specify question input stride")
  assert(config.entity_in_kernel_width ~= nil, "Must specify question input kernel width")
  assert(config.entity_hidden_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.entity_hidden_stride ~= nil, "Must specify question hidden stride")
  assert(config.entity_out_kernel_width ~= nil, "Must specify question hidden kernel width")
  assert(config.entity_out_stride ~= nil, "Must specify question hidden stride")
  assert(config.entity_num_classes ~= nil, "Must specify number of classes")
  assert(config.entity_dropout_prob ~= nil, "Must specify dropout probability")
  assert(config.entity_dropout ~= nil, "Must specify whether to use dropout or not")
end

-- enables dropouts on all layers
function Attention_Network:enable_dropouts()
  dmn.logger:print("====== Enabling dropouts ======")
  self.projection_layer:training()
  dmn.functions.enable_dropouts(self.layers)
  dmn.functions.enable_dropouts(self.predicate_dssm_layers)
  dmn.functions.enable_dropouts(self.predicate_embed_layers)
  dmn.functions.enable_dropouts(self.entity_dssm_layers)
  dmn.functions.enable_dropouts(self.entity_embed_layers)
end

-- disables dropouts on all layers
function Attention_Network:disable_dropouts()
  dmn.logger:print("====== Disabling dropouts ======")
  self.projection_layer:evaluate()
  dmn.functions.disable_dropouts(self.layers)
  dmn.functions.disable_dropouts(self.predicate_dssm_layers)
  dmn.functions.disable_dropouts(self.predicate_embed_layers)
  dmn.functions.disable_dropouts(self.entity_dssm_layers)
  dmn.functions.disable_dropouts(self.entity_embed_layers)
end

-- enables gpus on dmn network
function Attention_Network:set_gpu_mode()
  dmn.logger:print("====== Setting gpu mode ======")
  self.config.gpu_mode = true
  self.projection_layer:cuda()
  dmn.functions.set_gpu_mode(self.layers)
  dmn.functions.set_gpu_mode(self.predicate_dssm_layers)
  dmn.functions.set_gpu_mode(self.predicate_embed_layers)
  dmn.functions.set_gpu_mode(self.entity_dssm_layers)
  dmn.functions.set_gpu_mode(self.entity_embed_layers)
  if self.config.version == 'v2' then 
    self.attention_constant_layer:cuda()
    self.attention_constant_input = self.attention_constant_input:cuda()
  end
 
  self:init_values()
end

-- converts to cpus on dmn network
function Attention_Network:set_cpu_mode()
  dmn.logger:print("====== Setting cpu mode ======")
  self.config.gpu_mode = false
  self.projection_layer:double()
  dmn.functions.set_cpu_mode(self.layers)
  dmn.functions.set_cpu_mode(self.predicate_dssm_layers)
  dmn.functions.set_cpu_mode(self.predicate_embed_layers)
  dmn.functions.set_cpu_mode(self.entity_dssm_layers)
  dmn.functions.set_cpu_mode(self.entity_embed_layers)
  self.attention_constant_layer:double()
  if self.config.version == 'v2' then 
    self.attention_constant_layer:double()
    self.attention_constant_input = self.attention_constant_input:double()
  end
 
  self:init_values()
end

function Attention_Network:init_values()
  self.tensor_type = self.config.gpu_mode and torch.CudaTensor or torch.DoubleTensor
  self.attention_hid_inputs = self.attention_layer:new_initial_values()
  self.attention_constant_input = self.config.gpu_mode and torch.CudaTensor{1} or torch.IntTensor{1}
  self.question_lstm_hid_inputs = self.question_lstm_layer:new_initial_values()
end


-- Initializes all the layers of the dynamic memory network
function Attention_Network:init_layers(config)
  assert(config ~= nil, "Must specify config for answer layer")
  self.predicate_answer_layer = self:new_answer_layer(config)
  self.entity_answer_layer = self:new_answer_layer(config)

  -- for predicting the target entities via attention of question layer
  self.attention_layer = self:new_attention_layer(config)
  self.attention_constant_layer = nn.LookupTable(1, config.predicate_out_dim)

  -- for dummy input into attention layer
  -- for question embeddings
  self.question_embed_layer = self:new_question_embed_layer(config)
  self.question_lstm_layer = self:new_question_lstm_layer(config)
 
  -- for candidate entity embeddings
  self.master_entity_embed_layer = self:new_entity_embed_layer(config)
  self.master_entity_dssm_layer = self:new_entity_dssm_layer(config)
  
  -- for candidate predicate embeddings
  self.master_predicate_embed_layer = self:new_predicate_embed_layer(config)
  self.master_predicate_dssm_layer = self:new_predicate_dssm_layer(config)

  -- replicate those layers for multiple inputs, but share the parameters
  self.entity_embed_layers = {}
  self.entity_dssm_layers = {}

  self.predicate_embed_layers = {}
  self.predicate_dssm_layers = {}

  -- once you have attention vectors you want to project them to a similar space
  self.projection_layer = self:new_projection_layer()

  self.layers = {self.predicate_answer_layer,
                self.entity_answer_layer,
                self.attention_layer,
                self.question_embed_layer,
                self.question_lstm_layer,
                self.master_entity_embed_layer,
                self.master_entity_dssm_layer,
                self.master_predicate_embed_layer,
                self.master_predicate_dssm_layer}

end

function Attention_Network:new_projection_layer()
  local network = nn.Sequential()
                  --:add(nn.Linear(self.attention_layer.mem_dim, self.config.predicate_out_dim))
                  :add(nn.Tanh())
  return network
end

-- Returns new question dssm layer corresponding to config
function Attention_Network:new_answer_layer(config)
  local answer_layer = dmn.AnswerRerankModule{
                        rerank_criterion = config.answer_rerank_criterion,
                        gpu_mode = config.gpu_mode
                        }
  return answer_layer
end

-- Returns new lstm attention layer corresredconding to config
function Attention_Network:new_attention_layer(config)
  local attention_layer = dmn.Attention_LSTM_Decoder{
                        gpu_mode = config.gpu_mode,
                        in_dim = config.predicate_out_dim,
                        context_dim = config.question_mem_dim,
                        mem_dim = config.attention_mem_dim,
                        num_layers = config.attention_num_layers,
                        attention_type = config.attention_type
                        }
  return attention_layer
end

-- Returns new question layer corresponding to config
function Attention_Network:new_question_lstm_layer(config)
  local question_layer = dmn.LSTM_Decoder{
              gpu_mode = config.gpu_mode,
              in_dim = config.question_emb_dim,
              mem_dim = config.question_mem_dim,
              num_layers = config.question_num_layers,
              lstm_type = config.question_lstm_type
            }
  return question_layer
end

-- Returns new input dssm layer corresponding to config
function Attention_Network:new_entity_dssm_layer(config)
  assert(config ~= nil, "Must specify config for entity dssm layer")
  local entity_layer = dmn.DSSM_Layer{
              dssm_type = config.dssm_type,
              gpu_mode = config.gpu_mode,
              in_dim = config.entity_emb_dim,
              hidden_dim = config.entity_hidden_dim,
              out_dim = config.entity_out_dim,
              in_stride = config.entity_in_stride,
              in_kernel_width = config.entity_in_kernel_width,
              hidden_stride = config.entity_hidden_stride,
              hidden_kernel_width = config.entity_hidden_kernel_width,
              out_stride = config.entity_out_stride,
              out_kernel_width = config.entity_out_kernel_width
            }
  if self.master_entity_dssm_layer ~= nil then
     entity_layer:share(self.master_entity_dssm_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return entity_layer
end

-- Returns new input dssm layer corresponding to config
function Attention_Network:new_predicate_dssm_layer(config)
  assert(config ~= nil, "Must specify config for input dssm layer")
  local predicate_layer = dmn.DSSM_Layer{
              dssm_type = 'hidden_network',
              gpu_mode = config.gpu_mode,
              in_dim = config.predicate_emb_dim,
              hidden_dim = config.predicate_hidden_dim,
              out_dim = config.predicate_out_dim,
              in_stride = config.predicate_in_stride,
              in_kernel_width = config.predicate_in_kernel_width,
              hidden_stride = config.predicate_hidden_stride,
              hidden_kernel_width = config.predicate_hidden_kernel_width,
              out_stride = config.predicate_out_stride,
              out_kernel_width = config.predicate_out_kernel_width
            }
  if self.master_predicate_dssm_layer ~= nil then
     predicate_layer:share(self.master_predicate_dssm_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end
  return predicate_layer
end

-- Returns new question embed module corresponding to config
function Attention_Network:new_question_embed_layer(config)
  assert(config ~= nil, "Must specify config for question embed layer")
  local question_embed_layer = dmn.WordEmbedModule{
    gpu_mode = config.gpu_mode,
    emb_dim = config.question_emb_dim,
    num_classes = config.question_num_classes,
    dropout_prob = config.question_dropout_prob,
    dropout = config.question_dropout,
    hashing = self.question_vocab.hashed, -- use hashing if the vocab is hashed
    dropout_prob = config.question_dropout_prob,
    dropout = config.question_dropout,
  }
  return question_embed_layer
end

-- Returns new input embed module corresponding to config
function Attention_Network:new_entity_embed_layer(config)
  assert(config ~= nil, "Must specify config for input embed layer")

  local entity_embed_layer = dmn.WordEmbedModule{
    gpu_mode = config.gpu_mode,
    emb_dim = config.entity_emb_dim,
    num_classes = config.entity_num_classes,
    dropout_prob = config.entity_dropout_prob,
    dropout = config.entity_dropout,
    hashing = self.entity_vocab.hashed -- use hashing if the vocab is hashed
  }

  if self.master_entity_embed_layer ~= nil then
     entity_embed_layer:share(self.master_entity_embed_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end

  return entity_embed_layer
end

-- Returns new input embed module corresponding to config
function Attention_Network:new_predicate_embed_layer(config)
  assert(config ~= nil, "Must specify config for input embed layer")

  local predicate_embed_layer = dmn.WordEmbedModule{
    gpu_mode = config.gpu_mode,
    emb_dim = config.predicate_emb_dim,
    num_classes = config.predicate_num_classes,
    dropout_prob = config.predicate_dropout_prob,
    dropout = config.predicate_dropout,
    hashing = self.predicate_vocab.hashed -- use hashing if the vocab is hashed
  }

  if self.master_predicate_embed_layer ~= nil then
     predicate_embed_layer:share(self.master_predicate_embed_layer,
      'weight', 'bias', 'gradWeight', 'gradBias')
  end

  return predicate_embed_layer
end


-- Forward propagate.
-- question_indices: IntTensor which represents question indices
-- entity_indices:
-- predicate_indices: Indices for predicates
-- corr_entity_index: Correct entity index in table of entity_indices
-- corr_predicate_index: Correct predicate index in table of predicate_indices
-- returns cosine distance between two
function Attention_Network:forward(question_indices, predicate_indices, entity_indices, 
  corr_entity_index, corr_predicate_index,
  predicate_sampling_probs,
  entity_sampling_probs)
  assert(question_indices ~= nil, "Must specify question indices to DSSM net")
  assert(entity_indices ~= nil, "Must specify input indices to DSSM net")
  assert(predicate_indices ~= nil, "Must specify predicate indices")
  assert(corr_entity_index ~= nil, "Must specify correct entity index for softmax")
  assert(corr_predicate_index ~= nil, "Must specify correct predicate index for softmax")
  assert(predicate_sampling_probs ~= nil, "Must specify predicate sampling probs to forward")
  assert(entity_sampling_probs ~= nil, "Must specify entity sampling probs to forward")

  -- construct question embeddings, question lstms
  local t1 = sys.clock() 
  self.question_emb = self.question_embed_layer:forward(question_indices)
  self.question_lstm = self.question_lstm_layer:forward(self.question_emb, self.question_lstm_hid_inputs, false)

  local t2 = sys.clock()
  -- construct entity and predicate dssm representations
  self.entity_dssm = self:forward_entities(entity_indices, false) 
  self.predicate_dssm = self:forward_predicates(predicate_indices, false)

  local t3 = sys.clock()
  -- construct inputs to attention layer
  local attention_inputs = self.tensor_type(2, self.config.predicate_out_dim):zero()

  if self.config.version == 'v2' then 
    attention_inputs[1] = self.attention_constant_layer:forward(self.attention_constant_input)
  end

  attention_inputs[2] = self.predicate_dssm[corr_predicate_index]

  self.attention_inputs = attention_inputs

  self.context_vectors = self.attention_layer:forward(self.attention_inputs, self.question_lstm, self.attention_hid_inputs, false)
  self.projected_vectors = self.projection_layer:forward(self.context_vectors)

  local t4 = sys.clock()
  -- first projected vector should be entity
  local predicate_loss = self.predicate_answer_layer:forward(self.projected_vectors[1], self.predicate_dssm, 
    corr_predicate_index, predicate_sampling_probs) 

  local t5 = sys.clock()
  local entity_loss = self.entity_answer_layer:forward(self.projected_vectors[2], self.entity_dssm, 
    corr_entity_index, entity_sampling_probs) 

  local t6 = sys.clock()

  --dmn.logger:print("Forward on attention network")
  --dmn.logger:print(t6 - t5, t5 - t4, t4 - t3, t3 - t2, t2 - t1)
  return predicate_loss + entity_loss
end

-- returns dssm of input indices, predict or not determines whether to use same net for
-- forwarding
function Attention_Network:forward_entities(entity_indices, predict)
  assert(entity_indices ~= nil, "Must specify input indices to attention net")
  assert(predict ~= nil, "Must specify whether to predict or not")
  self.entity_emb = {}
  local outputs = {}

  for i = 1, #entity_indices do
    local curr_input = entity_indices[i]
    local curr_network_index = i
    if self.entity_embed_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new input embed layer for index ", i)
      self.entity_embed_layers[curr_network_index] = self:new_entity_embed_layer(self.config)
    end

    if self.entity_dssm_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new input dssm layer for index ", i)
      self.entity_dssm_layers[curr_network_index] = self:new_entity_dssm_layer(self.config)
    end
    self.entity_emb[i] = self.entity_embed_layers[curr_network_index]:forward(curr_input):clone()
    outputs[i] = self.entity_dssm_layers[curr_network_index]:forward(self.entity_emb[i]):clone()
  end

  return outputs
end

-- returns dssm of input indices, predict or not determines whether to use same net for
-- forwarding
function Attention_Network:forward_predicates(predicate_indices, predict)
  assert(predicate_indices ~= nil, "Must specify input indices to DSSM net")
  assert(predict ~= nil, "Must specify whether to predict or not")
  self.predicate_emb = {}
  local outputs = {}

  for i = 1, #predicate_indices do
    local curr_input = predicate_indices[i]
    local curr_network_index = i
    if self.predicate_embed_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new predicate embed layer for index ", i)
      self.predicate_embed_layers[curr_network_index] = self:new_predicate_embed_layer(self.config)
    end
    if self.predicate_dssm_layers[curr_network_index] == nil then
      dmn.logger:print("Creating a new input dssm layer for index ", i)
      self.predicate_dssm_layers[curr_network_index] = self:new_predicate_dssm_layer(self.config)
    end
    self.predicate_emb[i] = self.predicate_embed_layers[curr_network_index]:forward(curr_input):clone()
    outputs[i] = self.predicate_dssm_layers[curr_network_index]:forward(self.predicate_emb[i]):clone()
  end 

  return outputs
end

-- Backpropagate.
-- question_indices: IntTensor which represents question indices
-- entity_indices:
-- predicate_indices: Indices for predicates
-- corr_entity_index: Correct entity index in table of entity_indices
-- corr_predicate_index: Correct predicate index in table of predicate_indices
-- returns cosine distance between two
function Attention_Network:backward(question_indices, predicate_indices, entity_indices, 
  corr_entity_index, corr_predicate_index, predicate_sampling_probs,
  entity_sampling_probs)
  assert(question_indices ~= nil, "Must specify question indices to DSSM net")
  assert(entity_indices ~= nil, "Must specify input indices to DSSM net")
  assert(predicate_indices ~= nil, "Must specify predicate indices")
  assert(corr_entity_index ~= nil, "Must specify correct entity index for softmax")
  assert(corr_predicate_index ~= nil, "Must specify correct predicate index for softmax")
  assert(predicate_sampling_probs ~= nil, "Must specify predicate sampling probs to forward")
  assert(entity_sampling_probs ~= nil, "Must specify entity sampling probs to forward")

  local t1 = sys.clock()
  -- first backprop through answer layer, but make a copy
  self.projected_errs = self.tensor_type(2, self.config.predicate_out_dim):zero()
  
  -- unpack projected predicate errors
  local proj_pred_errs = self.predicate_answer_layer:backward(self.projected_vectors[1], self.predicate_dssm, 
    corr_predicate_index, predicate_sampling_probs)
  self.projected_errs[1] = proj_pred_errs[1]
  self.predicate_dssm_errs = proj_pred_errs[2]

  local t2 = sys.clock()
  -- unpack projected entity errors
  local proj_entity_errs = self.entity_answer_layer:backward(self.projected_vectors[2], self.entity_dssm, 
    corr_entity_index, entity_sampling_probs) 

  self.projected_errs[2] = proj_entity_errs[1]
  self.entity_dssm_errs = proj_entity_errs[2]

  local t3 = sys.clock()
  -- now backprop through projection layer
  local context_errs = self.projection_layer:backward(self.projected_vectors, self.projected_errs)

  local t4 = sys.clock()
  -- now backprop through the attention layer
  local attention_input_errs, attention_context_errs = self.attention_layer:backward(self.attention_inputs, 
                                                        self.question_lstm, 
                                                        self.attention_hid_inputs, 
                                                        false,
                                                        context_errs)

  if self.config.version == 'v2' then 
    self.attention_constant_layer:backward(self.attention_constant_input, attention_input_errs[1])
  end

  local t5 = sys.clock()
  -- add errors to predicate errors at the correct index
  self.predicate_dssm_errs[corr_predicate_index]:add(attention_input_errs[2])

  local t6 = sys.clock()
  -- backprop through question embeddings
  self.question_emb_errs = self.question_lstm_layer:backward(self.question_emb, self.question_lstm_hid_inputs, false, attention_context_errs)

  local t7 = sys.clock()
  self.question_embed_layer:backward(question_indices, self.question_emb_errs)

  local t8 = sys.clock()

  -- backprop through predicates
  self:backward_predicates(predicate_indices, self.predicate_dssm_errs)

  local t9 = sys.clock()

  -- backprop through entities
  self:backward_entities(entity_indices, self.entity_dssm_errs)
  local t10 = sys.clock()

  --dmn.logger:print("Backward on Attention Network")
  --dmn.logger:print(t10 - t9, t9 - t8, t8 - t7, t7 - t6, t6 - t5, t5 - t4, t4 - t3, t3 - t2, t2 - t1)
end

-- returns cosine distance between two
function Attention_Network:backward_predicates(predicate_indices, predicate_errs)
  assert(predicate_indices ~= nil, "Must specify input indices to Attention net")
  assert(predicate_errs ~= nil, "Must specify input errors to Attention net")

  for i = 1, #predicate_indices do
    -- get current input
    local curr_input_indices = predicate_indices[i]
    local curr_input_emb = self.predicate_emb[i]
    local curr_input_err = predicate_errs[i]

    local curr_err = self.predicate_dssm_layers[i]:backward(curr_input_emb, curr_input_err)
    local input_err = self.predicate_embed_layers[i]:backward(curr_input_indices, curr_err)
  end
end

-- returns cosine distance between two
function Attention_Network:backward_entities(entity_indices, entity_errs)
  assert(entity_indices ~= nil, "Must specify entity indices to Attention net")
  assert(entity_errs ~= nil, "Must specify entity errors to Attention net")

  for i = 1, #entity_indices do
    -- get current input
    local curr_input_indices = entity_indices[i]
    local curr_input_emb = self.entity_emb[i]
    local curr_input_err = entity_errs[i]

    local curr_err = self.entity_dssm_layers[i]:backward(curr_input_emb, curr_input_err)
    local input_err = self.entity_embed_layers[i]:backward(curr_input_indices, curr_err)
  end
end


function Attention_Network:predict_dataset(dataset, beam_size, num_predictions)
  assert(dataset ~= nil, "Must specify dataset to predict in")
  assert(beam_size ~= nil, "Must specify beam size to use for predictions")
  assert(num_predictions ~= nil, "Must specify number of predictions to predict")
  assert(num_predictions <= dataset.size, "Number of predictions must not exceed dataset size")
 
  self:disable_dropouts()
  local num_samples = 200
  local predictions = {}

  for i = 1, num_predictions do
    xlua.progress(i, num_predictions)
    local question_indices = dataset.questions[i]

    local positive_predicate = dataset.positive_predicates[i]
    local negative_predicate_indices = 
    self:sample_inputs(dataset.positive_predicates, i, num_samples)

    local positive_entity = dataset.positive_entities[i]
    local negative_entity_indices = 
      self:sample_inputs(dataset.positive_entities, i, num_samples)

    local entity_inputs = {positive_entity}
    local predicate_inputs = {positive_predicate}

    for j = 1, #negative_predicate_indices do 
      local cur_index = negative_predicate_indices[j]
      local negative_predicate = dataset.positive_predicates[cur_index]
      table.insert(predicate_inputs, negative_predicate)
    end

    for j = 1, #negative_entity_indices do 
      local cur_index = negative_entity_indices[j]
      local negative_entity = dataset.positive_entities[cur_index]
      table.insert(entity_inputs, negative_entity)
    end

    -- prediction gives indices from the top 50, so we want to actually get the 
    -- real entity and predicate indices
    all_predictions = self:predict_tokenized(question_indices, predicate_inputs, entity_inputs, beam_size)

    -- get actual predictions for current index
    local act_predictions = {}

    for j = 1, #all_predictions do 
        local prediction = all_predictions[j]
        local predicate_index = prediction[2]
        local entity_index = prediction[3]

        best_predicate_index = (predicate_index == 1) and i or 
                             negative_predicate_indices[predicate_index - 1]

        best_entity_index = (entity_index == 1) and i or 
                             negative_entity_indices[entity_index - 1]
        act_prediction = {prediction[1], best_predicate_index, best_entity_index} 
        table.insert(act_predictions, act_prediction)
    end

    table.insert(predictions, act_predictions)
  end

  return predictions
end

-- Returns rankings and likelihoods of predicates/entities
function Attention_Network:predict(question, candidate_predicates, candidate_entities, beam_size)
  assert(question ~= nil, "Must specify question to predict")
  assert(candidate_predicates ~= nil, "Must specify candidate predicates to forward")
  assert(candidate_entities ~= nil, "Must specify candidate entities to forward")
  assert(beam_size ~= nil, "Must specify beam size to predict")
  
  self:disable_dropouts()

  function get_tokens(sent, vocab)
    local indeces = datasets.get_input_tokens(sent, vocab, self.gpu_mode, self.char_level,
      self.pad_size)
    return indeces
  end

  function get_multi_tokens(sentences, vocab)
    local tokens = {}
    for i = 1, #sentences do
      local curr_tokens = get_tokens(sentences[i], vocab, self.gpu_mode,
        self.char_level, self.pad_size)
      table.insert(tokens, curr_tokens)
    end
    return tokens
  end

  -- get all of the tokens
  local question_tokens = get_tokens(question, self.question_vocab)
  local predicate_tokens = get_multi_tokens(candidate_predicates, self.predicate_vocab)  
  local entity_tokens = get_multi_tokens(candidate_entities, self.entity_vocab)  

  -- get predicted tokens
  local predicted_tokens = self:predict_tokenized(question_tokens, 
      predicate_tokens, entity_tokens, beam_size)
  
  -- get likelihoods and rankings
  local likelihoods = {}
  local rankings = {}
  local question_pred_focuses = {}
  local question_entity_focuses = {}

  for i = 1, beam_size do 
    local prediction = predicted_tokens[i]
    local likelihood = prediction[1]
    local predicate_index = prediction[2]
    local entity_index = prediction[3]

    local question_pred_focus = prediction[4]
    local question_entity_focus = prediction[5]

    table.insert(likelihoods, likelihood)
    table.insert(rankings, {predicate_index, entity_index})
    table.insert(question_pred_focuses, question_pred_focus)
    table.insert(question_entity_focuses, question_entity_focus)
  end
  
  return rankings, likelihoods, question_pred_focuses, question_entity_focuses
end

-- predicts with tokenized question indices and word indices
-- returns: {loglikelihood, best_predicate_index, best_entity_index}
function Attention_Network:predict_tokenized(question_indices, predicate_indices, entity_indices, beam_size)
  assert(question_indices ~= nil, "Must specify question indices to Attention net")
  assert(entity_indices ~= nil, "Must specify input indices to Attention net")
  assert(predicate_indices ~= nil, "Must specify predicate indices")
  assert(beam_size ~= nil, "Must specify beam size for prediction")

  self:forget()

  -- construct question embeddings, question lstms
  self.question_emb = self.question_embed_layer:forward(question_indices)
  self.question_lstm = self.question_lstm_layer:forward(self.question_emb, self.question_lstm_hid_inputs, false)

  -- construct entity and predicate dssm representations
  self.entity_dssm = self:forward_entities(entity_indices, false) 
  self.predicate_dssm = self:forward_predicates(predicate_indices, false)

  local function get_indices(attention_inputs, attention_hidden_inputs,
   answer_layer, candidate_dssms, beam_size)
    assert(attention_inputs ~= nil, "Must specify attention inputs for attention layer")
    assert(answer_layer ~= nil, "Must specify answer layer to predict softmax outputs")
    assert(candidate_dssms ~= nil, "Must candidate dssms for answer layer")
    assert(attention_hidden_inputs ~= nil, "Must specify hidden inputs to attention module")
    assert(beam_size ~= nil, "Must specify beam size for indices")

    -- gives both cell and hidden states, 
    local attention_outputs = self.attention_layer:tick(attention_inputs, self.question_lstm, 
      attention_hidden_inputs, false)

    -- extract hidden state from attention output
    local context_vectors = attention_outputs[2]

    -- extract importance of input vectors (summed up over hidden states)

    local projected_vectors = self.projection_layer:forward(context_vectors)

    -- first projected vector should be entity
    local predicate_indices = answer_layer:predict(projected_vectors, candidate_dssms, beam_size)

    -- for now do beam search with size 1: greedy search
    local best_indices = {}
    local best_likelihoods = {}

    for i = 1, #predicate_indices do 
      best_indices[i] = predicate_indices[i][2]
      best_likelihoods[i] = predicate_indices[i][1]
    end

    local copied_outputs = dmn.functions.deepcopy(attention_outputs)
    local question_importance = copied_outputs[3]
    return best_indices, best_likelihoods, {copied_outputs[1], copied_outputs[2]}, 
    question_importance
  end

  -- first predict predicate
  local init_inputs = self.tensor_type(self.config.predicate_out_dim):zero()
  local best_predicate_indeces, best_predicate_likelihoods, hidden_state, predicate_question_importance = 
  get_indices(init_inputs, 
              self.attention_hid_inputs, 
              self.predicate_answer_layer, 
              self.predicate_dssm,
              beam_size)

  -- then predict entity
  local best_predicate_index = best_predicate_indeces[1]
  local predicate_inputs = self.predicate_dssm[best_predicate_index]
  local best_entity_indeces, best_entity_likelihoods, hidden_state, entity_question_importance = 
  get_indices(predicate_inputs, 
    hidden_state, 
    self.entity_answer_layer, 
    self.entity_dssm,
    beam_size)

  local results = {}

  for i = 1, beam_size do 
    table.insert(results, 
        {best_predicate_likelihoods[i] + best_entity_likelihoods[i], 
         best_predicate_indeces[i], 
         best_entity_indeces[i],
         predicate_question_importance,
         entity_question_importance})
  end
  return results
end

-- test predictions: predictions to get sentences from
-- dataset: dataset to get candidates from
-- saves predictions in form: predicate \t entity
function Attention_Network:get_sentences(test_predictions, dataset)
  assert(test_predictions ~= nil, "Must specify test predictions")
  assert(dataset ~= nil, "Must specify dataset to get inputs from")

  local sentences = {}
  --predictions_file:write("LOSS " .. loss .. '\n')
  for i = 1, #test_predictions do
    local test_prediction = test_predictions[i][1]

    local candidate_predicates = dataset.raw_positive_predicates
    local candidate_entities = dataset.raw_positive_entities

    local predicate_index = test_prediction[2]
    local entity_index = test_prediction[3]

    local sentence = candidate_predicates[predicate_index] .. 
                      " \t " .. candidate_entities[entity_index] 
    --dmn.logger:print(sentence)
    table.insert(sentences, sentence)
  end
  return sentences
end

-- gets sentence for single test prediction
function Attention_Network:get_sentence(test_prediction, candidates)
    local likelihood = test_prediction[1]
    local tokens = test_prediction[2]
    local index = test_prediction[3]

    -- two cases, if embeddings then can reconstruct input vocab
    -- Remove tokens
    if self.input_vocab.hashed then
      assert(candidates ~= nil, "Must specify candidates if using hash vocab")
      local candidate = candidates[index]
      return candidate
    else 
      local sentence = table.concat(self.input_vocab:tokens(tokens), ' ')
      local sentence = string.gsub(sentence, "</s>", "")
      local sentence = string.gsub(sentence, "<s>", "")
      return sentence
    end   
end

function Attention_Network:eval(dataset)
  assert(dataset ~= nil, "Dataset to evaluate must not be null")

  local num_total_correct = 0
  local num_entities_correct = 0
  local num_predicates_correct = 0
  local num_total = dataset.size

  local beam_size = 5
  -- first get predictions from dataset
  local all_predictions = self:predict_dataset(dataset, beam_size, num_total)  
  for i = 1, #all_predictions do
    -- get prediction and proposed predicate/entity indeces
    local curr_predictions = all_predictions[i]
    local predicate_index = curr_predictions[1][2]
    local entity_index = curr_predictions[1][3]

    -- get the raw question, and proposed predicate/entity
    local question_raw = dataset.raw_questions[i]
    local positive_entity = dataset.raw_positive_entities[i]
    local positive_predicate = dataset.raw_positive_predicates[i]

    local predicate_raw = dataset.raw_positive_predicates[predicate_index]
    local entity_raw = dataset.raw_positive_entities[entity_index]

    -- get the desired predicate and the desired entity raw
    local desired_predicate_raw = dataset.raw_positive_predicates[i]
    local desired_entity_raw = dataset.raw_positive_entities[i]

    if i % 1000 == 0 then
      dmn.logger:printf("Collecting garbage at index " .. i)
      collectgarbage()
    end

    dmn.logger:print("Predictions")
    dmn.logger:print("Question " .. question_raw)
    dmn.logger:printf("Predicate desired: %-25s actual %-25s\n", desired_predicate_raw, predicate_raw)
    dmn.logger:printf("Entity desired: %-25s actual %-25s\n", desired_entity_raw, entity_raw)

    if predicate_raw == desired_predicate_raw then
      dmn.logger:print("CORRECT predicate")
      num_predicates_correct = num_predicates_correct + 1
    end

    if entity_raw == desired_entity_raw then
      dmn.logger:print("CORRECT entity")
      num_entities_correct = num_entities_correct + 1
    end

    if predicate_raw == desired_predicate_raw and entity_raw == desired_entity_raw then
      dmn.logger:print("BOTH CORRECT")
      num_total_correct = num_total_correct + 1
    end
  end

  local tot_accuracy_at_1 = num_total_correct / num_total
  local pred_accuracy_at_1 = num_predicates_correct / num_total
  local entity_accuracy_at_1 = num_entities_correct / num_total

  softmax.run_api:add_point(dmn.constants.VAL_INDEX, self.model_epoch, tot_accuracy_at_1)
  softmax.run_api:add_point(dmn.constants.VAL_INDEX, self.model_epoch, pred_accuracy_at_1)

  softmax.run_api:add_point(dmn.constants.TEST_INDEX, self.model_epoch, tot_accuracy_at_1)
  softmax.run_api:add_point(dmn.constants.TEST_INDEX, self.model_epoch, entity_accuracy_at_1)

  return tot_accuracy_at_1, pred_accuracy_at_1, entity_accuracy_at_1
end

-- helper function to do forward backward
-- question_indices: indices of question to forward/backward
-- input_indices: indices of input to forward/backward
-- expected_similarity: expected_similarity to forward/backward
-- returns RMSE between similarity and desired similarity
function Attention_Network:forward_backward(question_indices, predicate_indices, entity_indices, 
  corr_entity_index, corr_predicate_index,
  predicate_sampling_probs, entity_sampling_probs)
  assert(question_indices ~= nil, "Must specify question indices to forward/backward")
  assert(entity_indices ~= nil, "Must specify entity indices to forward/backward")
  assert(predicate_indices ~= nil, "Must specify predicate indices to forward/backward")
  assert(corr_entity_index ~= nil, "Must specify correct entity index to forward/backward")
  assert(corr_predicate_index ~= nil, "Must specify correct predicate index")
  assert(predicate_sampling_probs ~= nil, "Must specify predicate sampling probs to forward/backward")
  assert(entity_sampling_probs ~= nil, "Must specify entity sampling probs to forward/backward")


  local loss = self:forward(question_indices, predicate_indices, entity_indices, 
    corr_entity_index, corr_predicate_index, predicate_sampling_probs, entity_sampling_probs)
  
  self:backward(question_indices, predicate_indices, entity_indices, 
    corr_entity_index, corr_predicate_index, predicate_sampling_probs, entity_sampling_probs)  
  return loss
end

-- grad check on actual inputs
function Attention_Network:grad_check(dataset)
  assert(dataset ~= nil, "Dataset to evaluate must not be null")
  self:enable_dropouts()

  local indices = torch.randperm(dataset.size)
  local tot_loss = 0
  local tot_num_items = 0
  local num_samples = 30

  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = 2
    currIndex = 0

    local feval = function(x)
      self.grad_params:zero()
      local start = sys.clock()
      local loss = 0
      local num_items = 0
      for j = 1, batch_size do
        local idx = i

        local desired_predicate_index = 1
        local desired_entity_index = 1

        local question_indices = dataset.questions[idx]
        local entity_inputs = {dataset.positive_entities[idx], dataset.positive_entities[idx + 1]}
        local entity_sampling_probs = torch.DoubleTensor(2):fill(0.01)

        local predicate_sampling_probs = torch.DoubleTensor(2):fill(0.01)
        local predicate_inputs = {dataset.positive_predicates[idx], dataset.positive_predicates[idx + 1]}

        local curr_loss = self:forward_backward(question_indices, 
                                               predicate_inputs, 
                                               entity_inputs,
                                               desired_predicate_index,
                                               desired_entity_index,
                                               predicate_sampling_probs,
                                               entity_sampling_probs)
        loss = loss + curr_loss
        num_items = num_items + 1 
      end

      tot_loss = tot_loss + loss
      tot_num_items = tot_num_items + num_items
      loss = loss / num_items
      self.grad_params:div(num_items)

      -- clamp grad params in accordance to karpathy from -5 to 5
      --self.grad_params:clamp(-5, 5)
      -- regularization: BAD BAD BAD
      --loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      --self.grad_params:add(self.reg, self.params)
      print("Current loss", loss)
      print(currIndex, "of", self.params:size())
      currIndex = currIndex + 1
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    diff, DC, DC_est = optim.checkgrad(feval, self.params, 1e-7)
    dmn.logger:print("Gradient error for attention network  is")
    dmn.logger:print(diff)
    assert(diff < 1e-5, "Gradient is greater than tolerance")
    assert(false)
    self.optim_method(feval, self.params, self.optim_state)
  end


  average_loss = tot_loss / tot_num_items
  xlua.progress(dataset.size, dataset.size)
  return average_loss
end

function Attention_Network:train(dataset)
  assert(dataset ~= nil, "Dataset to evaluate must not be null")

  -- first set learning rate
  if self.config.optim_method_string == 'sgd' then 
    local cur_learning_rate = self.config.optim_state.learningRate
    self.config.optim_state.learningRate = 
      dmn.math_functions.get_learning_rate(cur_learning_rate, self.model_epoch)
  end

  self:enable_dropouts()
  self.model_epoch = self.model_epoch + 1

  local indices = torch.randperm(dataset.size)
  local tot_loss = 0
  local num_samples = 50
  local tot_num_items = 0
  local cur_collect_garbage = 1

  --dataset.size
  for i = 1, dataset.size, self.batch_size do --dataset.size, self.batch_size do

    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1
    currIndex = 0

    if i - cur_collect_garbage > 2000 then 
      dmn.functions.collect_garbage()
      cur_collect_garbage = i
    end

    local feval = function(x)
      self.grad_params:zero()
      local start = sys.clock()
      local loss = 0
      local num_items = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]

        local desired_predicate_index = 1
        local desired_entity_index = 1

        -- first forward/backward positive samples
        local question_indices = dataset.questions[idx]
        local positive_predicate = dataset.positive_predicates[idx]
        local negative_predicate_indices = self:sample_inputs(dataset.positive_predicates, positive_predicate, num_samples)

        local positive_entity = dataset.positive_entities[idx]
        local negative_entity_indices = self:sample_inputs(dataset.positive_entities, positive_entity, num_samples)

        local entity_inputs = {positive_entity}
        local predicate_inputs = {positive_predicate}

        -- unigram sampling probabilities
        local predicate_sampling_probs = self.tensor_type(num_samples + 1) 
        local entity_sampling_probs = self.tensor_type(num_samples + 1)

        local raw_predicate = dataset.raw_positive_predicates[idx]
        local raw_entity = dataset.raw_positive_entities[idx]

        predicate_sampling_probs[1] = dataset.positive_predicate_probabilities[raw_predicate] 
        entity_sampling_probs[1] = dataset.positive_entity_probabilities[raw_entity]

        -- Generate negative samples
        for k = 1, #negative_predicate_indices do 
          local cur_index = negative_predicate_indices[k]
          local cur_raw_predicate = dataset.raw_positive_predicates[cur_index]
          local negative_predicate = dataset.positive_predicates[cur_index]
          table.insert(predicate_inputs, negative_predicate)
          predicate_sampling_probs[k + 1] = dataset.positive_predicate_probabilities[cur_raw_predicate]
        end

        for k = 1, #negative_entity_indices do 
          local cur_index = negative_entity_indices[k]
          local cur_raw_entity = dataset.raw_positive_entities[cur_index]
          local negative_entity = dataset.positive_entities[cur_index]
          table.insert(entity_inputs, negative_entity)
          entity_sampling_probs[k + 1] = dataset.positive_entity_probabilities[cur_raw_entity]
        end


        local curr_loss = self:forward_backward(question_indices, 
                                       predicate_inputs, 
                                       entity_inputs,
                                       desired_predicate_index,
                                       desired_entity_index,
                                       predicate_sampling_probs,
                                       entity_sampling_probs)


        loss = loss + curr_loss
        num_items = num_items + 1
        
      end
      tot_loss = tot_loss + loss
      tot_num_items = tot_num_items + num_items

      if num_items == 0 then 
        num_items = 1
      end
      loss = loss / num_items
      self.grad_params:div(num_items)

      local running_weight_average = torch.abs(self.grad_params):sum() / self.grad_params:size(1)
      print("Grad average " .. running_weight_average)

      dmn.logger:print("Current loss", loss)
      fractional_epoch = self.model_epoch + i / dataset.size
      softmax.run_api:add_point(dmn.constants.TRAIN_INDEX, fractional_epoch, loss)

      currIndex = currIndex + 1
      return loss, self.grad_params
    end
    -- check gradients for lstm layer
    self.optim_method(feval, self.params, self.optim_state)
  end


  average_loss = tot_loss / tot_num_items

  xlua.progress(dataset.size, dataset.size)
  return average_loss
end


-- saves prediction to specified file path
function Attention_Network:save_predictions(predictions_save_path, loss, test_predictions, dataset)
  local predictions_file, err = io.open(predictions_save_path,"w")

  dmn.logger:print('writing predictions to ' .. predictions_save_path)
  --predictions_file:write("LOSS " .. loss .. '\n')
  local sentences = self:get_sentences(test_predictions, dataset)
  for i = 1, #sentences do
    local sentence = sentences[i]
    if i < #sentences then 
      predictions_file:write(sentence .. '\n')
    else
      predictions_file:write(sentence)
    end
  end
  predictions_file:close()
end

function Attention_Network:print_config()
  local num_params = self.params:size(1)
  dmn.logger:printf('%-25s = %d\n', 'num params', num_params)
  dmn.logger:printf('%-25s = %d\n', 'batch size', self.batch_size)
  dmn.logger:printf('%-25s = %s\n', 'optim_method', self.optim_method_string)
  dmn.logger:printf('%-25s = %s\n', 'optim_state', self.optim_state.learningRate)
  dmn.logger:printf('%-25s = %d\n', 'model epoch', self.model_epoch)
  dmn.logger:printf('%-25s = %s\n', 'character level', self.char_level)
  dmn.logger:printf('%-25s = %s\n', 'pad size', self.pad_size)
  for i = 1, #self.layers do
    dmn.logger:printf("\n\n===== dmn.logger:printing config for layer %s =====\n\n", self.layers[i])
    self.layers[i]:print_config()
  end
end

function Attention_Network:get_path(index)
  return softmax.run_api.job_id .. "_Attention_Network_char_level_" .. tostring(self.config.char_level)
  .. "_attention_type_" .. self.config.attention_type 
  .. "_" .. index .. ".th"
end

function Attention_Network:save(path, model_epoch)
  local config = dmn.functions.deepcopy(self.config)
  config.attention_type = config.attention_type or 'fine'
  config.gpu_mode = false
  config.model_epoch = (model_epoch ~= nil) and model_epoch or self.model_epoch

  -- move optim state to double as well
  for k,v in pairs(config.optim_state) do 
    if torch.typename(v) == 'torch.CudaTensor' then 
      config.optim_state[k] = v:double()
    end
  end

  torch.save(path, {
    params = self.params:double(),
    config = config,
    network_type = 'dmn.Attention_Network',
    optim_state = self.optim_state
  })
end

function Attention_Network.load(path)
  local state = torch.load(path)
  local config = state.config 

  config.question_lstm_type = (config.question_lstm_type == nil and 'gf_lstm') or config.question_lstm_type
  config.answer_rerank_criterion = (config.rerank_criterion == nil) and dmn.constants.RERANK_SOFTMAX_CRITERION
          or config.rerank_criterion
  config.attention_type = (config.attention_type == nil) and 'fine' or config.attention_type
  config.dssm_type = (config.dssm_type == nil) and 'hidden_network' or config.dssm_type
  config.version = (config.version == nil) and 'v1' or config.version
  local model = dmn.Attention_Network.new(config)
  model.params:copy(state.params)
  return model
end

