--[[

  Hidden Dummy Layer: Just feeds in zeros
--]]

local HiddenDummyLayer, parent = torch.class('dmn.HiddenDummyLayer', 'dmn.HiddenLayer')

function HiddenDummyLayer:__init(config)
   parent.__init(self, config)
   self.hidden_init = self:new_hidden_activations()
end

function HiddenDummyLayer:new_hidden_activations() 
  if self.num_layers == 1 then 
    return {torch.zeros(self.proj_dim), torch.zeros(self.proj_dim)}
  else 
    local modules = {}
    for i = 1, self.num_layers do
      table.insert(modules, {torch.zeros(self.proj_dim), torch.zeros(self.proj_dim)})
      table.insert(modules, {torch.zeros(self.proj_dim), torch.zeros(self.proj_dim)})
    end
    return modules
  end
end

-- Returns all of the weights of this module
function HiddenDummyLayer:getWeights()
end

function HiddenDummyLayer:getModules() 
  return {}
end

-- Sets gpu mode
function HiddenDummyLayer:set_gpu_mode()
end

function HiddenDummyLayer:set_cpu_mode()
end

-- Enable Dropouts
function HiddenDummyLayer:enable_dropouts()
end

-- Disable Dropouts
function HiddenDummyLayer:disable_dropouts()
end

-- Does a single forward step of concat layer, concatenating
-- 
function HiddenDummyLayer:forward(input_vals)
   return self.hidden_init
end

-- Does a single backward step of project layer
-- image_feats: input into hidden projection error
-- cell_errors: error of all hidden, cell units of lstm with respect to input
function HiddenDummyLayer:backward(input_vals, cell_errors)
end

-- Returns size of outputs of this combine module
function HiddenDummyLayer:getOutputSize()
end

function HiddenDummyLayer:getParameters()
end

-- zeros out the gradients
function HiddenDummyLayer:zeroGradParameters() 
end

function HiddenDummyLayer:normalizeGrads(batch_size)
end

