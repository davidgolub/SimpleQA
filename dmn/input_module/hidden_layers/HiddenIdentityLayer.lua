--[[

  Hidden Identity Layer: Projects image input into projection dimension twice. For feeding in
  image input into lstm
--]]

local HiddenIdentityLayer, parent = torch.class('dmn.HiddenIdentityLayer', 'dmn.HiddenLayer')

function HiddenIdentityLayer:__init(config)
   parent.__init(self, config)
end

function HiddenIdentityLayer:getModules() 
  return {}
end

-- Sets gpu mode
function HiddenIdentityLayer:set_gpu_mode()
  self.gpu_mode = true
end

function HiddenIdentityLayer:set_cpu_mode()
  self.gpu_mode = false
end

-- Enable Dropouts
function HiddenIdentityLayer:enable_dropouts()
end

-- Disable Dropouts
function HiddenIdentityLayer:disable_dropouts()
end

-- Does a single forward step of identity layer, returns inputs
-- 
function HiddenIdentityLayer:forward(inputs)
   return inputs
end

-- Does a single backward step of project layer
-- image_feats: input into hidden projection error
-- cell_errors: error of all hidden, cell units of lstm with respect to input
function HiddenIdentityLayer:backward(inputs, cell_errors)
   assert(inputs ~= nil)
   assert(cell_errors ~= nil)
   parent:backward(inputs, cell_errors, self.gpu_mode)
   
   return cell_errors
end

-- Returns size of outputs of this combine module
function HiddenIdentityLayer:getOutputSize()
  return self.mem_dim * self.num_layers
end

function HiddenIdentityLayer:getParameters()
  return {}
end

-- zeros out the gradients
function HiddenIdentityLayer:zeroGradParameters() 
end

function HiddenIdentityLayer:normalizeGrads(batch_size)
end

