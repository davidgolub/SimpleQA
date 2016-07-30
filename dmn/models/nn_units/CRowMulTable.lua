--[[
  Add a vector to every row of a matrix.
  Input: { [n x m], [m] }
  Output: [n x m]
--]]

local CRowMulTable, parent = torch.class('dmn.CRowMulTable', 'nn.Module')

function CRowMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CRowMulTable:updateOutput(input)
   self.output = input[1] * input[2]
   return self.output
end

function CRowMulTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[1]:resizeAs(input[1])
   self.gradInput[2]:resizeAs(input[2]):zero()

   local grad_inputs = torch.sum(torch.cmul(gradOutput, input[1]))
   self.gradInput[1]:copy(gradOutput * input[2])
   self.gradInput[2]:copy(grad_inputs)

   return self.gradInput
end