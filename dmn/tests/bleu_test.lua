--[[
Tests whether BLEU works properly
]]

require('..')

local prediction_path = '../datasets/Captioning/piano/test/predictions.txt'
local gold_path = '../datasets/Captioning/piano/test/captions.txt'

local bleu_results = dmn.eval_functions.bleu(prediction_path, gold_path)
for cur_result in bleu_results do
	dmn.logger:print(cur_result)
end
