--[[
Eval functions
]]

local EvalFunctions = torch.class('dmn.eval_functions')

-- Calculates BLEU scores on train, test and val sets
function EvalFunctions.bleu(predictions_path, gold_path)
  assert(predictions_path ~= nil, "Must specify predictions path to use")
  assert(gold_path ~= nil, "Must specify gold path to use")

  dmn.logger:print("Evaluating bleu score")

  -- First makes sure that we have no error
  local predict_lines = datasets.read_line_data(predictions_path)
  local gold_lines = datasets.read_line_data(gold_path)

  -- check if gold lines has a \t character, in which case we want to split the gold captions for the bleu score testing

  local cleaned_pred_path = "../tmp/tmp_pred" .. sys.clock() .. ".txt"
  local cleaned_gold_path = "../tmp/tmp_gold" .. sys.clock() .. ".txt"

  local cleaned_predict_lines = {}
  local cleaned_gold_lines = {}

  for i = 1, #predict_lines do
  	if predict_lines[i] ~= dmn.constants.ERROR_CONSTANT then 
      local lowered_prediction = predict_lines[i]:lower()
      local trimmed_prediction = dmn.functions.string_trim(lowered_prediction)
  		table.insert(cleaned_predict_lines, trimmed_prediction)
      -- able to handle both "\t" and non-tabbed data
      local gold_captions = string.split(gold_lines[i], "\t")
      for i = 1, #gold_captions do 
        if cleaned_gold_lines[i] == nil then 
          cleaned_gold_lines[i] = {}
        end
        local lowered_gold_caption = gold_captions[i]:lower()
        local trimmed_gold_caption = dmn.functions.string_trim(lowered_gold_caption)
        table.insert(cleaned_gold_lines[i], trimmed_gold_caption)
      end
  	end
  end

  local start_count = #cleaned_gold_lines[1]

  for i = 1, #cleaned_gold_lines do 
    local cur_index = i - 1
    local cur_save_path = cleaned_gold_path .. cur_index
    local cur_lines = cleaned_gold_lines[i]

    -- make sure that we only save scores that have equal number of lines to the start (otherwise BLEU messes up)
    if #cur_lines == start_count then 
      datasets.save_line_data(cur_lines, cur_save_path)
    end
  end
  datasets.save_line_data(cleaned_predict_lines, cleaned_pred_path)

  local results = io.popen("../dmn/eval/run_bleu.sh " ..  cleaned_pred_path .. ' ' .. cleaned_gold_path)
  local bleu_scores = results:lines()
  
  return bleu_scores
end