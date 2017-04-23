require('../../dmn')

cmd = torch.CmdLine()
cmd:option('-start_index', 1, 'start index')
cmd:option('-beam_size', 1000, 'beam size')
cmd:option('-num_results', 1000, 'num results')
cmd:option('-min_ngrams', 5, 'min ngrams')
cmd:option('-max_ngrams', 8, 'max ngrams')
cmd:option('-freebase_type', "Freebase_2M", "Freebase 2M")
cmd:text()

-- parse input params
params = cmd:parse(arg)
local min_ngrams = params.min_ngrams 
local max_ngrams = params.max_ngrams 
local num_results = params.num_results
local beam_size = params.beam_size 
local freebase_type = params.freebase_type
local rerank = false

-- Create a new job for these items--log the stuff
print("SimpleQuestions evaluation " .. freebase_type .. " " .. num_results)
-- model we use for testing
local model_paths = {
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_23.th",
"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_fixed_35.th",
--"../dmn/trained_models/2531_Attention_Network_char_level_false_attention_type_coarse_fixed_9.th"
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_25.th",
--"../dmn/trained_models/Attention_Network_char_level_false_attention_type_coarse_17.th"
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_fixed_20.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_fixed_26.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_fixed_27.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_21.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_coarse_19.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_fine_19.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_fine_17.th",
}

local stringed_models = table.concat(model_paths, " ")

--"../dmn/trained_models/Attention_Network_char_level_false_attention_type_coarse_17.th",
--"../dmn/trained_models/Attention_Network_char_level_true_attention_type_dropout_13.th"}

for i = 1, #model_paths do 
	local model_path = model_paths[i]
	softmax.qa_api:load_model_from_path(model_path)
end

local questions = datasets.read_line_data("../datasets/SimpleQuestions/test/questions.txt")
local predicates = datasets.read_line_data("../datasets/SimpleQuestions/test/positive_predicates.txt")
local entities = datasets.read_line_data("../datasets/SimpleQuestions/test/positive_entities.txt")

local entity_ids = datasets.read_tabbed_data("../datasets/SimpleQuestions/annotated_fb_data_test.txt")
local facts = datasets.read_line_data("../datasets/SimpleQuestions/test/object_names.txt")

local total_seen = 0
local total_corr = 0
local total_pred_corr = 0
local total_fact_corr = 0

local predicate_corr = 0
local entity_corr = 0

predicates_path = '../datasets/SimpleQuestions/test/predictions_predicates_' .. params.start_index 
.. '.txt'

entities_path = '../datasets/SimpleQuestions/test/predictions_entities_' .. params.start_index 
.. '.txt'

facts_path = '../datasets/SimpleQuestions/test/predictions_facts_' .. params.start_index 
.. '.txt'

results_path = '../datasets/SimpleQuestions/test/results.txt'

-- test predictions
local predicates_file = 
	io.open(predicates_path, 'w')

local entities_file = 
	io.open(entities_path, 'w')

local facts_file = 
	io.open(facts_path, 'w')

local results_file = 
	io.open(results_path, 'w')

for j = params.start_index, #questions do 
	dmn.logger:print("On index " .. j)
	local question = dmn.functions.string_trim(questions[j])
	
	--assert(false)
	local best_predicates, best_entities, best_ids, best_facts, fact_mappings, likelihoods,
 	candidate_predicates, candidate_entities = 
	softmax.qa_api:answer_v2(question, min_ngrams, max_ngrams, num_results, beam_size, rerank)

	local best_entity 
	local best_predicate
	local best_id
	local best_fact = nil 

	local cur_index = 1
	local best_entity = best_entities[1]

	while best_fact == nil or best_fact == "NO FACT"
		and cur_index <= beam_size do 
		best_id = best_ids[cur_index]
		best_entity = best_entities[cur_index]
		best_predicate = best_predicates[cur_index]
		best_fact = best_facts[cur_index]
		print(best_entity .. " " .. best_predicate .. " " .. best_fact) 
		cur_index = cur_index + 1
	end

	if cur_index > beam_size then 
		dmn.logger:print("Could not find valid fact!")
		best_id = (best_ids[cur_index] == nil) and "NO_ID" or best_ids[cur_index]
		best_predicate = best_predicates[1]
		best_entity = best_entities[1]
		best_fact = best_facts[1]
	end
	
	print(best_entity .. " " .. best_predicate .. " " .. best_fact) 

	if best_predicate == predicates[j] then 
		predicate_corr = predicate_corr + 1 
	end

	if best_entity == entities[j] then 
		entity_corr = entity_corr + 1 
	end

	local ids_match = best_id == actual_id
	local stripped_name = dmn.functions.strip_accents(entities[j])

	local parse_matches = string.lower(best_predicate) == string.lower(predicates[j]) and 
			string.lower(best_entity) == string.lower(stripped_name)

	if parse_matches then 
		total_corr = total_corr + 1
		if facts_match then 
			dmn.logger:print("Facts match ")
			total_fact_corr = total_fact_corr + 1
		end
	end

	total_seen = total_seen + 1
	
	local index = math.floor(datasets.freebase_api.num_calls / 90000 + 1)
	local num_calls = datasets.freebase_api.num_calls

	local msg = "Index Tot/Fact/Names/ pred/entity accuracy " .. j .. " " .. total_corr / total_seen .. " "
	    .. total_fact_corr / total_seen .. " " .. total_pred_corr / total_seen
		.. " " .. predicate_corr / total_seen .. " " .. entity_corr / total_seen .. " "
		.. question .. " " .. best_predicate .. " " .. best_entity 
		.. " FACT:" .. best_fact .. " " .. num_calls .. " " .. best_id

    dmn.logger:print(msg)
	
	results_file:write(msg .. "\n")
	entities_file:write(best_entity .. "\n")
	predicates_file:write(best_predicate .. "\n")
	facts_file:write(best_fact .. "\n")

	entities_file:flush()
	predicates_file:flush()
	facts_file:flush()
	results_file:flush()
end

entities_file:close()
predicates_file:close()
facts_file:close()
results_file:close()
