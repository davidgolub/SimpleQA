--[[
	Simple entity linker based on freebase API. Uses N-Grams to generate a candidate list of
	entities from freebase. Doesn't use any ML to rerank the candidates
]]

local EntityLinkerAPI = torch.class('softmax.EntityLinkerAPI')

function EntityLinkerAPI:__init(config)
end

-- Generates a list of entity names and predicate names from a question
-- Return results of the form: 
-- results.entity_candidates: entity names
-- results.predicate_candidate: candidate names
function EntityLinkerAPI:link(question, ngrams, num_results)
	assert(question ~= nil, "Must specify question to link")
	assert(ngrams ~= nil, "Must specify number of ngrams to use")
	assert(num_results ~= nil, "Must specify number of results to get back")

	local max_num_facts = 5000
	-- keep track of entity name mapping and entity name topic mapping
	local entity_name_mapping = {}
	local fact_mappings = {}

	local result = {}
	local use_char_level = false

	-- use both tokenized and split text
	local tokenized_text = datasets.tokenize_text(question, use_char_level)
	local split_text = question:split(' ')

	-- construct all ngrams: 
	-- for unigram get all nouns only.
	local all_posibilities = {} --datasets.nlp_util:nouns(question)
	for i = 1, ngrams do 
		local igrams = dmn.functions.ngrams(tokenized_text, i)
		local sgrams = dmn.functions.ngrams(split_text, i)

		dmn.functions.table_concat(all_posibilities, igrams)
		dmn.functions.table_concat(all_posibilities, sgrams)
	end

	local all_entities = {}
	local all_names = {}

	-- get candidates for all entities and names.
	-- gets freebase id and corresponding name
	for i = 1, #all_posibilities do 
		cur_name = all_posibilities[i]
		local entities, names = datasets.freebase_api:entities(cur_name, num_results)
		dmn.functions.table_concat(all_entities, entities)
		dmn.functions.table_concat(all_names, names)
	end

	for i = 1, #all_entities do 
		local cur_entity = all_entities[i]
		local cur_name = all_names[i]
		entity_name_mapping[cur_entity] = cur_name
	end

	local keyed_entities =  dmn.functions.create_dictionary(all_entities)
	local keyed_names = dmn.functions.create_dictionary(all_names)
	local unique_entities = dmn.functions.keys(keyed_entities)

	local all_predicates = {}
	local all_facts = {}

	for i = 1, #unique_entities do 
		if i % 20 == 0 then 
			dmn.logger:print("Extracting facts on entity " .. i 
				.. " from " .. #unique_entities)
		end
		local cur_entity = unique_entities[i]
		local cur_name = entity_name_mapping[cur_entity]
		local predicates, facts = datasets.freebase_api:facts(cur_entity, max_num_facts)
		dmn.functions.table_concat(all_predicates, predicates)

		for i = 1, #predicates do 
			local cur_predicate = predicates[i]
			if facts[i] ~= nil and facts[i].values ~= nil and facts[i].values[1] ~= nil
				and cur_name ~= nil and cur_predicate ~= nil then 
				fact_mappings[cur_name .. " " .. cur_predicate] = facts[i].values[1].text
			end
		end
	end

	local predicate_keys = dmn.functions.create_dictionary(all_predicates)
	local entity_keys = dmn.functions.create_dictionary(all_names)

	local predicate_candidates = dmn.functions.keys(predicate_keys)
	local entity_candidates = dmn.functions.keys(entity_keys)

	return predicate_candidates, entity_candidates, fact_mappings
end

-- Uses specified model to rerank question and return candidate queries (ngrams) to EL system
-- model: attention network to run over question
-- question: question to rerank
-- ngrams: number of ngrams to use
-- beam_size: beam size to use
-- rerank: whether to rerank or not
-- returns: candidate_queries, log_likelihoods
function EntityLinkerAPI:candidate_ngrams(model, question, min_ngrams, max_ngrams, beam_size, rerank)
	assert(rerank ~= nil, "must specify whether to rerank or not")
	assert(model ~= nil, "Must specify model to use for reranking")
	assert(question ~= nil, "Must specify question to use")
	assert(max_ngrams ~= nil, "Must specify max ngrams to use")
	assert(min_ngrams ~= nil, "Must specify min number of ngrams to use")
	assert(beam_size ~= nil, "Must specify beam size to use")
	assert(rerank ~= nil, "Must specify whether to rerank or not")

	local split_text = question:split(' ')
	local dummy_predicate = {'foo'}

	-- construct all ngrams: 
	-- for unigram get all nouns only.
	local all_possibilities = {} --datasets.nlp_util:nouns(question)

	-- get all min ngrams
	cur_min_ngrams = math.min(#split_text, min_ngrams)
	for i = cur_min_ngrams, max_ngrams do 
		local sgrams = dmn.functions.ngrams(split_text, i)
		dmn.functions.table_concat(all_possibilities, sgrams)
	end

	if rerank then  
		local candidate_queries = {}
		local log_likelihoods = {}
		local rankings, likelihoods, _, _ = softmax.qa_api:model_align(model, question, dummy_predicate, all_possibilities, 30)
		for i = 1, 5 do 
			local cur_ranking = rankings[i]
			local cur_likelihood = likelihoods[i]

			local pred_ranking = cur_ranking[1]
			local entity_ranking = cur_ranking[2]

			local predicate = dummy_predicate[pred_ranking]
			local entity = all_possibilities[entity_ranking]

			table.insert(candidate_queries, entity)
			table.insert(log_likelihoods, cur_likelihood)
		end

		return candidate_queries, log_likelihoods
	else
		return {question}, 0.0
	end
end
-- Gets candidate entities, names and types from question based off of n grams
function EntityLinkerAPI:candidate_entities(all_posibilities, num_results)
	assert(all_posibilities ~= nil, "Must specify all queries to use")
	assert(num_results ~= nil, "Must specify number of results for candidate entities")

	local all_entities = {}
	local all_names = {}
	local all_types = {}

	-- get candidates for all entities and names.
	-- gets freebase id and corresponding name
	for i = 1, #all_posibilities do 
		if i % 10 == 0 then 
			print("On " .. i  .. " of " .. #all_posibilities)
		end
		cur_name = all_posibilities[i]
		local entities, names, types = datasets.freebase_api:entities(cur_name, num_results)
		dmn.functions.table_concat(all_entities, entities)
		dmn.functions.table_concat(all_names, names)
		dmn.functions.table_concat(all_types, types)
	end

	return all_entities, all_names, all_types
end

-- Returns a filtered list of entity ids/names/types
function EntityLinkerAPI:filter_entities(model, question, ids, names, types, beam_size)
	assert(model ~= nil, "Must specify model to use")
	assert(question ~= nil, "Must specify question to use")
	assert(ids ~= nil, "Must specify ids to use")
	assert(names ~= nil, "Must specify names to use")
	assert(types ~= nil, "Must specify types to use")
	assert(beam_size ~= nil, "Must specify beam_size to to use")


	local filtered_ids = {}
	local filtered_names = {}
	local filtered_types = {}

	local predicates = {"foo"}
	local predicates = {}

	for i = 1, beam_size do 
		table.insert(predicates, "test")
	end

	-- filter out names
	local unique_names = dmn.functions.unique_values(names)
	local rankings, likelihoods, _, _ = model:align(
											question, 
											predicates, 
											unique_names, 
											beam_size)

	dmn.logger:print("Filtering with beam size " .. beam_size)
	for i = 1, beam_size do 
		local entity_ranking = rankings[i][2]
		local likely_name = unique_names[entity_ranking]

		-- loop through and get all indices that match this entity name
		for j = 1, #ids do 
			local cur_name = names[j]
			if cur_name == likely_name then 
				table.insert(filtered_ids, ids[j])
				table.insert(filtered_names, cur_name)
				table.insert(filtered_types, types[j])
			end

			if #filtered_ids > beam_size then break end
		end
	end


	return filtered_ids, filtered_names, filtered_types
end

-- Generates a list of entity names and predicate names from a question
-- Return results of the form: 
-- results.entity_candidates: entity names
-- results.predicate_candidate: candidate names
function EntityLinkerAPI:candidate_facts(all_entities, all_names)
	assert(all_entities ~= nil, "Must specify entities to prune")
	assert(all_names ~= nil, "Must specify names to prune")

	-- keep number of facts local to this function for now
	local num_facts = 500

	-- keep track of entity name mapping and entity name topic mapping
	local entity_name_mapping = {}
	local fact_mappings = {}
	local result = {}

	for i = 1, #all_entities do 
		local cur_entity = all_entities[i]
		local cur_name = all_names[i]
		entity_name_mapping[cur_entity] = cur_name
	end

	local all_predicates = {}
	local all_facts = {}

	for i = 1, #all_entities do 
		if i % 20 == 0 then 
			dmn.logger:print("Extracting facts on entity " .. i 
				.. " from " .. #all_entities)
		end
		local cur_entity = all_entities[i]
		local cur_name = entity_name_mapping[cur_entity]
		local predicates, facts = datasets.freebase_api:facts(cur_name, num_facts)
		dmn.functions.table_concat(all_predicates, predicates)

		for i = 1, #predicates do 
			local cur_predicate = predicates[i]
			if facts[i] ~= nil and facts[i].values ~= nil and facts[i].values[1] ~= nil
				and cur_name ~= nil and cur_predicate ~= nil then 
				fact_mappings[cur_name .. " " .. cur_predicate] = facts[i].values[1].text
			end
		end
	end

	local predicate_candidates = dmn.functions.unique_values(all_predicates)
	local entity_candidates = dmn.functions.unique_values(all_names)

	return predicate_candidates, entity_candidates, fact_mappings
end