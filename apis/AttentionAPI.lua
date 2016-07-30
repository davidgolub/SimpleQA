--[[

  QA API: Aligns a question to most likely predicates and entities from a structured knowledge base.
  Given the question, produces a ranked list of the most likely {predicate, entity pairs}
--]]

local QA_API = torch.class('softmax.AttentionAPI')

function QA_API:__init(config)
   -- load modelf
   self.config = config
   self.model_path = config.model_path or "models/cm_5.th"
   self.models = {}
   self.path_to_idx = {}

   --self.model = self:load_model(self.model_path)
end

-- Loads model from current model path and sets the network to load from that model
-- Adds model to our index
function QA_API:load_model_from_path(model_path)
  assert(model_path ~= nil, "Must specify path to load model from")
  local model = dmn.Attention_Network.load(model_path)

  -- dmn.logger:print configurations
  model:print_config()

  -- set it to predict mode
  model:disable_dropouts()

  dmn.logger:print("Adding model at index" .. #self.models)

  table.insert(self.models, model)
  self.path_to_idx[model_path] = #self.models

  return model
end

function QA_API:load_model_from_binary(model)
  assert(model ~= nil, "Must specify model binary to load from")

  -- dmn.logger:print configurations
  model:print_config()

     -- set it to predict mode
  model:disable_dropouts()

  table.insert(self.models, model)
  self.path_to_idx[model:get_path(model.model_epoch)] = #self.models
  return model
end

-- Reranks predicates and entities for a question
-- returns: rankings, likelihoods, question predicate/entity focus
function QA_API:align(question, predicates, entities, beam_size)
  assert(question ~= nil, "Must specify question to use for Attention Network")
  assert(predicates ~= nil, "Must specify candidate predicates to use for Attention Network")
  assert(entities ~= nil, "Must specify candidate entities to use for Attention Network")
  assert(beam_size ~= nil, "Beam size must be specified")
 
  -- get all results from model ensemble and aggregate them

  local ranking_to_idx = {}

  local final_rankings = {}
  local final_likelihoods = {}

  local returned_rankings = {}
  local returned_likelihoods = {}

  local question_pred_focuses, question_entity_focuses

  if #predicates == 0 then
    for i = 1, beam_size do
      table.insert(predicates, "FOO")
    end
  end
  if #entities == 0 then
    for i = 1, beam_size do
      table.insert(entities, "BAR")
    end
  end

  assert(#predicates > 0, "Must give positive # of predicates")
  assert(#entities > 0, "Must give positive # of entities")

  for i = 1, #self.models do 
    local cur_model = self.models[i]
    local rankings, likelihoods, cur_question_pred_focuses, cur_question_entity_focuses = 
      cur_model:predict(question, predicates, entities, beam_size)

    question_pred_focuses = cur_question_pred_focuses
    question_entity_focuses = cur_question_entity_focuses
    --assert(false, json.encode(question_entity_focuses))

    cur_size = 0
    for j = 1, #rankings do 
      local ranking = rankings[j]

      concated_ranking = table.concat(ranking)

      local cur_idx = cur_size + 1
      if ranking_to_idx[concated_ranking] ~= nil then 
        cur_idx = ranking_to_idx[concated_ranking]
      else 
        ranking_to_idx[tostring(concated_ranking)] = cur_size 
        cur_size = cur_size + 1
      end

      local cur_cost = (final_rankings[cur_idx] == nil) and 0 or final_rankings[cur_idx][1]
      final_rankings[cur_idx] = {cur_cost + likelihoods[j], ranking}
    end
  end

  reranked_values = dmn.functions.topk(final_rankings, #final_rankings)


  for i = 1, #reranked_values do
    local log_loss = reranked_values[i][1]
    local ranking = reranked_values[i][2]

    table.insert(returned_rankings, ranking)
    table.insert(returned_likelihoods, log_loss)
  end

  return returned_rankings, returned_likelihoods, question_pred_focuses, question_entity_focuses
end

-- Reranks predicates/entities for a single model
function QA_API:model_align(model, question, predicates, entities, beam_size)
  assert(question ~= nil, "Must specify question to use for Attention Network")
  assert(predicates ~= nil, "Must specify candidate predicates to use for Attention Network")
  assert(entities ~= nil, "Must specify candidate entities to use for Attention Network")
  assert(beam_size ~= nil, "Beam size must be specified")
  assert(model ~= nil, "Model to predict with must not be null")

  local rankings, likelihoods, question_pred_focuses, question_entity_focuses = 
    model:predict(question, predicates, entities, beam_size)

  return rankings, likelihoods, question_pred_focuses, question_entity_focuses
end


-- returns best predicate and name for question
function QA_API:answer_v2(question, min_ngrams, max_ngrams, num_results, beam_size, rerank)
  assert(question ~= nil, "Must specify question to answer")
  assert(min_ngrams ~= nil, "Must specify number of ngrams to use")
  assert(max_ngrams ~= nil, "Must specify max number of ngrams to use")
  assert(beam_size ~= nil, "Must specify beam size to use for reranking")
  assert(num_results ~= nil, "Must specify number of results to request from api")
  assert(rerank ~= nil, "Must specify whether to rerank ngrams or use raw ones")

  local num_facts = 10000

  dmn.logger:print("Answering question " .. question)
  dmn.logger:print("Getting all ids")
  all_possibilities, log_likelihoods = softmax.entity_linker_api:candidate_ngrams(
    softmax.qa_api.models[1],
    question, 
    min_ngrams, 
    max_ngrams,
    beam_size,
    rerank)

  -- take the top one
  local query = {all_possibilities[1]}
  local ids, all_names, all_types = 
  softmax.entity_linker_api:candidate_entities(query, num_results)

  dmn.logger:print("Generating candidates")
  --[[
  candidate_predicates = {
  '/people/person/composer',
  '/song/writer/author',
  '/people/person/place_of_birth'
  }
  candidate_entities = {'Foo bar',
'Bar foo',
'Bye bye love',
'song'}
]]

candidate_ids = {}
fact_mappings = {}
  local candidate_predicates, candidate_entities, candidate_ids, fact_mappings = 
    datasets.freebase_api:facts(ids, num_facts)

  dmn.logger:print("Reranking candidates")

  local rankings, likelihoods, question_pred_focuses, question_entity_focuses 
        = softmax.qa_api:align(
                              question, 
                              candidate_predicates, 
                              candidate_entities, 
                              beam_size
                              )

  likely_predicates = {}
  likely_names = {}
  likely_facts = {}
  likely_ids = {}

  for i = 1, #rankings do 
    local ranking = rankings[i]
    local entity_ranking = ranking[2]
    local predicate_ranking = ranking[1]

    local likely_id = candidate_ids[entity_ranking]
    local likely_name = candidate_entities[entity_ranking]
    local likely_predicate = candidate_predicates[predicate_ranking]
    local likely_fact = fact_mappings[likely_name .. " " .. likely_predicate]

    if likely_fact == nil then 
      --dmn.logger:print("NULL FACT")
      likely_fact = "NO FACT"
    end
    table.insert(likely_ids, likely_id)
    table.insert(likely_predicates, likely_predicate)
    table.insert(likely_names, likely_name)
    table.insert(likely_facts, likely_fact)
  end

  return likely_predicates, likely_names, likely_ids, likely_facts, fact_mappings,
  likelihoods, question_pred_focuses, question_entity_focuses, 
  candidate_predicates, candidate_entities
end

-- returns best predicate and name for question
function QA_API:answer_v3(question, min_ngrams, max_ngrams, num_results, beam_size, rerank)
  assert(question ~= nil, "Must specify question to answer")
  assert(min_ngrams ~= nil, "Must specify number of ngrams to use")
  assert(max_ngrams ~= nil, "Must specify max number of ngrams to use")
  assert(beam_size ~= nil, "Must specify beam size to use for reranking")
  assert(num_results ~= nil, "Must specify number of results to request from api")
  assert(rerank ~= nil, "Must specify whether to rerank ngrams or use raw ones")

  local num_facts = 10000

  dmn.logger:print("Answering question " .. question)
  dmn.logger:print("Getting all ids")
  all_possibilities, log_likelihoods = softmax.entity_linker_api:candidate_ngrams(
    softmax.qa_api.models[1],
    question, 
    min_ngrams, 
    max_ngrams,
    beam_size,
    rerank)

  -- take the top one
  local query = {all_possibilities[1]}
  local ids, all_names, all_types = 
  softmax.entity_linker_api:candidate_entities(query, num_results)

  dmn.logger:print("Generating candidates")
  local candidate_predicates, candidate_entities, fact_mappings = 
    datasets.freebase_api:facts(ids, num_facts)

  dmn.logger:print("Reranking candidates")

  local entity_rankings, likelihoods, question_pred_focuses, question_entity_focuses 
        = softmax.qa_api:align(
                              question, 
                              {'foo'}, 
                              candidate_entities, 
                              beam_size
                              )

  local predicate_rankings, likelihoods, question_pred_focuses, question_entity_focuses 
        = softmax.qa_api:align(
                              question, 
                              candidate_predicates, 
                              candidate_entities, 
                              beam_size
                              )
  likely_predicates = {}
  likely_names = {}
  likely_facts = {}

  for i = 1, #entity_rankings do 
    local entity_ranking = entity_rankings[i][2]
    local predicate_ranking = predicate_rankings[i][1]

    local likely_name = candidate_entities[entity_ranking]
    local likely_predicate = candidate_predicates[predicate_ranking]
    local likely_fact = fact_mappings[likely_name .. " " .. likely_predicate]

    if likely_fact == nil then 
      --dmn.logger:print("NULL FACT")
      likely_fact = "NO FACT"
    end
    table.insert(likely_predicates, likely_predicate)
    table.insert(likely_names, likely_name)
    table.insert(likely_facts, likely_fact)
  end

  return likely_predicates, likely_names, likely_facts, fact_mappings,
  likelihoods, question_pred_focuses, question_entity_focuses, 
  candidate_predicates, candidate_entities
end

function QA_API:answer(question, ngrams, filter_beam_size, rank_beam_size, query_beam_size, rerank)
  assert(question ~= nil, "Must specify question to answer")
  assert(ngrams ~= nil, "Must specify number of ngrams to use")
  assert(filter_beam_size ~= nil, "Must specify filter beam size to use")
  assert(rank_beam_size ~= nil, "Must specify rank beam size to use")
  assert(query_beam_size ~= nil, "Must specify query beam size to use")
  assert(rerank ~= nil, "Must specify whether to rerank ngrams or use raw ones")

  dmn.logger:print("Answering question " .. question)
  dmn.logger:print("Getting all ids")
  all_possibilities, log_likelihoods = softmax.entity_linker_api:candidate_ngrams(
    softmax.qa_api.models[1],
    question, 
    ngrams, 
    query_beam_size,
    rerank)
  local ids, all_names, all_types = softmax.entity_linker_api:candidate_entities(all_possibilities, 500)

  dmn.logger:print("Filtering ids and names from")
  local filtered_ids, filtered_names, filtered_types = 
    softmax.entity_linker_api:filter_entities(
      softmax.qa_api, 
      question,
      ids, 
      all_names, 
      all_types,
      query_beam_size)

  dmn.logger:print("Generating candidates")
  local candidate_predicates, candidate_entities, fact_mappings = 
    softmax.entity_linker_api:candidate_facts(filtered_ids, filtered_names)


  dmn.logger:print("Reranking candidates")

  local rankings, likelihoods, question_pred_focuses, question_entity_focuses 
        = softmax.qa_api:align(
                              question, 
                              candidate_predicates, 
                              candidate_entities, 
                              rank_beam_size
                              )

  likely_predicates = {}
  likely_names = {}
  likely_facts = {}

  for i = 1, #rankings do 
    local ranking = rankings[i]
    local entity_ranking = ranking[2]
    local predicate_ranking = ranking[1]

    local likely_name = candidate_entities[entity_ranking]
    local likely_predicate = candidate_predicates[predicate_ranking]
    local likely_fact = fact_mappings[likely_name .. " " .. likely_predicate]

    table.insert(likely_predicates, likely_predicate)
    table.insert(likely_names, likely_name)
    table.insert(likely_facts, likely_fact)
  end

  return likely_predicates, likely_names, likely_facts, 
  likelihoods,
  question_pred_focuses, question_entity_focuses, 
  candidate_predicates, candidate_entities
end


