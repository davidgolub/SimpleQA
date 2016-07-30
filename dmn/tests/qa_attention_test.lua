--[[
Tests qa attention model: that attention it gives out is legit
]]

require('..')


local cur_model = dmn.Attention_Network.load('test_model.th')

local question = "When was foo born"
local entities = {"FOO", "bar"}
local predicates = {"people/person/location", "people/person/born_in"}

local rankings, likelihoods, question_pred_focuses, question_entity_focuses 
= softmax.qa_api:model_align(cur_model, question, predicates, entities, 2)

print(rankings)
print(likelihoods)
print(question_pred_focuses[1])
print(question_entity_focuses[1])

