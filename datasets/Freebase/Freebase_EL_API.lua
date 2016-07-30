--[[

  FreebaseAPI: Get candidate entities and facts from freebase using elasticsearch
  
--]]

local FreebaseAPI = torch.class('datasets.Freebase_EL_API')

function FreebaseAPI:__init(config)
   self.base_url = 'http://localhost:5000'

   self.entity_freebase_endpoint = self.base_url .. "/api/v1/freebase/name?query=%s&num_results=%d&remove_stopwords=True"--filter=suggest&key=%s"
   self.name_freebase_endpoint = self.base_url .. "/api/v1/freebase/name?query=%s&num_results=%d&remove_stopwords=True"
   self.topic_freebase_endpoint = self.base_url .. "/api/v1/freebase/fact?topic_ids=%s&num_results=%d&remove_stopwords=True"

   self.num_calls = 0
end

-- Keep track of number of calls
function FreebaseAPI:increment_num_calls()
  self.num_calls = self.num_calls + 1
end

-- Returns freebase ids of all entities that match name
function FreebaseAPI:entities(query, num_results)
  assert(query ~= nil, "topic id must not be null")
  assert(num_results ~= nil, "Number of results must not be null")

  self:increment_num_calls()

  local encoded_query = dmn.io_functions.url_encode(query)

  -- get features from url
  local extract_url = string.format(self.entity_freebase_endpoint, 
    encoded_query, 
    num_results)
  --dmn.logger:print("Extracting from url " .. extract_url)

  local num_tries = 0

  -- extract topics
  local html  = dmn.io_functions.http_request(extract_url)
  local json_vals

-- Hacky error handling for image load issues
  dmn.io_functions.trycatch(
    function() 
      json_vals = dmn.io_functions.json_decode(html)
    end,
    function(err)
      --dmn.logger:print("ERROR OCCURED LOADING IMAGE")
      print("ERROR DECODING JSON " .. err .. tostring(html))
      json_vals = nil
  end)

  if json_vals == nil or json_vals["result"] == nil then 
    name = "ERROR OCCURED"
    dmn.logger:print("ERROR REQUESTING")
    dmn.logger:print("ERROR HTTPS REQUESTING " .. html)
  end

  while (json_vals == nil or json_vals["result"] == nil) and (num_tries < dmn.constants.NUM_RETRIES) do 
    -- try again
    local msg = "Error requesting url " .. extract_url .. " retrying again"  .. num_tries
    dmn.logger:print(msg)


    num_tries = num_tries + 1
    -- request url again
    html  = dmn.io_functions.http_request(extract_url)
      -- Hacky error handling for image load issues
    dmn.io_functions.trycatch(
      function() 
        json_vals = dmn.io_functions.json_decode(html)--lua_json.parse(html)
      end,
      function(err)
        --dmn.logger:print("ERROR OCCURED LOADING IMAGE")
        softmax.run_api:add_error_log("ERROR DECODING JSON " .. err .. tostring(html))
        json_vals = nil
    end)
  end

  local names = json_vals["result"]
  local ids = {}
  local entity_names = {}
  local entity_types = {}

  for i = 1, #names do
    local cur_entity = names[i]
    local cur_id = cur_entity["freebase_id"]
    local cur_name = cur_entity["freebase_name"]
    local cur_types = {"TYPES_NOT_SUPPORTED"}

    table.insert(entity_names, cur_name)
    table.insert(ids, cur_id)
    table.insert(entity_types, cur_types)
  end

  return ids, entity_names, entity_types
end

-- Gets facts about a topic id
-- image_url: URL to extract images from, must be not null
-- returns: Torch double tensor of size 1024 with googlenet features from image url
function FreebaseAPI:facts(topic_ids, num_results)
  assert(topic_ids ~= nil, "topic ids must not be null")
  assert(num_results ~= nil, "Number of results must not be null")
  
  self:increment_num_calls()

  csv_topic_ids = table.concat(topic_ids, ',')
-- get features from url
  local extract_url = string.format(self.topic_freebase_endpoint, 
    csv_topic_ids, 
    num_results)


  names = {}
  topics = {}
  ids = {}
  facts = {}
  -- Hacky error handling for web request issues
  dmn.io_functions.trycatch(
    function() 

      -- extract topics
      local html = dmn.io_functions.http_request(extract_url)
      local json_vals = dmn.io_functions.json_decode(html)
      local topics_desc = json_vals["result"]

      -- insert them
      for k,v in pairs(topics_desc) do   
        if not dmn.functions.string_starts(k, "/type/object") then 
          local id = v['src_freebase_id']
          local name = v['src_freebase_name']
          local topic = v['pred_freebase_name']
          local tgt_freebase_id = v['tgt_freebase_id']
          table.insert(names, name)
          table.insert(ids, id)
          table.insert(topics, topic)
          facts[name .. " " .. topic] = tgt_freebase_id
        end
      end
    end,

    function(err)
      dmn.logger:print("ERROR OCCURED requesting url " .. extract_url .. " " .. err)

      topics = {"ERROR OCCURED " .. topic_id}
    end)
  return topics, names, ids, facts
end
