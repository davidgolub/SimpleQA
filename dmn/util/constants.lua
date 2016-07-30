--[[
  Constants
]]

local constants = torch.class('dmn.constants')

constants.TRAIN_INDEX = 0
constants.VAL_INDEX = 1
constants.TEST_INDEX = 2
constants.TRAIN_FRACTION = 0.6

-- error constant
constants.ERROR_CONSTANT = 'ERROR OCCURED'

-- number of retries for request
constants.NUM_RETRIES = 5

-- Constants for context image captioning
constants.PER = 'PERSON'
constants.ORG = 'ORGANIZATION'
constants.MISC = 'MISCELLANEOUS'
constants.LOC = 'LOCATION'
constants.YEAR = 'TIMEOFYEAR'
constants.MONTH = 'TIMEMONTH'

constants.NO_CONTEXT = '<NO_CONTEXT>'

-- Authentication constants
constants.USERNAME = 'foo'
constants.PASSWORD = 'bar'

-- Local constants
constants.LOCAL_MODEL_DIR = 'models/'

-- Cloud constants
constants.CLOUD_MODEL_DIR = 'softmax_models'
constants.CLOUD_LOG_DIR = 'softmax_logs'
constants.CLOUD_PREDICTIONS_DIR = 'softmax_predictions'

-- Directory constants
constants.MAIN_PATH = '../'

-- URL constants for logs
constants.JOB_ENDPOINT = 'http://ec2-52-33-179-156.us-west-2.compute.amazonaws.com:8000/api/v1/'
--'http://127.0.0.1:8000/api/v1/'
--'http://127.0.0.1:8000/api/' 
-- For tokenizing
constants.CHAR_LEVEL = true
constants.WORD_LEVEL = false
constants.NO_TOKENIZATION = 2

-- For loading model
constants.IMAGE_CLASSIFICATION_SHOP_APP_TAG_ID = 3609
constants.IMAGE_CLASSIFICATION_SHOP_APP_COLOR_ID = 3544
constants.CONTEXT_DESCRIBE_ID = 1
constants.DMN_ID = 1
constants.DSSM_ID = 1
constants.CONTEXT_DSSM_ID = 1

-- For classification
constants.CLASSIFY_SINGLE_CLASS = 'CLASSIFY_SINGLE_CLASS'
constants.CLASSIFY_MULTI_CLASS = 'CLASSIFY_MULTI_CLASS'
constants.CLASSIFY_TRANSFER_LEARNING = 'CLASSIFY_TRANSFER_LEARNING'

constants.RERANK_NCE_CRITERION = 'RERANK_NCE_CRITERION'
constants.RERANK_SOFTMAX_CRITERION = 'RERANK_SOFTMAX_CRITERION'

-- For dataset types
constants.DATASET_VALIDATION_TYPE = 'DATASET_VALIDATION_TYPE'
constants.DATASET_TRAINING_TYPE = 'DATASET_TRAINING_TYPE'
constants.DATASET_TESTING_TYPE = 'DATASET_TESTING_TYPE'

-- returns network from string representation, useful for loading models
function constants.get_network(string_name)
	assert(string_name ~= nil, "Must specify name of network")
	local net 
	if string_name == 'dmn.Attention_Network' then
		return dmn.Attention_Network
	elseif string_name == 'dmn.Captioner_Network' then
		return dmn.Captioner_Network
	elseif string_name == 'dmn.Context_Captioner_Network' then 
		return dmn.Context_Captioner_Network 
	elseif string_name == 'dmn.Context_DSSM_Network' then
		return dmn.Context_DSSM_Network 
	elseif string_name == 'dmn.DMN_Network' then
		return dmn.DMN_Network 
	elseif string_name == 'dmn.Image_Classification_Network' then
		return dmn.Image_Classification_Network 
	elseif string_name == 'dmn.Image_Classification_Network' then
		return dmn.Image_Classification_Network 
	else 
		error("Invalid network type " .. string_name .. " specified")
	end
end