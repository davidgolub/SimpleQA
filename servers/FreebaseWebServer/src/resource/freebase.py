from flask import jsonify
from flask.ext.restful import Resource
from flask import request 
from flask_restful import reqparse

import re
from model.abc import db
from model import User
import config
from util.freebase import FreebaseObject, FreebaseFact 
import util.tokenizer as tokenizer


class FreebaseNameAPI(Resource):
	# Gets all of the names for specified query
    def get(self):
    	# Get query and number of desired results
        #try:
        query = request.args.get('query')
        remove_stopwords = request.args.get('remove_stopwords')

        # Replace '\''
        query = query.replace('\'', " ")
        query = tokenizer.replace_accents(query.encode('utf-8'))

        removed_stopwords_query = tokenizer.remove_stopwords(query)
    	num_results = request.args.get('num_results')

    	fb_helper = config.FREEBASE_HELPER

    	names, num_items = fb_helper.get_names(removed_stopwords_query, num_results)
    	jsoned_names = [{'freebase_name': name.freebase_name, 'freebase_id': name.freebase_id} for name in names]
        topic_ids = [name.freebase_id for name in names]
        topic_names = [name.freebase_name for name in names]
        
        # Only keep names that have an alias in the sentence
        cleaned_names = tokenizer.clean_name_arr(topic_names, query)
        removed_substring_names = tokenizer.remove_substrings_arr(cleaned_names)
        cleaned_jsoned_names = filter(lambda name: name['freebase_name'] \
            in set(removed_substring_names), jsoned_names)
        return jsonify(result=cleaned_jsoned_names,cleaned_names=cleaned_names,raw_names=jsoned_names,num_items=num_items)
        #except Exception as e:
        #    print("Error requesting %s" % e)
        #    jsoned_names = [{'freebase_name': "NONE", 'freebase_id': '/m/test'}]
        #    return jsonify(result=jsoned_names,num_items=num_items)

class FreebaseFactAPI(Resource):
	# Gets all of the names for specified query
    def get(self):
    	# Get topic_id and number of desired results
    	topic_ids = request.args.get('topic_ids').split(',')
    	num_results = request.args.get('num_results')
        num_results_per_topic = request.args.get("num_results_per_topic")
        if num_results_per_topic is None:
            num_results_per_topic = 10
        else:
            num_results_per_topic = int(num_results_per_topic)

    	fb_helper = config.FREEBASE_HELPER
    	facts, filtered_facts, name_fact_mapper, num_items = \
            fb_helper.get_facts_by_ids(topic_ids, \
            num_results=num_results, num_results_per_topic=num_results_per_topic)

    	jsoned_facts = [{'src_freebase_name': fact.src.freebase_name,
    	'src_freebase_id': fact.src.freebase_id, 
    	'pred_freebase_name': fact.pred.freebase_name,
    	'pred_freebase_id': fact.pred.freebase_id,
    	'tgt_freebase_name': fact.tgt.freebase_name,
    	'tgt_freebase_id': fact.tgt.freebase_id } for fact in facts]
  
        jsoned_filtered_facts = [{'src_freebase_name': fact.src.freebase_name,
        'src_freebase_id': fact.src.freebase_id, 
        'pred_freebase_name': fact.pred.freebase_name,
        'pred_freebase_id': fact.pred.freebase_id,
        'tgt_freebase_name': fact.tgt.freebase_name,
        'tgt_freebase_id': fact.tgt.freebase_id } for fact in filtered_facts] 

        print(jsoned_filtered_facts)
        print(jsoned_facts)
        return jsonify(result=jsoned_filtered_facts, \
            raw_facts = jsoned_facts, \
            num_items=num_items,
            num_results_per_topic=num_results_per_topic)
