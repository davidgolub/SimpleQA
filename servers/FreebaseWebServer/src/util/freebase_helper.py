from datetime import datetime
from elasticsearch import Elasticsearch
from freebase import FreebaseObject, FreebaseFact
from collections import Counter
from collections import defaultdict

class FreebaseHelper(object):
	FREEBASE_IP = 'softmaxfreebase.cloudapp.net:9200'
	FREEBASE_2M = 'FB_2M'
	FREEBASE_5M = 'FB_5M'

	"""An elasticsearch wrapper that helps index data """
	def __init__(self, ip_addresses, create_index, timeout):
		self.name = 'ElasticSearch'
		self.name_index = 'names_v3'
		self.fact_index = 'facts_v3'
		self.es = Elasticsearch(ip_addresses, timeout=timeout)

		if create_index:
			self.create_indeces()

	def create_indeces(self):
		# Delete indeces
		self.delete_index(self.name_index)
		self.delete_index(self.fact_index)

		# Initialize the indices
		self.create_index(self.name_index)

		# Initialize the facts
		self.create_index(self.fact_index)

	def set_index(self, index_name):
		""" Sets the index to be index_name.
			Either FREEBASE_2M or Freebase_5M
		"""
		if index_name == FreebaseHelper.FREEBASE_2M:
			self.name_index = 'names_v3'
			self.fact_index = 'facts_v3'
		elif index_name == FreebaseHelper.FREEBASE_5M:
			self.name_index = 'fb_5m_names_v1'
			self.fact_index = 'fb_5m_facts_v1'
		else:
			raise Exception("Unknown index given %s" % index_name)

	def delete_index(self, index_name):
		""" Deletes index with specified name 
			index_name: Index to delete 
		"""
		if self.es.indices.exists(index_name):
			print("Deleting index with name %s" % index_name)
			self.es.indices.delete(index=index_name)
		else:
			print("Index with name %s does not exist so no need to delete" % index_name)

	def create_index(self, index_name): 
		""" Creates index with specified name 
			index_name: Index to create
		"""
		print("Creating index with name %s" % index_name)
		index_settings = {
			  "mappings": {
			    "fact": {
			      "properties": {
			        "src_freebase_name": {
			          "type":  "string",
			          "index": "not_analyzed" 
			        },
			        "src_freebase_name_analyzed": {
			          "type":  "string",
			        },
			        "src_freebase_id": {
			          "type":  "string",
			          "index": "not_analyzed" 
			        },
			        "tgt_freebase_id": {
			          "type":  "string",
			          "index": "not_analyzed" 
			        },
			        "predicate": {
			          "type":  "string",
			          "index": "not_analyzed"
			        },
			        "tgt_freebase_name": {
			          "type":  "string",
			        }
			      }
			    }
			  },
		    "settings": {
		    "analysis": {
		      "analyzer": {
		        "evolutionAnalyzer": {
		          "tokenizer": "standard",
		          "filter": [
		            "standard",
		            "lowercase",
		            "custom_shingle"
		          ]
		        }
		      },
		      "filter": {
		        "custom_shingle": {
		            "type": "shingle",
		            "min_shingle_size": "1",
		            "max_shingle_size": "5",
		            "output_unigrams" : "True"
		        }
		      }
		    }
		  }
		}

		self.es.indices.create(
		    index=index_name,
		    body=index_settings,
		)

	def index_names(self, freebase_objs):
		bulk_operands = []
		for i in range(0, len(freebase_objs)):
			if i % 50000 == 1:
				print("Index stuff on index %s" % i)
				self.es.bulk(index = self.name_index, body = bulk_operands, refresh = True)
				bulk_operands = []

			cur_obj = freebase_objs[i]
			cur_id = cur_obj.id 
			cur_freebase_id = cur_obj.freebase_id 
			cur_name = cur_obj.freebase_name 
			cur_aliases = cur_obj.freebase_aliases
			cur_description = cur_obj.freebase_description

			cur_doc = {
			  'freebase_id': cur_freebase_id,
			  'name' : cur_name,
			  'description' : cur_description,
			  'aliases' : ' '.join(cur_aliases)
			}

			#for i in range(0, len(cur_aliases)):
			#	key = 'alias_%s' % i 
			#	value = cur_aliases[i]
			#	cur_doc[key] = value
			
			op_dict = {
			  "index": {
			     "_index": self.name_index, 
			     "_id": cur_id,
			     "_type": 'name'
			   }
			}
			bulk_operands.append(op_dict)
			bulk_operands.append(cur_doc)

		self.es.bulk(index = self.name_index, body = bulk_operands, refresh = True)

	def index_name(self, freebase_obj):
		""" Indexes freebase name
			freebase_obj: Object to index
		"""
		self.index_names([freebase_obj])
		
	def index_fact(self, freebase_fact):
		self.index_facts([freebase_fact])

	def index_facts(self, freebase_facts):
		""" Indexes freebase fact of the form src, predicate, tgt 
			freebase_facts: (has a src, predicate tgt)
		"""

		bulk_operands = []
		for i in range(0, len(freebase_facts)):
			if i % 5000 == 1:
				print("Index stuff on index %s" % i)
				self.es.bulk(index = self.fact_index, body = bulk_operands, refresh = True)
				bulk_operands = []

			cur_fact = freebase_facts[i]
			src_obj = cur_fact.src
			predicate = cur_fact.pred
			tgt_obj = cur_fact.tgt

			src_id = src_obj.id 
			src_freebase_id = src_obj.freebase_id 
			src_freebase_name = src_obj.freebase_name 

			predicate_name = predicate.freebase_name 

			tgt_id = tgt_obj.id 
			tgt_freebase_id = tgt_obj.freebase_id 
			tgt_freebase_name = tgt_obj.freebase_name 

			cur_doc = {
			  'src_freebase_id': src_freebase_id,
			  'src_freebase_name': src_freebase_name,
			  'src_freebase_name_analyzed': src_freebase_name,
			  'predicate' : predicate_name,
			  'tgt_freebase_id' : tgt_freebase_id,
			  'tgt_freebase_name' : tgt_freebase_name
			}

			op_dict = {
			  "index": {
			     "_index": self.fact_index, 
			     "_id": src_id,
			     "_type": 'fact'
			   }
			}
			bulk_operands.append(op_dict)
			bulk_operands.append(cur_doc)

		self.es.bulk(index = self.fact_index, body = bulk_operands, refresh = True)

	def get_names(self, query, num_results):
		""" Returns all freebase objects that match name
			query: Query to run against freebase names index
			num_results: Number of results to return
		"""
		print(query)
		freebase_objs = []
		elastic_query = {
		  "query": {
		  #"match": {
		  # 	"name" : query
		  #}
		   "multi_match": {
		    	"query": query,
		    	"fields": ["name", "description^2"], #"aliases"] 
		   }
		  },
		  "size" : num_results
		}
		res = self.es.search(index=self.name_index, body=elastic_query)
		num_total = res['hits']['total']
		for hit in res['hits']['hits']:
			src = hit["_source"]
			cur_obj = FreebaseObject(-1, src["freebase_id"], src["name"], [], src["description"])
			freebase_objs.append(cur_obj)
		return freebase_objs, num_total

	def get_names_by_ids(self, topic_ids):
		""" Returns all freebase facts for topic_id
			num_results: Number of results to return 
		"""

		query_terms = map(lambda topic_id:  {"term" : { "freebase_id" : topic_id }}, topic_ids)
		elastic_query = {
		    "query": {
		    	"bool": {
				      "should" : query_terms,
				       "minimum_should_match" : 1,
				    }
				  }
		}

		freebase_objs = []
		res = self.es.search(index=self.name_index, body=elastic_query)
		num_total = res['hits']['total']
		for hit in res['hits']['hits']:
			src = hit["_source"]
			cur_obj = FreebaseObject(-1, src["freebase_id"], src["name"], [], src["description"])
			freebase_objs.append(cur_obj)
		return freebase_objs, num_total

	def get_facts_by_id(self, topic_id, num_results, num_results_per_topic):
		""" Returns all freebase facts for topic_id
			num_results: Number of results to return 
		"""
		freebase_facts, filtered_facts, name_facts_counter, num_total = self.get_facts_by_ids([topic_id], num_results, num_results_per_topic)
		
		return freebase_facts, filtered_facts, name_facts_counter, num_total

	def get_facts_by_ids(self, topic_ids, num_results, num_results_per_topic):
		""" Returns all freebase facts for topic_id
			num_results: Number of results to return 
		"""

		query_terms = map(lambda topic_id:  {"term" : { "src_freebase_id" : topic_id }}, topic_ids)
		elastic_query = {
		    "query": {
		    	"bool": {
				      "should" : query_terms,
				       "minimum_should_match" : 1,
				    }
				  },
		    "size": num_results
		}

		seen_facts = {}

		res = self.es.search(index=self.fact_index, body=elastic_query)
		
		# Total freebase facts
		freebase_facts = []
		facts_seen = {}
		num_total = res['hits']['total']

		# Keep track of number of facts per name per id
		name_facts_counter = defaultdict(Counter)

		# Facts by id
		for hit in res['hits']['hits']:
		   	for hit in res['hits']['hits']:
				src = hit["_source"]
				src_id = src["src_freebase_id"]
				key = src_id + src["predicate"]
				name = src["src_freebase_name"]
				if key not in facts_seen:
					name_facts_counter[name].update([src_id])
					facts_seen[key] = facts_seen
					src_obj = FreebaseObject(-1, src["src_freebase_id"], src["src_freebase_name"])
					tgt_obj = FreebaseObject(-1, src["tgt_freebase_id"], src["tgt_freebase_name"])
					pred_obj = FreebaseObject(-1, src["predicate"], src["predicate"])

					freebase_fact = FreebaseFact(src_obj, pred_obj, tgt_obj)
					freebase_facts.append(freebase_fact)
		
		all_ids = []
		for k, id_counts in name_facts_counter.iteritems():
			curr_ids = map(lambda x:x[0], id_counts.most_common(num_results_per_topic))
			print(curr_ids)
			all_ids.extend(curr_ids)

		corr_ids = set(all_ids)

		filtered_facts = filter(lambda fact: fact.src.freebase_id in corr_ids, freebase_facts)
		return freebase_facts, filtered_facts, name_facts_counter, num_total


	def get_facts_by_name(self, topic_name, num_results):
		""" Returns all freebase facts for topic_id
			num_results: Number of results to return 
		"""
		elastic_query = {
		  "query": {
		  	"match": {
		  	 	"src_freebase_name_analyzed" : topic_name
		  	}
		  },
		  "size" : num_results
		}

		freebase_facts = []
		res = self.es.search(index=self.fact_index, body=elastic_query)
		
		num_total = res['hits']['total']
		for hit in res['hits']['hits']:
		   	for hit in res['hits']['hits']:
				src = hit["_source"]
				src_obj = FreebaseObject(-1, src["src_freebase_id"], src["src_freebase_name"])
				tgt_obj = FreebaseObject(-1, src["tgt_freebase_id"], src["tgt_freebase_name"])
				pred_obj = FreebaseObject(-1, src["predicate"], src["predicate"])

				freebase_fact = FreebaseFact(src_obj, pred_obj, tgt_obj)
				freebase_facts.append(freebase_fact)
		
		return freebase_facts, num_total