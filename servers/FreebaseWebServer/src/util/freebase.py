class FreebaseObject(object):
	def __init__(self, cur_id, freebase_id, freebase_name, 
		freebase_aliases = ['NOALIAS'], 
		freebase_description = 'NODESCRIPTION'):

		self.freebase_aliases = freebase_aliases
		self.freebase_description = freebase_description
		self.freebase_name = freebase_name
		self.freebase_id = freebase_id
		self.id = cur_id

class FreebaseFact(object):
	""" Creates a new freebase fact with specified src, predicate and target """
	def __init__(self, src, pred, tgt):
		self.src = src 
		self.pred = pred 
		self.tgt = tgt

		