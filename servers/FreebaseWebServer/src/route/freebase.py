from flask import Blueprint
from flask.ext.restful import Api


freebase_blueprint = Blueprint('freebase', __name__)
freebase_blueprint_api = Api(freebase_blueprint)


from resource.freebase import FreebaseNameAPI, FreebaseFactAPI

#, 'query', 'num_results'
freebase_blueprint_api.add_resource(FreebaseNameAPI, '/api/v1/freebase/name')

#, 'topic_ids', 'num_results'
freebase_blueprint_api.add_resource(FreebaseFactAPI, '/api/v1/freebase/fact')
