from flask import Blueprint
from flask.ext.restful import Api


user_blueprint = Blueprint('user', __name__)
user_blueprint_api = Api(user_blueprint)


from resource.user import UserAPI, UserListAPI
user_blueprint_api.add_resource(UserListAPI, '/user')
user_blueprint_api.add_resource(UserAPI, '/user/<int:id>')
