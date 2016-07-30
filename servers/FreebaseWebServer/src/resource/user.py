from flask import jsonify
from flask.ext.restful import Resource

from model.abc import db
from model import User
from client import superhero
from util import parse_params


class UserListAPI(Resource):
    def get(self):
        return jsonify(data=[user.json for user in User.query])

    @parse_params(
        {'name': 'email', 'type': str, 'required': True},
        {'name': 'password', 'type': str, 'required': True}
    )
    def post(self, params):
        user = User(**params)
        db.session.add(user)
        db.session.commit()
        return user.json


class UserAPI(Resource):
    def get(self, id):
        user = User.query.get(id)
        user_dict = user.json
        user_dict['is_superhero'] = superhero.is_superhero(user.email)
        return user_dict
