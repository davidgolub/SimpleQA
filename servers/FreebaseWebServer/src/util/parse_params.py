from functools import wraps
from flask.ext.restful import reqparse


def parse_params(*arguments):
    def wrapper(func):
        @wraps(func)
        def decorated_function(*args, **kwargs):
            parser = reqparse.RequestParser()
            for argument in arguments:
                parser.add_argument(**argument)
            params = parser.parse_args()

            return func(*args, params=params, **kwargs)
        return decorated_function
    return wrapper
