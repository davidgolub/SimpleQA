from werkzeug.security import generate_password_hash, check_password_hash

from .abc import db, BaseModel


class User(db.Model, BaseModel):
    __tablename__ = 'auth_user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(120))

    def __init__(self, email=None, password=None):
        if email:
            self.email = email.lower()
        if password:
            self.set_password(password)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)
