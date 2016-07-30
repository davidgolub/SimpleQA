import json
import unittest
from mock import patch

from server import server
from model.abc import db
from model import User


class TestUser(unittest.TestCase):

    def setUp(self):
        server.config['TESTING'] = True
        self.client = server.test_client()

        db.create_all()

        self.user = User('joe@example.fr', 'super-secret-password')
        db.session.add(self.user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    @patch('client.superhero.is_superhero')
    def test_get_user(self, is_superhero_mock):
        is_superhero_mock.return_value = True

        response = self.client.get(
            '/user/%d' % self.user.id,
        )
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data.decode('utf-8'))
        self.assertEqual(result['email'], 'joe@example.fr')
        self.assertEqual(result['is_superhero'], True)

if __name__ == '__main__':
    unittest.main()
