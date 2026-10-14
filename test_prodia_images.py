"""Regression checks for the Prodia bridge; no paid upstream requests."""

import unittest
from unittest.mock import Mock, patch

import requests
from flask import Flask

from modules.prodia_images import PRODIA_JOB_TYPE, PRODIA_JOB_URL, prodia_images


class ProdiaImagesTest(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(prodia_images)
        self.client = app.test_client()
        self.payload = {
            'api_key': '  Bearer test-token  ',
            'accept': 'image/jpeg;quality=84;progressive=1',
            'job': {
                'type': PRODIA_JOB_TYPE,
                'config': {'prompt': 'A green meadow', 'width': 1365, 'height': 768,
                           'steps': 4, 'seed': 123},
            },
        }

    @patch('modules.prodia_images.requests.post')
    def test_header_and_binary_image_without_browser_authorization(self, post):
        post.return_value = Mock(status_code=200, content=b'jpeg-bytes',
                                 headers={'Content-Type': 'image/jpeg'})
        response = self.client.post('/api/vibeos/images', json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'jpeg-bytes')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        post.assert_called_once_with(
            PRODIA_JOB_URL,
            headers={'Authorization': 'Bearer test-token', 'Accept': self.payload['accept']},
            json=self.payload['job'], timeout=(10, 120), allow_redirects=False,
        )

    @patch('modules.prodia_images.requests.post')
    def test_missing_or_malformed_key_never_calls_upstream(self, post):
        for key, status in [('', 401), (None, 401), ('Bearer', 401), ('bad\nkey', 400)]:
            with self.subTest(key=key):
                self.payload['api_key'] = key
                response = self.client.post('/api/vibeos/images', json=self.payload)
                self.assertEqual(response.status_code, status)
                self.assertIn('error', response.json)
        post.assert_not_called()

    @patch('modules.prodia_images.requests.post')
    def test_invalid_job_does_not_call_upstream(self, post):
        for config in [{'prompt': 'x'}, {'prompt': 'valid', 'width': 1921},
                       {'prompt': 'valid', 'steps': 5}, {'prompt': 'valid', 'seed': True},
                       {'prompt': 'valid', 'url': 'https://example.com'}]:
            with self.subTest(config=config):
                self.payload['job']['config'] = config
                self.assertEqual(self.client.post('/api/vibeos/images', json=self.payload).status_code, 400)
        self.payload['job']['type'] = 'workflow.serial.v1'
        self.assertEqual(self.client.post('/api/vibeos/images', json=self.payload).status_code, 400)
        post.assert_not_called()

    @patch('modules.prodia_images.requests.post')
    def test_upstream_error_status_and_retry_header_preserved(self, post):
        post.return_value = Mock(status_code=429,
                                 headers={'Content-Type': 'application/json', 'Retry-After': '3'})
        post.return_value.json.return_value = {'error': 'No capacity for test-token'}
        response = self.client.post('/api/vibeos/images', json=self.payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['Retry-After'], '3')
        self.assertEqual(response.json['error'], 'No capacity for [redacted]')

    @patch('modules.prodia_images.requests.post')
    def test_authentication_error_is_visible(self, post):
        post.return_value = Mock(status_code=400, headers={})
        post.return_value.json.return_value = {'error': 'authorization header is required'}
        response = self.client.post('/api/vibeos/images', json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], 'authorization header is required')

    @patch('modules.prodia_images.requests.post')
    def test_redirect_and_non_image_success_rejected(self, post):
        for status in (302, 200):
            with self.subTest(status=status):
                post.return_value = Mock(status_code=status, headers={'Content-Type': 'text/html'})
                response = self.client.post('/api/vibeos/images', json=self.payload)
                self.assertEqual(response.status_code, 502)

    @patch('modules.prodia_images.requests.post')
    def test_plain_text_error_is_preserved(self, post):
        post.return_value = Mock(status_code=400, headers={'Content-Type': 'text/plain'},
                                 text='authorization header is required')
        post.return_value.json.side_effect = ValueError()
        response = self.client.post('/api/vibeos/images', json=self.payload)
        self.assertEqual(response.json['error'], 'authorization header is required')

    @patch('modules.prodia_images.requests.post')
    def test_network_failures(self, post):
        for error, status in [(requests.Timeout(), 504), (requests.ConnectionError(), 502)]:
            with self.subTest(error=error):
                post.side_effect = error
                response = self.client.post('/api/vibeos/images', json=self.payload)
                self.assertEqual(response.status_code, status)


if __name__ == '__main__':
    unittest.main()
