import json
import sys
from loguru import logger

CREDENTIALS_FILE = 'credentials.json'

def load_credentials() -> dict:
	try:
		with open(CREDENTIALS_FILE, 'r') as f:
			return json.load(f)
	except FileNotFoundError:
		logger.error(f'File {CREDENTIALS_FILE} not found.')
		sys.exit(1)

def update_credentials(new_credentials: dict):
	credentials = load_credentials()
	credentials.update(new_credentials)
	with open(CREDENTIALS_FILE, 'w') as f:
		json.dump(credentials, f, indent=4)

def get_access_token() -> str:
	credentials = load_credentials()
	return credentials['access_token']