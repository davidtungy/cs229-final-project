import os
import base64
import requests
import webbrowser
import time
from loguru import logger

import credentials


def construct_init_auth_url(app_key: str) -> str:
    auth_url = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri=https://127.0.0.1'
    
    logger.info("Click to authenticate:")
    logger.info(auth_url)

    return auth_url

def construct_headers_and_payload(returned_url, app_key, app_secret):
    response_code = f'{returned_url[returned_url.index("code=") + 5: returned_url.index("%40")]}@'

    credentials = f'{app_key}:{app_secret}'
    base64_credentials = base64.b64encode(credentials.encode('utf-8')).decode(
        'utf-8'
    )

    headers = {
        'Authorization': f'Basic {base64_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    payload = {
        'grant_type': 'authorization_code',
        'code': response_code,
        'redirect_uri': 'https://127.0.0.1',
    }

    return headers, payload

def retrieve_tokens(headers, payload) -> dict:
    init_token_response = requests.post(
        url='https://api.schwabapi.com/v1/oauth/token',
        headers=headers,
        data=payload,
    )

    init_tokens_dict = init_token_response.json()

    return init_tokens_dict

def refresh_tokens(app_key, app_secret, refresh_token):
    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
    }
    headers = {
        'Authorization': f"Basic {base64.b64encode(f'{app_key}:{app_secret}'.encode()).decode()}",
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    refresh_token_response = requests.post(
        url="https://api.schwabapi.com/v1/oauth/token",
        headers=headers,
        data=payload,
    )
    if refresh_token_response.status_code == 200:
        logger.info("Retrieved new tokens successfully using refresh token.")
    else:
        logger.error(
            f"Error refreshing access token: {refresh_token_response.text}"
        )
        return None

    refresh_token_dict = refresh_token_response.json()

    logger.debug(refresh_token_dict)

    logger.info("Token dict refreshed.")

    credentials.update_credentials(refresh_token_dict)


def main():
    cred = credentials.load_credentials()
    app_key, app_secret = cred['app_key'], cred['app_secret']
    cs_auth_url = construct_init_auth_url(app_key)
    webbrowser.open(cs_auth_url)

    logger.info('Paste returned URL:')
    returned_url = input()

    init_token_headers, init_token_payload = construct_headers_and_payload(
        returned_url, app_key, app_secret
    )

    init_tokens_dict = retrieve_tokens(
        headers=init_token_headers, payload=init_token_payload
    )

    logger.debug(init_tokens_dict)

    credentials.update_credentials(init_tokens_dict)

    try:
        while True:
            time.sleep(29 * 60)
            refresh_token = credentials.load_credentials()['refresh_token']
            refresh_tokens(app_key, app_secret, refresh_token)
    except KeyboardInterrupt:
        logger.info('Exiting...')


if __name__ == "__main__":
    main()