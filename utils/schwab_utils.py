import requests

from authentication import credentials

BASE_URL = 'https://api.schwabapi.com'

def get_daily_price_history(symbol) -> dict[str, any]:
    url = f'{BASE_URL}/marketdata/v1/pricehistory'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {credentials.get_access_token()}'
    }
    params = {
        'symbol': symbol,
        'periodType': 'year',
        'period': '5', # 1, 2, 3, 5, 10, 15, 20
        'frequencyType': 'daily'
    }

    response = requests.get(url, headers=headers, params=params)

    # Ensure the request was successful
    response.raise_for_status()

    return response.json()['candles']



def test_function():
	print(credentials.get_access_token())