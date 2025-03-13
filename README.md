# one-sigma

Install dependencies:
1. Run `pip install -r requirements.txt`

Set up access to Charles Schwab API:
1. Run `python authentication/auth.py`
2. Login to Charles Schwab account and grant access
3. Copy url and paste into terminal

Choose either linear or transformer mode type:
1. Run `python regression.py <model_type> <stock_symbol>`