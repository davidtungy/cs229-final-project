import requests
import sqlparse
from urllib.parse import quote_plus

def request_earnings(symbol):
    sql_query = '''
        SELECT *
        FROM `earnings_calendar`
        WHERE `act_symbol` = 'AAPL'
        ORDER BY `act_symbol` ASC, `date` ASC
        LIMIT 50;
    '''
    formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
    url = f'https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master?q={quote_plus(formatted_sql)}'
    response = requests.get(url)
    print(response.json()['rows'])