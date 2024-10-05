import requests
import pandas as pd

class DataQuery:
    def __init__(self, interval, start,end):
        self.interval = interval
        self.start = start
        self.end = end
    
    def fetch(self, sym):
        url = 'https://query2.finance.yahoo.com/v8/finance/chart/' + sym

        headers= {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'}
        payload = {
        'includeAdjustedClose': 'true',
        'interval': self.interval,
        'period1': str(int(self.start.timestamp())),
        'period2': str(int(self.end.timestamp()))
        }

        response = requests.get(url, headers=headers, params=payload).json()

        #Error in request returns NoneType, thus return
        if not response:
            print(response)
            return

        result = response['chart']['result'][0]

        #No price data (possibly querying weekends)
        try:
            ts = result['timestamp']
        except KeyError:
            return

        times = pd.to_datetime(ts,unit='s').strftime("%d-%m-%Y %H:%M:%S")
        #Columns for historical data
        opn = result["indicators"]["quote"][0]['open']
        high = result["indicators"]["quote"][0]['high']
        low = result["indicators"]["quote"][0]['low']
        close = result["indicators"]["quote"][0]['close']
        adjclose = result["indicators"]["adjclose"][0]['adjclose']
        volume = result["indicators"]["quote"][0]['volume']

        data = {
            'time':times,
            'open':opn,
            'high':high,
            'low':low,
            'close':close,
            'adjclose':adjclose,
            'volume':volume
        }
        
        return pd.DataFrame(data)