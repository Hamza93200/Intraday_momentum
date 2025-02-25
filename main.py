
from datetime import datetime, timedelta
import pandas as pd
import hashlib
import hmac
import base64
import numpy as np
import json
import requests
import time


############################################ Bitget API ##############################################################################################################
API_KEY = "bg_9fe5954aea4db68bcb1b72dfd8ad03c3"
SECRET_KEY = "75dc3ada3f4eaab4704a9fa8974aec381321e1c8bc973cf53ee5e63bc8adecc1"
PASSPHRASE = "Pakistan93200"
BASE_URL = "https://api.bitget.com"




############################################ Functions ##############################################################################################################

def generate_signature(timestamp, method, endpoint, body=""):
    message = f"{timestamp}{method}{endpoint}{body}"
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode("utf-8")

def close_positions(product_type="USDT-FUTURES"):

    if not API_KEY or not SECRET_KEY or not PASSPHRASE:
        print("Error: API credentials are missing.")
        return

    endpoint = "/api/v2/mix/order/close-positions"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000)) 

    body = json.dumps({
        "productType": product_type
    }) 

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": generate_signature(timestamp, "POST", endpoint, body),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=body)

    try:
        data = response.json()  # Convert response to JSON
        print(json.dumps(data, indent=4))  # Pretty-print the response
    except ValueError:
        print("Error: Unable to decode JSON response.")
        return None

    return data


#def generate_signature(timestamp, method, request_path, body=""):
    #message = f"{timestamp}{method}{request_path}{body}"
    #signature = hmac.new(
        #SECRET_KEY.encode(), message.encode(), hashlib.sha256
    #).digest()
    #return base64.b64encode(signature).decode()


def get_crypto_balance():
    if not API_KEY or not SECRET_KEY or not PASSPHRASE:
        print("Error: API credentials are missing. Set them as environment variables.")
        return pd.DataFrame(columns=["Asset", "Balance"])

    endpoint = "/api/v2/spot/account/assets"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000)) 

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": generate_signature(timestamp, "GET", endpoint),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    try:
        data = response.json()  # Convert response to JSON
    except ValueError:
        print("Error: Unable to decode JSON response.")
        return pd.DataFrame(columns=["Asset", "Balance"])

    #print("API Response:", data)  

    if response.status_code == 200:
        if data.get("code") == "00000" and "data" in data:
            balances = data["data"]
            
            if not balances:
                print("Warning: No balance data found.")
                return pd.DataFrame(columns=["Asset", "Balance"])

            df = pd.DataFrame(balances)
            #print("Available columns:", df.columns.tolist())

            asset_column = "coin" if "coin" in df.columns else "asset"
            balance_column = "available" if "available" in df.columns else "balance"

            if asset_column not in df.columns or balance_column not in df.columns:
                print("Error: Unexpected API response structure.")
                return pd.DataFrame(columns=["Asset", "Balance"])
            
            df = df[[asset_column, balance_column]]
            df.rename(columns={asset_column: "Asset", balance_column: "Balance"}, inplace=True)
            df["Balance"] = df["Balance"].astype(float) 
            return df
        else:
            print(f"Error: {data.get('msg', 'Unknown error')}")
    else:
        print(f"HTTP Error {response.status_code}: {response.text}")

    return pd.DataFrame(columns=["Asset", "Balance"])



def set_leverage(symbol, leverage, holdSide, product_type="USDT-FUTURES", margin_mode="isolated"):
    endpoint = "/api/v2/mix/account/set-leverage"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000))

    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "leverage": str(leverage),
        "marginCoin":"USDT",
        "holdSide" : holdSide
    }

    body_json = json.dumps(body)

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": generate_signature(timestamp, "POST", endpoint, body_json),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=body_json)

    try:
        data = response.json()
        print("Leverage Set Response:", json.dumps(data, indent=4))
        return data
    except ValueError:
        print("Error: Unable to decode JSON response.")
        return None

def get_usdt_balance():

    if not API_KEY or not SECRET_KEY or not PASSPHRASE:
        print("Error: API credentials are missing. Set them as environment variables.")
        return pd.DataFrame(columns=["Asset", "Balance"])

    endpoint ="/api/v2/mix/account/accounts?productType=USDT-FUTURES"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000)) 

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": generate_signature(timestamp, "GET", endpoint),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    try:
        
        data = response.json()['data'][0]['available']  # Convert response to JSON
        print(f'available usdt in USDT FUTURES account {data}')
        return data
    except ValueError:
        print("Error: Unable to decode JSON response and get USDT balance on USDT Futures.")
        return
    

    
def place_order(
    symbol,
    size=None,
    side=None,
    trade_side=None,
    leverage =None,
    product_type="USDT-FUTURES",
    margin_mode="isolated",
    order_type="market",
    margin_coin="USDT"
):

    if not API_KEY or not SECRET_KEY or not PASSPHRASE:
        print("Error: API credentials are missing.")
        return
    
    if not symbol or not side or not size:
        print("Error: 'symbol', 'side', and 'size' are required parameters.")
        return
    
    if side == "buy":
        holdSide = "long"
    elif side =="sell":
        holdSide ="short"

    

    endpoint = "/api/v2/mix/order/place-order"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000))

    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": str(size),  # Convert size to string to match API requirements
        "side": side,  # "buy" or "sell"
        "orderType": order_type  # "limit" or "market"
    }


    if trade_side is not None:
        body["tradeSide"] = trade_side  # Only required in hedge mode

    body_json = json.dumps(body)  # Convert body to JSON format

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": generate_signature(timestamp, "POST", endpoint, body_json),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,  # Added as per API example
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=body_json)

    try:
        data = response.json()  # Convert response to JSON
        set_leverage(symbol, leverage,holdSide)
        print(json.dumps(data, indent=4))  # Pretty-print the response
    except ValueError:
        print("Error: Unable to decode JSON response.")
        return None

    return data



def get_historical_1m_data1(symbol: str, limit: int = 288):

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": "5m",
        "limit": limit  # 288 * 5m = 1440m (24h)
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Erreur API Binance: {response.status_code} - {response.text}")
        return []
    
    data = response.json()
    
    # Extract relevant data (timestamp, close price)
    df = pd.DataFrame(
        [(int(candle[0]), float(candle[4])) for candle in data],
        columns=["timestamp", "close"]
    )
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    return df


def get_historical_1m_data(symbol: str, limit: int = 288):
    """
    Fetches historical 5-minute candlestick data from CoinGecko with rate limiting.

    Parameters:
        symbol (str): Cryptocurrency ID (e.g., "bitcoin").
        limit (int): Number of data points (default = 288, representing 24 hours).

    Returns:
        pd.DataFrame: DataFrame with timestamp and close price.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1",
        "interval": "5m"
    }

    retries = 3  # Retry up to 3 times if rate-limited
    for attempt in range(retries):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            break  # Success, exit loop
        elif response.status_code == 429:
            print(f"⚠️ Rate limit exceeded (Attempt {attempt+1}/{retries}). Waiting before retry...")
            time.sleep(15)  # Wait before retrying
        else:
            print(f"❌ Error API CoinGecko: {response.status_code} - {response.text}")
            return pd.DataFrame()  # Return empty DataFrame on failure

    data = response.json().get("prices", [])

    if not data:
        print("⚠️ No data received from CoinGecko.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df.tail(limit)



def get_prices_df(token_pool):
    df_combined = None
    
    for token in token_pool:
        df = get_historical_1m_data(token+"USDT")
        
        if df is not None:
            df.rename(columns={"close": token+'USDT'}, inplace=True)  # Rename 'close' to token name
            
            if df_combined is None:
                df_combined = df
            else:
                df_combined = df_combined.join(df, how="outer")  # Align timestamps
    
    return df_combined

def compute_and_sort_returns(df):

    returns = np.log(df / df.shift(1))
    avg_returns = returns.mean()*100
    sorted_avg_returns = avg_returns.sort_values(ascending=False).to_frame(name="Average Return")
    
    return sorted_avg_returns

def get_tops(df):
    sorted_returns = compute_and_sort_returns(df)
    best_tokens = sorted_returns[sorted_returns["Average Return"] > 0].head(5).index.tolist()

    worst_tokens = sorted_returns.tail(5).index.tolist()

    best_tokens += [None] * (5 - len(best_tokens))

    # Create a DataFrame for better display
    results_df = pd.DataFrame({
        "Best Tokens": best_tokens,
        "Worst Tokens": worst_tokens
    })

    return results_df

token_pool = ['BTC', 'ETH', 'SOL', 'XRP', 'LTC', 'DOGE',
              'SUI', 'TRUMP', 'ADA', 'BNB', 'TRX', 'PEPE',
              'LINK', 'AVAX', 'WIF', 'BERA', 'AAVE',
              'ATOM', 'NEAR', 'INJ','APT','ICP',
             'DOT', 'SHIB','XLM','HBAR','TON','OM','UNI','TAO']



df_prices = get_prices_df(token_pool)
print(df_prices)
