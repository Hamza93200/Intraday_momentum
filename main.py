Sub ComputeSpotReturns()

    Dim wsSrc As Worksheet
    Dim wsDst As Worksheet
    Dim lastRow As Long, lastCol As Long
    Dim colIndex As Long, i As Long
    Dim headers() As String
    Dim dates() As Variant
    Dim data() As Variant
    Dim currencies() As String
    Dim currencyCount As Long
    
    Set wsSrc = ThisWorkbook.Sheets("Spot")
    
    ' Prepare Returns sheet
    On Error Resume Next
    Set wsDst = ThisWorkbook.Sheets("Returns")
    If wsDst Is Nothing Then
        Set wsDst = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        wsDst.Name = "Returns"
    Else
        wsDst.Cells.Clear
    End If
    On Error GoTo 0
    
    ' Identify data range
    lastRow = wsSrc.Cells(wsSrc.Rows.Count, 1).End(xlUp).Row
    lastCol = wsSrc.Cells(1, wsSrc.Columns.Count).End(xlToLeft).Column
    
    ' Read headers
    ReDim headers(1 To lastCol)
    For i = 1 To lastCol
        headers(i) = wsSrc.Cells(1, i).Value
    Next i
    
    ' Get date range
    dates = wsSrc.Range(wsSrc.Cells(2, 1), wsSrc.Cells(lastRow, 1)).Value

    ' Identify currency columns (skip column 1 = DATES)
    currencyCount = 0
    ReDim currencies(1 To lastCol - 1)
    For colIndex = 2 To lastCol
        If InStr(headers(colIndex), "Curncy") > 0 Then
            currencyCount = currencyCount + 1
            currencies(currencyCount) = headers(colIndex)
        End If
    Next colIndex
    ReDim Preserve currencies(1 To currencyCount)

    ' Define maturity buckets
    Dim maturities As Variant
    maturities = Array("Daily", "Weekly", "Monthly", "Quarterly")

    ' Initialize write row
    Dim writeRow As Long: writeRow = 1
    wsDst.Cells(writeRow, 1).Value = "PeriodStart"

    ' Write headers dynamically
    Dim m As Variant, c As Variant
    Dim colOffset As Long: colOffset = 2

    For Each m In maturities
        For Each c In currencies
            Dim cleanName As String
            cleanName = Replace(c, " Curncy", "")
            wsDst.Cells(writeRow, colOffset).Value = m & "_abs_diff_" & cleanName
            wsDst.Cells(writeRow, colOffset + 1).Value = m & "_rel_diff_" & cleanName
            colOffset = colOffset + 2
        Next c
    Next m

    ' Now compute grouped returns
    Dim dateDict As Object
    Set dateDict = CreateObject("Scripting.Dictionary")

    ' Read all data into memory
    ReDim data(1 To lastRow - 1, 1 To lastCol)
    For i = 2 To lastRow
        For colIndex = 1 To lastCol
            data(i - 1, colIndex) = wsSrc.Cells(i, colIndex).Value
        Next colIndex
    Next i

    ' Loop over maturities
    writeRow = 2

    For Each m In maturities
        dateDict.RemoveAll
        Dim formatKey As String

        For i = 1 To UBound(data)
            Dim dt As Date: dt = data(i, 1)

            Select Case m
                Case "Daily": formatKey = Format(dt, "yyyy-mm-dd")
                Case "Weekly": formatKey = Format(dt, "yyyy") & "-" & Format(dt, "ww", vbMonday)
                Case "Monthly": formatKey = Format(dt, "yyyy-mm")
                Case "Quarterly": formatKey = Format(dt, "yyyy") & "-Q" & Int((Month(dt) - 1) / 3) + 1
            End Select

            If Not dateDict.exists(formatKey) Then
                dateDict.Add formatKey, Array(i, i) ' [start index, end index]
            Else
                dateDict(formatKey)(1) = i ' update end index
            End If
        Next i

        ' Write values for each group
        Dim groupKey As Variant
        Dim wCol As Long: wCol = 2 + (Application.Match(m & "_abs_diff_" & Replace(currencies(1), " Curncy", ""), wsDst.Rows(1), 0) - 1)

        For Each groupKey In dateDict.Keys
            Dim idxStart As Long, idxEnd As Long
            idxStart = dateDict(groupKey)(0)
            idxEnd = dateDict(groupKey)(1)

            ' Write period start date
            wsDst.Cells(writeRow, 1).Value = data(idxStart, 1)

            For c = 1 To currencyCount
                Dim colNum As Long
                colNum = Application.Match(currencies(c), headers, 0)

                Dim vStart As Variant, vEnd As Variant
                vStart = data(idxStart, colNum)
                vEnd = data(idxEnd, colNum)

                If IsNumeric(vStart) And IsNumeric(vEnd) And vStart <> 0 Then
                    Dim absDiff As Double, relDiff As Double
                    absDiff = Abs(vEnd - vStart)
                    relDiff = absDiff / vStart

                    wsDst.Cells(writeRow, wCol).Value = absDiff
                    wsDst.Cells(writeRow, wCol + 1).Value = relDiff
                Else
                    wsDst.Cells(writeRow, wCol).Value = ""
                    wsDst.Cells(writeRow, wCol + 1).Value = ""
                End If

                wCol = wCol + 2
            Next c

            writeRow = writeRow + 1
            wCol = 2 + (Application.Match(m & "_abs_diff_" & Replace(currencies(1), " Curncy", ""), wsDst.Rows(1), 0) - 1)
        Next groupKey
    Next m

    MsgBox "Returns computation complete!", vbInformation

End Sub







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
