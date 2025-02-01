import requests
import sqlite3
from datetime import datetime, timedelta

# Function to fetch data from the API
def fetch_data(tdate):
    url = "http://www.cninfo.com.cn/data20/shareholeder/detail"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "referer": "http://www.cninfo.com.cn/new/commonUrl?url=data/person-stock-data-tables",
    }
    params = {
        "type": "inc",  # Request type (e.g., increase)
        "tdate": tdate,  # Target date
    }
    response = requests.post(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed, status code: {response.status_code}")
        return None

# Function to create SQLite database and table
def create_db():
    conn = sqlite3.connect('shareholder_data.db')
    c = conn.cursor()
    # Create table with fields matching the JSON records
    c.execute('''
        CREATE TABLE IF NOT EXISTS shareholder_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            DECLAREDATE TEXT,
            SECCODE TEXT,
            SECNAME TEXT,
            VARYDATE TEXT,
            F002V TEXT,
            F004N REAL,
            F005N REAL,
            F007V TEXT
        )
    ''')
    conn.commit()
    return conn

# Function to save records to SQLite database
def save_records(conn, records):
    c = conn.cursor()
    for record in records:
        c.execute('''
            INSERT INTO shareholder_records (
                DECLAREDATE, SECCODE, SECNAME, VARYDATE, F002V, F004N, F005N, F007V
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.get('DECLAREDATE'),
            record.get('SECCODE'),
            record.get('SECNAME'),
            record.get('VARYDATE'),
            record.get('F002V'),
            record.get('F004N'),
            record.get('F005N'),
            record.get('F007V'),
        ))
    conn.commit()

# Main function to fetch data and save to SQLite
def main():
    # Create the database and table
    conn = create_db()

    # Define the date range (e.g., from today to 1 year ago)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Loop through the date range and fetch data for each date
    current_date = start_date
    while current_date <= end_date:
        tdate = current_date.strftime('%Y-%m-%d')
        print(f"Fetching data for date: {tdate}")
        data = fetch_data(tdate)
        if data and 'data' in data and 'records' in data['data']:
            records = data['data']['records']
            if records:
                save_records(conn, records)
                print(f"Saved {len(records)} records for date: {tdate}")
        current_date += timedelta(days=1)

    # Close the database connection
    conn.close()
    print("Data fetching and saving completed.")

if __name__ == "__main__":
    main()