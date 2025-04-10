'''import httpx

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("✅ Testing / endpoint...")
    response = httpx.get(f"{BASE_URL}/")
    print("Status:", response.status_code)
    print("Response:", response.json())

def test_fetch_data():
    print("\n📦 Testing /fetch-data endpoint...")
    response = httpx.get(f"{BASE_URL}/fetch-data")
    print("Status:", response.status_code)
    try:
        json_data = response.json()
        print("Response:", json_data)
    except Exception as e:
        print("❌ Error decoding JSON:", e)
        print("Raw response:", response.text)

def test_hospital_data(hospital_id="Hospital_A"):
    print(f"\n🔍 Testing /data/{hospital_id} endpoint...")
    response = httpx.get(f"{BASE_URL}/data/{hospital_id}")
    print("Status:", response.status_code)
    try:
        json_data = response.json()
        if "error" in json_data:
            print("Error:", json_data)
        else:
            print(f"Records fetched: {len(json_data.get('records', []))}")
            print("First few records:")
            for record in json_data.get("records", [])[:5]:
                print(record)
    except Exception as e:
        print("❌ Error decoding JSON:", e)
        print("Raw response:", response.text)

if __name__ == "__main__":
    test_root()
    test_fetch_data()
    test_hospital_data("Hospital_A")  # You can test other hospital IDs as needed
'''

# aggergated data weekly

'''
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("✅ Testing / endpoint...")
    res = requests.get(f"{BASE_URL}/")
    print("Status:", res.status_code)
    print("Response:", res.json())

def test_fetch_data():
    print("\n📦 Testing /fetch-data endpoint...")
    res = requests.get(f"{BASE_URL}/fetch-data")
    print("Status:", res.status_code)
    print("Response:", res.json())

def test_raw_data(hospital_id="Hospital_A"):
    print(f"\n🔍 Testing /data/{hospital_id} endpoint...")
    res = requests.get(f"{BASE_URL}/data/{hospital_id}")
    print("Status:", res.status_code)
    if res.status_code == 200:
        data = res.json()
        print(f"Records: {len(data)}")
        print("Sample:", data[:3])
    else:
        print("Error:", res.json())

def test_aggregated_data(hospital_id="Hospital_A"):
    print(f"\n📊 Testing /aggregated/{hospital_id} endpoint...")
    res = requests.get(f"{BASE_URL}/aggregated/{hospital_id}")
    print("Status:", res.status_code)
    if res.status_code == 200:
        data = res.json()
        print(f"Aggregated Records: {len(data)}")
        print("Sample:", data[:3])
    else:
        print("Error:", res.json())

if __name__ == "__main__":
    test_root()
    test_fetch_data()
    test_raw_data()
    test_aggregated_data()
'''
'''
import requests
import time

base_url = "http://127.0.0.1:8000"

def test_root():
    print("✅ Testing / endpoint...")
    response = requests.get(f"{base_url}/")
    print("Status:", response.status_code)
    print("Response:", response.json())

def test_fetch_data():
    print("\n📦 Testing /fetch-data endpoint...")
    response = requests.get(f"{base_url}/fetch-data")
    print("Status:", response.status_code)
    try:
        data = response.json()
        print("Response:", data)
        return data
    except Exception as e:
        print("❌ Error decoding JSON:", e)
        print(response.text)
        return None

def test_data(hospital_id):
    print(f"\n📄 Testing /data/{hospital_id} endpoint...")
    response = requests.get(f"{base_url}/data/{hospital_id}")
    print("Status:", response.status_code)
    try:
        data = response.json()
        print("✅ Records:", len(data))
        print("📌 Sample record:", data[0] if data else "No records")
    except Exception as e:
        print("❌ Error:", e)
        print(response.text)

def test_aggregated(hospital_id):
    print(f"\n📊 Testing /aggregated/{hospital_id} endpoint...")
    response = requests.get(f"{base_url}/aggregated/{hospital_id}")
    print("Status:", response.status_code)
    try:
        data = response.json()
        print("✅ Aggregated rows:", len(data))
        print("📌 Sample aggregated row:", data[0] if data else "No data")
    except Exception as e:
        print("❌ Error:", e)
        print(response.text)

if __name__ == "__main__":
    test_root()
    result = test_fetch_data()

    if result and "Hospital_A" in result and result["Hospital_A"]["status"] == "success":
        time.sleep(1.5)  # allow a brief delay before hitting cache
        test_data("Hospital_A")
        test_aggregated("Hospital_A")
    else:
        print("\n⚠️ Skipping /data and /aggregated tests due to fetch failure.")
'''
import requests
import json
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient

# API base URL
BASE_URL = "http://127.0.0.1:8000"
HOSPITAL_ID = "Hospital_A"

def test_root():
    print("✅ Testing / endpoint...")
    r = requests.get(f"{BASE_URL}/")
    print("Status:", r.status_code)
    print("Response:", r.json())

def test_fetch_data():
    print("\n📦 Testing /fetch-data endpoint...")
    r = requests.get(f"{BASE_URL}/fetch-data")
    print("Status:", r.status_code)
    data = r.json()
    print("Response:", json.dumps(data, indent=2))
    return data.get(HOSPITAL_ID, {}).get("status") == "success"

def test_data():
    print("\n📄 Testing /data endpoint...")
    r = requests.get(f"{BASE_URL}/data/{HOSPITAL_ID}")
    print("Status:", r.status_code)
    if r.status_code == 200:
        records = r.json()
        print(f"✅ Records: {len(records)}")
        if records:
            print("📌 Sample record:", records[0])
    else:
        print("❌ Error:", r.text)

def test_aggregated():
    print("\n📊 Testing /aggregated endpoint...")
    r = requests.get(f"{BASE_URL}/aggregated/{HOSPITAL_ID}")
    print("Status:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print(f"✅ Aggregated rows: {len(data)}")
        if data:
            print("📌 Sample aggregated row:", data[0])
    else:
        print("❌ Error:", r.text)

def test_influx_query():
    print("\n📥 Testing InfluxDB data push...")
    client = InfluxDBClient(url="http://localhost:8086", token="1sg0BfTOZ4DCyWn_Y68vHEgUnfqty-TuG4V3iT6_NZOd_w0j8tSJH6YfD9fQvNSAW6yS2fMjeQlUIw2n7MTT6A==", org="aismartlive")
    query_api = client.query_api()

    now = datetime.utcnow()
    one_week_ago = now - timedelta(days=7)
    flux = f'''
    from(bucket: "pro")
    |> range(start: {one_week_ago.isoformat()}Z, stop: {now.isoformat()}Z)
    |> filter(fn: (r) => r["_measurement"] == "hospital_feedback")
    |> filter(fn: (r) => r["hospital_id"] == "{HOSPITAL_ID}")
    |> limit(n:5)
    '''
    try:
        tables = query_api.query(flux)
        count = sum(1 for _ in tables)
        print(f"✅ Found {count} records for {HOSPITAL_ID} in InfluxDB.")
    except Exception as e:
        print("❌ InfluxDB query failed:", e)

if __name__ == "__main__":
    test_root()
    if test_fetch_data():
        test_data()
        test_aggregated()
        test_influx_query()
    else:
        print("⚠️ Skipping /data and /aggregated tests due to fetch failure.")
