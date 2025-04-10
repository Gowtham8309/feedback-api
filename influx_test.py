import requests

BASE_URL = "http://127.0.0.1:8000"
USERNAME = "Hospital_A"
PASSWORD = "secret"

def test_token():
    print("🔐 Testing /token...")
    response = requests.post(f"{BASE_URL}/token", data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200, f"Token request failed: {response.status_code}"
    token = response.json()["access_token"]
    print("✅ Token received:", token)
    return token

def test_data(token):
    print("\n📄 Testing /data...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/data", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        records = response.json()
        print(f"✅ Records: {len(records)}")
        print("📌 Sample record:", records[0] if records else "No records")
    else:
        print("❌ Error:", response.text)

def test_aggregated(token):
    print("\n📊 Testing /aggregated...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/aggregated", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Aggregated rows: {len(data)}")
        print("📌 Sample aggregated row:", data[0] if data else "No rows")
    else:
        print("❌ Error:", response.text)

def test_grafana_embed(token):
    print("\n📺 Testing /grafana-embed...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/grafana-embed", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Embed iframe loaded (HTML response)...")
    else:
        print("❌ Error:", response.text)

if __name__ == "__main__":
    token = test_token()
    test_data(token)
    test_aggregated(token)
    test_grafana_embed(token)
