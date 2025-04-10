
'''
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio

app = FastAPI()

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/1BMlV9imgOrIxVHMRxParAzbvTCW-UesF/export?format=xlsx",
    #"Hospital_B": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago]
    return filtered_df

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df
            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df)
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())  # Print the DataFrame content
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data/{hospital_id}")
async def get_hospital_data(hospital_id: str):
    if hospital_id not in hospital_data_cache:
        return JSONResponse(status_code=404, content={"error": "No data found for this hospital"})
    df = hospital_data_cache[hospital_id]
    df = df.copy()
    df['visit_date'] = df['visit_date'].astype(str)  # Convert Timestamps to string
    return JSONResponse(content=df.to_dict(orient="records"))

# Run with: uvicorn main:app --reload
'''


'''
# data aggeration based upon visit_data+department
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os

app = FastAPI()

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    #"Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx"
    "Hospital_A": "https://docs.google.com/spreadsheets/d/1BMlV9imgOrIxVHMRxParAzbvTCW-UesF/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}
hospital_aggregated_cache = {}

# Directory to store JSON files
OUTPUT_DIR = "hospital_feedback_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago]
    return filtered_df

# Aggregate data by visit_date and department
def aggregate_feedback(df: pd.DataFrame) -> pd.DataFrame:
    feedback_cols = df.select_dtypes(include='number').columns.tolist()
    grouped = df.groupby(["visit_date", "department"])[feedback_cols].mean().reset_index()
    return grouped

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Save raw filtered data
            filtered_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw.json")
            filtered_df.to_json(filtered_path, orient="records", date_format="iso")

            # Aggregate and save
            agg_df = aggregate_feedback(filtered_df)
            aggregated_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_aggregated.json")
            agg_df.to_json(aggregated_path, orient="records", date_format="iso")

            # Cache aggregated
            hospital_aggregated_cache[hospital_id] = agg_df

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [filtered_path, aggregated_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(agg_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data/{hospital_id}")
async def get_hospital_data(hospital_id: str):
    if hospital_id not in hospital_data_cache:
        return JSONResponse(status_code=404, content={"error": "No data found for this hospital"})
    df = hospital_data_cache[hospital_id].copy()
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated/{hospital_id}")
async def get_aggregated_data(hospital_id: str):
    if hospital_id not in hospital_aggregated_cache:
        return JSONResponse(status_code=404, content={"error": "No aggregated data found for this hospital"})
    df = hospital_aggregated_cache[hospital_id].copy()
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))
'''


'''
#overall aggerated weekly data(visit_date=startdate) round off allvalues to one decimal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os

app = FastAPI()

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/1BMlV9imgOrIxVHMRxParAzbvTCW-UesF/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}
hospital_aggregated_cache = {}

# Directory to store JSON files
OUTPUT_DIR = "hospital_feedback_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago]
    return filtered_df

# Aggregate data so that department appears only once per week
def aggregate_feedback(df: pd.DataFrame) -> pd.DataFrame:
    feedback_cols = df.select_dtypes(include='number').columns.tolist()
    df['week'] = df['visit_date'].dt.to_period('W').apply(lambda r: r.start_time)
    grouped = df.groupby(['week', 'department'])[feedback_cols].mean().reset_index()
    grouped.rename(columns={'week': 'visit_date'}, inplace=True)  # Optional: rename 'week' to 'visit_date' for consistency
    grouped[feedback_cols] = grouped[feedback_cols].round(1)  # Round to one decimal place
    return grouped

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Save raw filtered data
            filtered_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw.json")
            filtered_df.to_json(filtered_path, orient="records", date_format="iso")

            # Aggregate and save
            agg_df = aggregate_feedback(filtered_df)
            aggregated_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_aggregated.json")
            agg_df.to_json(aggregated_path, orient="records", date_format="iso")

            # Cache aggregated
            hospital_aggregated_cache[hospital_id] = agg_df

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [filtered_path, aggregated_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(agg_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data/{hospital_id}")
async def get_hospital_data(hospital_id: str):
    if hospital_id not in hospital_data_cache:
        return JSONResponse(status_code=404, content={"error": "No data found for this hospital"})
    df = hospital_data_cache[hospital_id].copy()
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated/{hospital_id}")
async def get_aggregated_data(hospital_id: str):
    if hospital_id not in hospital_aggregated_cache:
        return JSONResponse(status_code=404, content={"error": "No aggregated data found for this hospital"})
    df = hospital_aggregated_cache[hospital_id].copy()
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))
'''

'''
#code upto influxdb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

app = FastAPI()

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}

# Directory to store output files
OUTPUT_DIR = "hospital_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# InfluxDB setup
INFLUXDB_TOKEN = "1sg0BfTOZ4DCyWn_Y68vHEgUnfqty-TuG4V3iT6_NZOd_w0j8tSJH6YfD9fQvNSAW6yS2fMjeQlUIw2n7MTT6A=="
INFLUXDB_ORG = "aismartlive"
INFLUXDB_BUCKET = "pro"
INFLUXDB_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago].copy()
    return filtered_df

# Push to InfluxDB Cloud
def push_to_influx(hospital_id: str, df: pd.DataFrame):
    for _, row in df.iterrows():
        try:
            point = Point("hospital_feedback") \
                .tag("hospital_id", hospital_id) \
                .tag("department", str(row.get("department", "Unknown"))) \
                .field("ease_of_appointment", float(row.get("ease_of_appointment", 0))) \
                .field("accessibility", float(row.get("accessibility", 0))) \
                .field("ease_of_finding", float(row.get("ease_of_finding", 0))) \
                .field("parking", float(row.get("parking", 0))) \
                .field("atmosphere", float(row.get("atmosphere", 0))) \
                .field("receptionist", float(row.get("receptionist", 0))) \
                .field("seat_availability", float(row.get("seat_availability", 0))) \
                .field("waiting_time", float(row.get("waiting_time", 0))) \
                .field("doctor_approach", float(row.get("doctor_approach", 0))) \
                .field("doubt_clearing", float(row.get("doubt_clearing", 0))) \
                .field("problem_explanation", float(row.get("problem_explanation", 0))) \
                .field("overall_satisfaction", float(row.get("overall_satisfaction", 0))) \
                .time(row["visit_date"].isoformat())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"❌ Error writing point for {hospital_id}: {e}")

# Export DataFrame to CSV
def export_to_csv(hospital_id: str, df: pd.DataFrame):
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Export raw filtered data
            raw_csv_path = export_to_csv(hospital_id, filtered_df)

            # Push to InfluxDB
            push_to_influx(hospital_id, filtered_df)

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [raw_csv_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data/{hospital_id}")
async def get_hospital_data(hospital_id: str):
    if hospital_id not in hospital_data_cache:
        return JSONResponse(status_code=404, content={"error": "No data found for this hospital"})
    df = hospital_data_cache[hospital_id].copy()
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated/{hospital_id}")
async def get_aggregated_data(hospital_id: str):
    if hospital_id not in hospital_data_cache:
        return JSONResponse(status_code=404, content={"error": "No data found for this hospital"})

    df = hospital_data_cache[hospital_id].copy()

    if df.empty:
        return JSONResponse(status_code=404, content={"error": "No records to aggregate"})

    numeric_cols = df.select_dtypes(include=["number"]).columns
    agg_df = df.groupby(["visit_date", "department"])[numeric_cols].mean().reset_index()
    agg_df[numeric_cols] = agg_df[numeric_cols].round(1)
    agg_df["visit_date"] = agg_df["visit_date"].astype(str)

    return JSONResponse(content=agg_df.to_dict(orient="records"))
'''

'''
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

app = FastAPI()

# OAuth2 setup for hospital login (mocked users for demo)
users_db = {
    "Hospital_A": {"username": "Hospital_A", "password": "secret"},
    "Hospital_B": {"username": "Hospital_B", "password": "secret"}
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx",
    "Hospital_B": "https://docs.google.com/spreadsheets/d/1BMlV9imgOrIxVHMRxParAzbvTCW-UesF/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}

# Directory to store output files
OUTPUT_DIR = "hospital_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Toggle for Cloud or Local
USE_INFLUXDB_CLOUD = False  # Set True if using Cloud

if USE_INFLUXDB_CLOUD:
    INFLUXDB_TOKEN = "your_cloud_token"
    INFLUXDB_ORG = "your_cloud_org"
    INFLUXDB_BUCKET = "your_cloud_bucket"
    INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
else:
    INFLUXDB_TOKEN = "1sg0BfTOZ4DCyWn_Y68vHEgUnfqty-TuG4V3iT6_NZOd_w0j8tSJH6YfD9fQvNSAW6yS2fMjeQlUIw2n7MTT6A=="
    INFLUXDB_ORG = "aismartlive"
    INFLUXDB_BUCKET = "pro"
    INFLUXDB_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Dependency to extract and validate current hospital user from token
async def get_current_hospital(token: str = Depends(oauth2_scheme)):
    if token in users_db:
        return token
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago].copy()
    return filtered_df

# Push to InfluxDB
def push_to_influx(hospital_id: str, df: pd.DataFrame):
    for _, row in df.iterrows():
        try:
            point = Point("hospital_feedback") \
                .tag("hospital_id", hospital_id) \
                .tag("department", str(row.get("department", "Unknown"))) \
                .field("ease_of_appointment", float(row.get("ease_of_appointment", 0))) \
                .field("accessibility", float(row.get("accessibility", 0))) \
                .field("ease_of_finding", float(row.get("ease_of_finding", 0))) \
                .field("parking", float(row.get("parking", 0))) \
                .field("atmosphere", float(row.get("atmosphere", 0))) \
                .field("receptionist", float(row.get("receptionist", 0))) \
                .field("seat_availability", float(row.get("seat_availability", 0))) \
                .field("waiting_time", float(row.get("waiting_time", 0))) \
                .field("doctor_approach", float(row.get("doctor_approach", 0))) \
                .field("doubt_clearing", float(row.get("doubt_clearing", 0))) \
                .field("problem_explanation", float(row.get("problem_explanation", 0))) \
                .field("overall_satisfaction", float(row.get("overall_satisfaction", 0))) \
                .time(row["visit_date"].isoformat())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"❌ Error writing point for {hospital_id}: {e}")

# Export DataFrame to CSV
def export_to_csv(hospital_id: str, df: pd.DataFrame):
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Export raw filtered data
            raw_csv_path = export_to_csv(hospital_id, filtered_df)

            # Push to InfluxDB
            push_to_influx(hospital_id, filtered_df)

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [raw_csv_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data")
async def get_hospital_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None:
        raise HTTPException(status_code=404, detail="No data found for this hospital")
    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated")
async def get_aggregated_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No records to aggregate")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    agg_df = df.groupby(["visit_date", "department"])[numeric_cols].mean().reset_index()
    agg_df[numeric_cols] = agg_df[numeric_cols].round(1)
    agg_df["visit_date"] = agg_df["visit_date"].astype(str)

    return JSONResponse(content=agg_df.to_dict(orient="records"))

@app.get("/grafana-embed")
async def grafana_iframe(current_hospital: str = Depends(get_current_hospital)):
    dashboard_uid = "eeigzswxw0f0ge"
    grafana_base_url = "http://localhost:3000"
    iframe_code = f"""
    <iframe
        src='{grafana_base_url}/d/{dashboard_uid}?orgId=1&refresh=5m&var-hospital_id={current_hospital}'
        width='100%'
        height='600'
        frameborder='0'>
    </iframe>
    """
    return HTMLResponse(content=iframe_code)
'''
'''
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

app = FastAPI()

# OAuth2 setup for hospital login (mocked users for demo)
users_db = {
    "Hospital_A": {"username": "Hospital_A", "password": "secret"},
    "Hospital_B": {"username": "Hospital_B", "password": "secret"}
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx",
    "Hospital_B": "https://docs.google.com/spreadsheets/d/1BMlV9imgOrIxVHMRxParAzbvTCW-UesF/export?format=xlsx"
}


# Global cache to store data after processing
hospital_data_cache = {}

# Directory to store output files
OUTPUT_DIR = "hospital_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Toggle for Cloud or Local
USE_INFLUXDB_CLOUD = False  # Set True if using Cloud

if USE_INFLUXDB_CLOUD:
    INFLUXDB_TOKEN = "your_cloud_token"
    INFLUXDB_ORG = "your_cloud_org"
    INFLUXDB_BUCKET = "your_cloud_bucket"
    INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
else:
    INFLUXDB_TOKEN = ""
    INFLUXDB_ORG = ""
    INFLUXDB_BUCKET = "pro"
    INFLUXDB_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Dependency to extract and validate current hospital user from token
async def get_current_hospital(token: str = Depends(oauth2_scheme)):
    if token in users_db:
        return token
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago].copy()
    return filtered_df

# Push to InfluxDB
def push_to_influx(hospital_id: str, df: pd.DataFrame):
    for _, row in df.iterrows():
        try:
            point = Point("hospital_feedback") \
                .tag("hospital_id", hospital_id) \
                .tag("department", str(row.get("department", "Unknown"))) \
                .field("ease_of_appointment", float(row.get("ease_of_appointment", 0))) \
                .field("accessibility", float(row.get("accessibility", 0))) \
                .field("ease_of_finding", float(row.get("ease_of_finding", 0))) \
                .field("parking", float(row.get("parking", 0))) \
                .field("atmosphere", float(row.get("atmosphere", 0))) \
                .field("receptionist", float(row.get("receptionist", 0))) \
                .field("seat_availability", float(row.get("seat_availability", 0))) \
                .field("waiting_time", float(row.get("waiting_time", 0))) \
                .field("doctor_approach", float(row.get("doctor_approach", 0))) \
                .field("doubt_clearing", float(row.get("doubt_clearing", 0))) \
                .field("problem_explanation", float(row.get("problem_explanation", 0))) \
                .field("overall_satisfaction", float(row.get("overall_satisfaction", 0))) \
                .time(row["visit_date"].isoformat())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"❌ Error writing point for {hospital_id}: {e}")

# Export DataFrame to CSV
def export_to_csv(hospital_id: str, df: pd.DataFrame):
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Load from backup CSV if cache is missing
def load_from_backup_csv(hospital_id: str) -> pd.DataFrame:
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["visit_date"])
    raise FileNotFoundError("Weekly backup not available")

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Export raw filtered data
            raw_csv_path = export_to_csv(hospital_id, filtered_df)

            # Push to InfluxDB
            push_to_influx(hospital_id, filtered_df)

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [raw_csv_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data")
async def get_hospital_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No data found for this hospital")

    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated")
async def get_aggregated_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None or df.empty:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No records to aggregate")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    agg_df = df.groupby(["visit_date", "department"])[numeric_cols].mean().reset_index()
    agg_df[numeric_cols] = agg_df[numeric_cols].round(1)
    agg_df["visit_date"] = agg_df["visit_date"].astype(str)

    return JSONResponse(content=agg_df.to_dict(orient="records"))

@app.get("/grafana-embed")
async def grafana_iframe(current_hospital: str = Depends(get_current_hospital)):
    dashboard_uid = "your_dashboard_uid"
    grafana_base_url = "https://your-grafana.cloud/grafana"
    iframe_code = f"""
    <iframe
        src='{grafana_base_url}/d/{dashboard_uid}?orgId=1&refresh=5m&var-hospital_id={current_hospital}'
        width='100%'
        height='600'
        frameborder='0'>
    </iframe>
    """
    return HTMLResponse(content=iframe_code)'''

'''
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

app = FastAPI()

# OAuth2 setup for hospital login (mocked users for demo)
users_db = {
    "Hospital_A": {"username": "Hospital_A", "password": "secret"},
    "Hospital_B": {"username": "Hospital_B", "password": "secret"}
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}

# Directory to store output files
OUTPUT_DIR = "hospital_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Toggle for Cloud or Local
USE_INFLUXDB_CLOUD = False  # Set True if using Cloud

if USE_INFLUXDB_CLOUD:
    INFLUXDB_TOKEN = "your_cloud_token"
    INFLUXDB_ORG = "your_cloud_org"
    INFLUXDB_BUCKET = "your_cloud_bucket"
    INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
else:
    INFLUXDB_TOKEN = "1sg0BfTOZ4DCyWn_Y68vHEgUnfqty-TuG4V3iT6_NZOd_w0j8tSJH6YfD9fQvNSAW6yS2fMjeQlUIw2n7MTT6A=="
    INFLUXDB_ORG = "aismartlive"
    INFLUXDB_BUCKET = "pro"
    INFLUXDB_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Dependency to extract and validate current hospital user from token
async def get_current_hospital(token: str = Depends(oauth2_scheme)):
    if token in users_db:
        return token
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

@app.get("/login")
async def login_page():
    html = """
    <html>
        <head><title>Hospital Login</title></head>
        <body style="font-family:sans-serif;padding:40px">
            <h2>Hospital Management Login</h2>
            <form action="/token" method="post">
                <label for="username">Hospital ID:</label><br>
                <input type="text" id="username" name="username" /><br><br>
                <label for="password">Password:</label><br>
                <input type="password" id="password" name="password" /><br><br>
                <input type="submit" value="Login" />
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html)

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago].copy()
    return filtered_df

# Push to InfluxDB
def push_to_influx(hospital_id: str, df: pd.DataFrame):
    for _, row in df.iterrows():
        try:
            point = Point("hospital_feedback") \
                .tag("hospital_id", hospital_id) \
                .tag("department", str(row.get("department", "Unknown"))) \
                .field("ease_of_appointment", float(row.get("ease_of_appointment", 0))) \
                .field("accessibility", float(row.get("accessibility", 0))) \
                .field("ease_of_finding", float(row.get("ease_of_finding", 0))) \
                .field("parking", float(row.get("parking", 0))) \
                .field("atmosphere", float(row.get("atmosphere", 0))) \
                .field("receptionist", float(row.get("receptionist", 0))) \
                .field("seat_availability", float(row.get("seat_availability", 0))) \
                .field("waiting_time", float(row.get("waiting_time", 0))) \
                .field("doctor_approach", float(row.get("doctor_approach", 0))) \
                .field("doubt_clearing", float(row.get("doubt_clearing", 0))) \
                .field("problem_explanation", float(row.get("problem_explanation", 0))) \
                .field("overall_satisfaction", float(row.get("overall_satisfaction", 0))) \
                .time(row["visit_date"].isoformat())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"❌ Error writing point for {hospital_id}: {e}")

# Export DataFrame to CSV
def export_to_csv(hospital_id: str, df: pd.DataFrame):
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Load from backup CSV if cache is missing
def load_from_backup_csv(hospital_id: str) -> pd.DataFrame:
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["visit_date"])
    raise FileNotFoundError("Weekly backup not available")

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Export raw filtered data
            raw_csv_path = export_to_csv(hospital_id, filtered_df)

            # Push to InfluxDB
            push_to_influx(hospital_id, filtered_df)

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [raw_csv_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data")
async def get_hospital_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No data found for this hospital")

    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated")
async def get_aggregated_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None or df.empty:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No records to aggregate")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    agg_df = df.groupby(["visit_date", "department"])[numeric_cols].mean().reset_index()
    agg_df[numeric_cols] = agg_df[numeric_cols].round(1)
    agg_df["visit_date"] = agg_df["visit_date"].astype(str)

    return JSONResponse(content=agg_df.to_dict(orient="records"))

@app.get("/grafana-embed")
async def grafana_iframe(current_hospital: str = Depends(get_current_hospital)):
    dashboard_uid = "eeigzswxw0f0ge"
    grafana_base_url = "http://localhost:3000"
    iframe_code = f"""
    <iframe
        src='{grafana_base_url}/d/{dashboard_uid}?orgId=1&refresh=5m&var-hospital_id={current_hospital}'
        width='100%'
        height='600'
        frameborder='0'>
    </iframe>
    """
    return HTMLResponse(content=iframe_code)
from fastapi.staticfiles import StaticFiles

app.mount("/web", StaticFiles(directory="frontend"), name="frontend")'''


from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import os
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# OAuth2 setup for hospital login (mocked users for demo)
users_db = {
    "Hospital_A": {"username": "Hospital_A", "password": "secret"},
    "Hospital_B": {"username": "Hospital_B", "password": "secret"}
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Hospital sheet URLs
HOSPITAL_DATA_URLS = {
    "Hospital_A": "https://docs.google.com/spreadsheets/d/120etRhAKekBcPf8HgBuZI90xkie7qHxmV56KTEup0bY/export?format=xlsx"
}

# Global cache to store data after processing
hospital_data_cache = {}

# Directory to store output files
OUTPUT_DIR = "hospital_feedback_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scheduler setup to run weekly
scheduler = AsyncIOScheduler()

# Toggle for Cloud or Local
USE_INFLUXDB_CLOUD = True  # Set True if using Cloud

if USE_INFLUXDB_CLOUD:
    INFLUXDB_TOKEN = "vgthlMcGKILtUqin3_sKYUeoVmaJsseezJDxYrQCktwthXzKeFd87-ySfNu0W-GeJBriiufkDiWcCsA4osRANw=="
    INFLUXDB_ORG = "aismartlive"
    INFLUXDB_BUCKET = "feedback"
    INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
else:
    INFLUXDB_TOKEN = "1sg0BfTOZ4DCyWn_Y68vHEgUnfqty-TuG4V3iT6_NZOd_w0j8tSJH6YfD9fQvNSAW6yS2fMjeQlUIw2n7MTT6A=="
    INFLUXDB_ORG = "aismartlive"
    INFLUXDB_BUCKET = "pro"
    INFLUXDB_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Mount static login UI for management
app.mount("/web", StaticFiles(directory="templates", html=True), name="web")

# Dependency to extract and validate current hospital user from token
async def get_current_hospital(token: str = Depends(oauth2_scheme)):
    if token in users_db:
        return token
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

# Function to download Google Sheet
async def download_sheet(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
    return df

# Filter for last week's data and clean DataFrame
def filter_last_week_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    one_week_ago = datetime.now() - timedelta(days=7)
    filtered_df = df[df['visit_date'] >= one_week_ago].copy()
    return filtered_df

# Push to InfluxDB
def push_to_influx(hospital_id: str, df: pd.DataFrame):
    for _, row in df.iterrows():
        try:
            point = Point("hospital_feedback") \
                .tag("hospital_id", hospital_id) \
                .tag("department", str(row.get("department", "Unknown"))) \
                .field("ease_of_appointment", float(row.get("ease_of_appointment", 0))) \
                .field("accessibility", float(row.get("accessibility", 0))) \
                .field("ease_of_finding", float(row.get("ease_of_finding", 0))) \
                .field("parking", float(row.get("parking", 0))) \
                .field("atmosphere", float(row.get("atmosphere", 0))) \
                .field("receptionist", float(row.get("receptionist", 0))) \
                .field("seat_availability", float(row.get("seat_availability", 0))) \
                .field("waiting_time", float(row.get("waiting_time", 0))) \
                .field("doctor_approach", float(row.get("doctor_approach", 0))) \
                .field("doubt_clearing", float(row.get("doubt_clearing", 0))) \
                .field("problem_explanation", float(row.get("problem_explanation", 0))) \
                .field("overall_satisfaction", float(row.get("overall_satisfaction", 0))) \
                .time(row["visit_date"].isoformat())
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"❌ Error writing point for {hospital_id}: {e}")

# Export DataFrame to CSV
def export_to_csv(hospital_id: str, df: pd.DataFrame):
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

# Load from backup CSV if cache is missing
def load_from_backup_csv(hospital_id: str) -> pd.DataFrame:
    week_key = datetime.now().strftime('%Y-%W')
    csv_path = os.path.join(OUTPUT_DIR, f"{hospital_id}_raw_{week_key}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["visit_date"])
    raise FileNotFoundError("Weekly backup not available")

# Core function to fetch and process all hospital sheets
async def fetch_and_process_data():
    print("\n📦 Starting fetch and process...")
    results = {}
    for hospital_id, url in HOSPITAL_DATA_URLS.items():
        try:
            df = await download_sheet(url)
            filtered_df = filter_last_week_data(df)
            hospital_data_cache[hospital_id] = filtered_df

            # Export raw filtered data
            raw_csv_path = export_to_csv(hospital_id, filtered_df)

            # Push to InfluxDB
            push_to_influx(hospital_id, filtered_df)

            results[hospital_id] = {
                "status": "success",
                "records_last_7_days": len(filtered_df),
                "saved_files": [raw_csv_path]
            }
            print(f"✅ {hospital_id}: {len(filtered_df)} records")
            print(filtered_df.head())
        except Exception as e:
            results[hospital_id] = {"status": "error", "error": str(e)}
            print(f"❌ {hospital_id} failed: {e}")
    return results

# Scheduler startup
@app.on_event("startup")
async def startup_event():
    print("🔁 Starting scheduler...")
    scheduler.add_job(lambda: asyncio.create_task(fetch_and_process_data()), "cron", day_of_week="mon", hour=0)
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "✅ Hospital Feedback FastAPI is running"}

@app.get("/fetch-data")
async def fetch_data_endpoint():
    result = await fetch_and_process_data()
    return JSONResponse(content=result)

@app.get("/data")
async def get_hospital_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No data found for this hospital")

    df['visit_date'] = df['visit_date'].astype(str)
    return JSONResponse(content=df.to_dict(orient="records"))

@app.get("/aggregated")
async def get_aggregated_data(current_hospital: str = Depends(get_current_hospital)):
    df = hospital_data_cache.get(current_hospital)
    if df is None or df.empty:
        try:
            df = load_from_backup_csv(current_hospital)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No records to aggregate")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    agg_df = df.groupby(["visit_date", "department"])[numeric_cols].mean().reset_index()
    agg_df[numeric_cols] = agg_df[numeric_cols].round(1)
    agg_df["visit_date"] = agg_df["visit_date"].astype(str)

    return JSONResponse(content=agg_df.to_dict(orient="records"))

@app.get("/grafana-embed")
async def grafana_iframe(current_hospital: str = Depends(get_current_hospital)):
    dashboard_uid = "ad43ab65-c7f6-464b-b5d9-1bfc994169f0"
    grafana_base_url = "https://bevaragowtham02.grafana.net"
    iframe_code = f"""
    <iframe
        src='{grafana_base_url}/d/{dashboard_uid}?orgId=1&refresh=5m&var-hospitalId ={current_hospital}'
        width='100%'
        height='600'
        frameborder='0'>
    </iframe>
    """
    return HTMLResponse(content=iframe_code)
