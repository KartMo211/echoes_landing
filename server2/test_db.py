# test_db.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print(f"NEO4J_URI: {uri}")
print(f"NEO4J_USERNAME: {user}")
print(f"NEO4J_PASSWORD: {password}")

print(f"Testing connection to: {uri} with user: {user}")

try:
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        print("✅ Connection SUCCESSFUL!")
except Exception as e:
    print(f"❌ Connection FAILED: {e}")