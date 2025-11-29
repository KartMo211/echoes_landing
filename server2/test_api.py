import asyncio
import httpx
import json

BASE_URL = "https://digital-companion-ai-b6xf.onrender.com"
USERNAME = "testuser_api"
PASSWORD = "testpassword123"
CHARACTER_NAME = "TestBot"

async def test_api():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        print(f"Testing API at {BASE_URL}...\n")

        # 1. Test Root
        print("1. Testing Root Endpoint (GET /)...")
        try:
            resp = await client.get("/")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")
            assert resp.status_code == 200
        except Exception as e:
            print(f"   FAILED: {e}")
            return

        # 2. Test Register
        print("\n2. Testing Registration (POST /register)...")
        try:
            resp = await client.post("/register", json={"username": USERNAME, "password": PASSWORD})
            if resp.status_code == 400 and "already registered" in resp.text:
                print("   User already exists (Expected if running multiple times).")
            else:
                print(f"   Status: {resp.status_code}")
                print(f"   Response: {resp.json()}")
                assert resp.status_code == 200 or resp.status_code == 400
        except Exception as e:
            print(f"   FAILED: {e}")

        # 3. Test Login
        print("\n3. Testing Login (POST /token)...")
        token = None
        user_id = None
        try:
            resp = await client.post("/token", json={"username": USERNAME, "password": PASSWORD})
            print(f"   Status: {resp.status_code}")
            data = resp.json()
            print(f"   Response: {data}")
            assert resp.status_code == 200
            token = data.get("token") # Note: The API returns user_id and username, not a JWT token field in the response model shown in app.py, but let's check.
            # Looking at app.py: return Token(user_id=user["user_id"], username=user["username"])
            # So there is no "token" field, just user_id.
            user_id = data.get("user_id")
            print(f"   Logged in as User ID: {user_id}")
        except Exception as e:
            print(f"   FAILED: {e}")
            return

        if not user_id:
            print("   Login failed to return user_id. Aborting authenticated tests.")
            return

        headers = {"X-User-ID": user_id}

        # 4. Test Create Character
        print("\n4. Testing Create Character (POST /characters)...")
        try:
            # First delete if exists to ensure clean state
            await client.delete(f"/characters/{CHARACTER_NAME}", headers=headers)
            
            files = {'file': ('history.txt', 'User: Hello\nBot: Hi there!', 'text/plain')}
            data = {
                'name': CHARACTER_NAME,
                'description': 'A test bot for API verification.',
                'my_name': 'Tester'
            }
            resp = await client.post("/characters", data=data, files=files, headers=headers)
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")
            assert resp.status_code == 201
        except Exception as e:
            print(f"   FAILED: {e}")

        # 5. Test Get Characters
        print("\n5. Testing Get Characters (GET /characters)...")
        try:
            resp = await client.get("/characters", headers=headers)
            print(f"   Status: {resp.status_code}")
            chars = resp.json()
            print(f"   Response: Found {len(chars)} characters")
            # print(f"   Data: {chars}")
            assert resp.status_code == 200
            assert any(c['name'] == CHARACTER_NAME for c in chars)
        except Exception as e:
            print(f"   FAILED: {e}")

        # 6. Test Chat
        print("\n6. Testing Chat (POST /chat)...")
        try:
            chat_payload = {
                "character_name": CHARACTER_NAME,
                "message": "Hello, who are you?"
            }
            resp = await client.post("/chat", json=chat_payload, headers=headers)
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {resp.json()}")
            assert resp.status_code == 200
        except Exception as e:
            print(f"   FAILED: {e}")

        # 7. Test Delete Character
        print("\n7. Testing Delete Character (DELETE /characters/{name})...")
        try:
            resp = await client.delete(f"/characters/{CHARACTER_NAME}", headers=headers)
            print(f"   Status: {resp.status_code}")
            assert resp.status_code == 204
            print("   Character deleted successfully.")
        except Exception as e:
            print(f"   FAILED: {e}")

        print("\n--- API Tests Completed ---")

if __name__ == "__main__":
    asyncio.run(test_api())
