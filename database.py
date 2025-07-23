# database.py
from pymongo import MongoClient
from auth import hash_password

MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

db = client["chatbot_db"]
session_collection = db["sessions"]
user_collection = db["users"]

if __name__ == "__main__":
    # Tạo user mẫu nếu chưa có
    username = "admin"
    password = "admin123"
    if user_collection.find_one({"username": username}) is None:
        user_collection.insert_one({
            "username": username,
            "password": hash_password(password)
        })
        print(f"Đã tạo user mẫu: {username}/{password}")
    else:
        print(f"User {username} đã tồn tại.")