import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("service_account.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def get_a_user_id():
    users = db.collection("users").limit(1).stream()
    for user in users:
        print(user.id)
        return user.id
    print("No users found")
    return None

if __name__ == "__main__":
    get_a_user_id()
