import requests
from firebase_admin import auth, firestore
from config import db, Config
from datetime import datetime  # Added for timestamp

def authenticate_user(email, password):
    """
    Authenticates a user. Returns (user_data, error_message).
    """
    try:
        # Step 1: Check Firebase first (password optional)
        user = auth.get_user_by_email(email)
        user_doc = db.collection("users").document(user.uid).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return {
                "user_email": email,
                "tenant_id": user_data.get("tenant_id"),
                "external": user_data.get("external", False)
            }, None  # Return None for error_message on success
    except auth.UserNotFoundError:
        pass
    except Exception as e:
        # Return error message instead of printing
        return None, f"Error checking Firebase user: {str(e)}"

    # Step 2: Check LMS authentication if user not found in Firebase (password required)
    if password is None:
        # Return error message for UI display
        return None, "Password is required for LMS authentication."

    try:
        response = requests.post(
            Config.LMS_AUTH_URL,
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                user_data = data["user"]
                tenant_id = user_data["tenant_id"]
                return {
                    "user_email": user_data["email"],
                    "tenant_id": tenant_id,
                    "external": False
                }, None  # Return None for error_message on success
            else:
                # Return LMS-specific error message
                return None, f"LMS authentication failed: {data.get('error')}"
    except Exception as e:
        # Return error message instead of printing
        return None, f"Error authenticating with LMS: {str(e)}"

    # Default error if no authentication succeeds
    return None, "User not found in Firebase or LMS."

def register_external_user(email, password, user_name, tenant_id):
    """
    Registers a new user in Firebase. Returns (user_data, error_message).
    """
    try:
        user = auth.create_user(email=email, password=password, display_name=user_name)

        user_data = {
            "external_user": True,
            "user_email": email,
            "user_name": user_name,
            "tenant_id": tenant_id,
            "is_super": False
        }
        db.collection("users").document(user.uid).set(user_data)

        return user_data, None  # Return None for error_message on success
    except ValueError as e:
        # Catch specific errors (e.g., invalid password) and return as message
        return None, f"Error registering external user: {str(e)}"
    except Exception as e:
        # Return generic error message instead of printing
        return None, f"Error registering external user: {str(e)}"
    


# Moved from main.py
def save_session_to_firestore(user_data, session_id):
    """Save user session to Firestore."""
    db.collection("sessions").document(session_id).set({
        "user_email": user_data["user_email"],
        "tenant_id": user_data.get("tenant_id"),
        "external": user_data.get("external", False),
        "timestamp": datetime.now().isoformat()
    })

# Moved from main.py
def load_session_from_firestore(session_id):
    """Load user session from Firestore."""
    doc = db.collection("sessions").document(session_id).get()
    if doc.exists:
        return doc.to_dict()
    return None

# Moved from main.py
def delete_session_from_firestore(session_id):
    """Delete user session from Firestore."""
    db.collection("sessions").document(session_id).delete()