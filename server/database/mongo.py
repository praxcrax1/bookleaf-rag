import motor.motor_asyncio
from config import config

MONGO_URI = config.mongo_uri
DB_NAME = getattr(config, 'db_name', 'Cluster0')

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Helper to get any collection by name
def get_collection(name: str):
    return db[name]

books_collection = get_collection(getattr(config, 'collection_name', 'books'))
users_collection = get_collection('users')

async def get_user_books(author_id: str):
    """Fetch all books for a given author_id."""
    cursor = books_collection.find({"author_id": author_id})
    return await cursor.to_list(length=100)

async def get_or_create_user(author_id: str):
    """Find user by author_id, or create if not exists."""
    user = await users_collection.find_one({"author_id": author_id})
    if not user:
        new_user = {"author_id": author_id, "books": []}
        await users_collection.insert_one(new_user)
        return new_user
    return user

async def register_user(email: str, password: str, author_id: str = None):
    existing = await users_collection.find_one({"email": email})
    if existing:
        return False, "Email already registered"
    user_doc = {"email": email, "password": password}
    if author_id:
        user_doc["author_id"] = author_id
    await users_collection.insert_one(user_doc)
    return True, "User registered successfully"

async def authenticate_user(email: str, password: str):
    user = await users_collection.find_one({"email": email})
    if not user or user["password"] != password:
        return None
    return user

async def get_user_book_summary(user_id: str):
    """Get a comprehensive summary of all books for a given user_id (author_id)."""
    try:
        # Find all books for this user
        cursor = books_collection.find({"author_id": user_id})
        books = await cursor.to_list(length=None)  # Get all books
        
        if not books:
            return {
                "user_id": user_id,
                "total_books": 0,
                "books": [],
                "summary": "No books found for this user."
            }
        
        # Categorize books by status
        status_counts = {}
        book_details = []
        
        for book in books:
            status = book.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            book_details.append({
                "book_id": book.get("book_id"),
                "title": book.get("title"),
                "status": status,
                "stage_notes": book.get("stage_notes", "No notes available")
            })
        
        # Create summary
        total_books = len(books)
        summary_text = f"User has {total_books} book(s). "
        
        if status_counts:
            status_summary = ", ".join([f"{count} {status}" for status, count in status_counts.items()])
            summary_text += f"Status breakdown: {status_summary}."
        
        return {
            "user_id": user_id,
            "total_books": total_books,
            "status_breakdown": status_counts,
            "books": book_details,
            "summary": summary_text
        }
        
    except Exception as e:
        return {
            "user_id": user_id,
            "error": f"Failed to retrieve book summary: {str(e)}"
        }
