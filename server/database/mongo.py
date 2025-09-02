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

async def register_user(email: str, password: str):
    existing = await users_collection.find_one({"email": email})
    if existing:
        return False, "Email already registered", None
    
    user_doc = {
        "email": email, 
        "password": password
    }
    
    # MongoDB automatically generates _id, we'll use that as user_id
    result = await users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)
    return True, "User registered successfully", user_id

async def authenticate_user(email: str, password: str):
    user = await users_collection.find_one({"email": email})
    if not user or user["password"] != password:
        return None
    return user

def get_user_book_summary(user_id: str):
    """Get a comprehensive summary of all books for a given user_id (author_id) - SYNCHRONOUS VERSION."""
    try:
        # Use sync pymongo instead of motor for this simple operation
        import pymongo
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        books_collection = db[getattr(config, 'collection_name', 'books')]
        
        # Find all books for this user
        books = list(books_collection.find({"author_id": user_id}))
        
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
        
        client.close()  # Clean up connection
        
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
