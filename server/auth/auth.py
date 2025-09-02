from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from config import config

# Use config for secret and algorithm
SECRET_KEY = getattr(config, 'jwt_secret', 'your-secret-key')
ALGORITHM = getattr(config, 'jwt_algorithm', 'HS256')
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy decode function for JWT (replace with your logic)
def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_jwt(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
