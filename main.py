import uvicorn
from dotenv import load_dotenv
from src.api.routes import app
from config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    ) 