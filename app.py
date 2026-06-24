from fastapi import FastAPI, Request, HTTPException, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc
from sqlalchemy.sql import func
import os
from datetime import datetime, timedelta
from models import TokenMapping, ProcessedFile, User
from backend.database import db_session, init_db, Base, engine
from file_watcher import FileWatcher
from quantum_tokenizer import QuantumTokenizer
from tokenization_rules import TokenizationRules
from backend.blockchain_manager import BlockchainManager
import pandas as pd
import logging
import bcrypt
from jose import JWTError, jwt
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

app = FastAPI(title="Quantum-Inspired Tokenization Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class NoCache(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

app.mount("/static", NoCache(directory="static"), name="static")

init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'input')
PROCESSED_FOLDER = os.path.join(DATA_DIR, 'processed')
ARCHIVE_FOLDER = os.path.join(DATA_DIR, 'archive')

for directory in [DATA_DIR, UPLOAD_FOLDER, PROCESSED_FOLDER, ARCHIVE_FOLDER]:
    os.makedirs(directory, exist_ok=True)

tokenizer = QuantumTokenizer()
blockchain = BlockchainManager()
file_watcher = FileWatcher(UPLOAD_FOLDER, PROCESSED_FOLDER, tokenizer, TokenizationRules)

@app.on_event("startup")
async def startup_event():
    file_watcher.start()
    logger.info("File watcher started")

@app.on_event("shutdown")
async def shutdown_event():
    file_watcher.stop()
    logger.info("File watcher stopped")
    db_session.remove()

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except (JWTError, IndexError):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user = db_session.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def seed_sample_data():
    try:
        if db_session.query(TokenMapping).count() > 0:
            return
        sample_mappings = [
            TokenMapping(
                field_name="SSN",
                original_value="123-45-6789",
                token_value="TOK_SSN_001",
                source_dept="HR",
                dest_dept="Finance",
                usage_count=5
            ),
            TokenMapping(
                field_name="Email",
                original_value="john.doe@example.com",
                token_value="TOK_EMAIL_001",
                source_dept="Sales",
                dest_dept="Marketing",
                usage_count=3
            ),
            TokenMapping(
                field_name="Credit Card",
                original_value="4111-1111-1111-1111",
                token_value="TOK_CC_001",
                source_dept="Sales",
                dest_dept="Finance",
                usage_count=2
            )
        ]
        db_session.add_all(sample_mappings)

        sample_files = [
            ProcessedFile(
                filename="employees.csv",
                records_processed=100,
                status="Completed",
                processed_at=datetime.now() - timedelta(hours=1)
            ),
            ProcessedFile(
                filename="customers.csv",
                records_processed=250,
                status="Completed",
                processed_at=datetime.now() - timedelta(hours=2)
            ),
            ProcessedFile(
                filename="transactions.csv",
                records_processed=0,
                status="Failed",
                error_message="Invalid file format",
                processed_at=datetime.now() - timedelta(hours=3)
            )
        ]
        db_session.add_all(sample_files)
        db_session.commit()
        logger.info("Database seeded with sample data")
    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")

seed_sample_data()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = db_session.query(User).filter(User.username == username).first()
    if not user or not bcrypt.checkpw(password.encode(), user.hashed_password.encode()):
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "error": "Invalid username or password"
        })

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {"request": request})

@app.post("/register")
async def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    if db_session.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(request, "register.html", {
            "request": request,
            "error": "Username already exists"
        })

    if db_session.query(User).filter(User.email == email).first():
        return templates.TemplateResponse(request, "register.html", {
            "request": request,
            "error": "Email already exists"
        })

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = User(username=username, email=email, hashed_password=hashed_password)
    db_session.add(user)
    db_session.commit()

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    current_user = get_current_user(request)
    return templates.TemplateResponse(request, "index.html", {"request": request, "user": current_user})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

ALL_FIELDS = ['Name', 'Phone', 'Email', 'Salary', 'Department', 'Position']

def get_file_records_count(file_path):
    try:
        return len(pd.read_csv(file_path))
    except Exception:
        return 0

@app.get("/stats")
async def get_stats(request: Request):
    current_user = get_current_user(request)
    try:
        input_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
        processed_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.csv')]

        total_records = 0
        field_stats = {}

        for filename in processed_files:
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            try:
                df = pd.read_csv(file_path)
                total_records += len(df)
                for field in df.columns:
                    if field not in field_stats:
                        field_stats[field] = {'count': 0, 'unique_values': set()}
                    field_stats[field]['count'] += int(df[field].notna().sum())
                    field_stats[field]['unique_values'].update(df[field].dropna().unique())
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")

        for field in field_stats:
            field_stats[field]['unique_values'] = len(field_stats[field]['unique_values'])

        recent_files = []
        all_files = []

        for filename in processed_files:
            file_path = os.path.join(PROCESSED_FOLDER, filename)
            stat = os.stat(file_path)
            file_info = {
                'filename': filename,
                'status': 'Completed',
                'processed_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'records': get_file_records_count(file_path)
            }
            all_files.append(file_info)

        for filename in input_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            stat = os.stat(file_path)
            file_info = {
                'filename': filename,
                'status': 'Pending',
                'processed_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'records': get_file_records_count(file_path)
            }
            all_files.append(file_info)

        recent_files = sorted(all_files, key=lambda x: x['processed_at'], reverse=True)[:10]

        token_mappings = {field: {} for field in ALL_FIELDS}

        for directory in [ARCHIVE_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(directory):
                if not filename.endswith('.csv'):
                    continue
                file_path = os.path.join(directory, filename)
                try:
                    df = pd.read_csv(file_path)
                    source_dept, dest_dept = TokenizationRules.parse_filename(filename)
                    if not source_dept or not dest_dept:
                        continue
                    rules = TokenizationRules.get_rules(source_dept, dest_dept)
                    if not rules:
                        continue
                    for field in df.columns:
                        if field not in ALL_FIELDS:
                            continue
                        if field in rules['tokenize']:
                            if directory == ARCHIVE_FOLDER:
                                processed_file = os.path.join(PROCESSED_FOLDER, f'processed_{filename}')
                                if os.path.exists(processed_file):
                                    archive_df = df
                                    processed_df = pd.read_csv(processed_file)
                                    for orig_val, token_val in zip(archive_df[field].dropna(), processed_df[field].dropna()):
                                        if str(orig_val) not in token_mappings[field]:
                                            token_mappings[field][str(orig_val)] = str(token_val)
                        elif field in rules['pass_through']:
                            for value in df[field].dropna().unique():
                                if str(value) not in token_mappings[field]:
                                    token_mappings[field][str(value)] = str(value)
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {str(e)}")

        active_tokens = sum(len(mappings) for mappings in token_mappings.values())

        return {
            'total_records': total_records,
            'active_tokens': active_tokens,
            'field_stats': field_stats,
            'recent_files': recent_files,
            'token_mappings': token_mappings
        }
    except Exception as e:
        logging.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.get("/token-mappings")
async def get_all_mappings(request: Request):
    current_user = get_current_user(request)
    try:
        mappings = db_session.query(TokenMapping).order_by(
            TokenMapping.field_name,
            desc(TokenMapping.created_at)
        ).all()

        return [{
            'id': mapping.id,
            'field': mapping.field_name,
            'original': mapping.original_value,
            'original_truncated': mapping.original_value_truncated,
            'token_a': mapping.token_value_a,
            'token_b': mapping.token_value_b,
            'risk_score': mapping.risk_score,
            'risk_tier': mapping.risk_tier,
            'expiry_minutes': mapping.expiry_minutes,
            'expires_at': mapping.expires_at.isoformat() if mapping.expires_at else None,
            'blockchain_tx_hash': mapping.blockchain_tx_hash,
            'blockchain_token_id': mapping.blockchain_token_id,
            'blockchain_contract': mapping.blockchain_contract,
            'is_revoked': mapping.is_revoked,
            'source_dept': mapping.source_dept,
            'dest_dept': mapping.dest_dept,
            'usage_count': mapping.usage_count,
            'created_at': mapping.created_at.isoformat()
        } for mapping in mappings]
    except Exception as e:
        logger.error(f"Error fetching token mappings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch token mappings: {str(e)}")

@app.get("/api/mappings/{field}")
async def get_mappings(field: str, request: Request):
    current_user = get_current_user(request)
    try:
        mappings = db_session.query(TokenMapping).filter(
            TokenMapping.field_name == field
        ).order_by(desc(TokenMapping.created_at)).all()

        return [{
            "original": m.original_value,
            "token": m.token_value,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "last_used": m.last_used_at.isoformat() if m.last_used_at else None,
            "usage_count": m.usage_count or 0
        } for m in mappings]
    except Exception as e:
        logger.error(f"Error getting mappings for field {field}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching mappings")

@app.get("/api/fields")
async def get_fields(request: Request):
    current_user = get_current_user(request)
    try:
        fields = db_session.query(TokenMapping.field_name).distinct().all()
        return [field[0] for field in fields]
    except Exception as e:
        logger.error(f"Error getting fields: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching fields")

@app.get("/processed-files")
async def get_processed_files():
    try:
        files = db_session.query(ProcessedFile).order_by(
            desc(ProcessedFile.processed_at)
        ).all()

        return [{
            "filename": f.filename,
            "status": f.status,
            "records_processed": f.records_processed,
            "processed_at": f.processed_at.isoformat() if f.processed_at else None,
            "error_message": f.error_message,
            "source_dept": f.source_dept,
            "dest_dept": f.dest_dept
        } for f in files]
    except Exception as e:
        logger.error(f"Error getting processed files: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching processed files")

@app.get("/api/files")
async def get_files(request: Request):
    current_user = get_current_user(request)
    try:
        input_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
        processed_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.csv')]
        archive_files = [f for f in os.listdir(ARCHIVE_FOLDER) if f.endswith('.csv')]

        processing_files = db_session.query(ProcessedFile.filename, ProcessedFile.status).all()
        file_status = {f.filename: f.status for f in processing_files}

        return {
            "input": [{"filename": f, "status": file_status.get(f, "pending")} for f in input_files],
            "processed": [{"filename": f, "status": file_status.get(f, "success")} for f in processed_files],
            "archive": archive_files
        }
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching files")

@app.get("/blockchain/stats")
async def get_blockchain_stats(request: Request):
    current_user = get_current_user(request)
    try:
        return blockchain.get_gas_summary()
    except Exception as e:
        logger.error(f"Error fetching blockchain stats: {str(e)}")
        return {"error": str(e), "total_tokens": 0, "active_tokens": 0}
