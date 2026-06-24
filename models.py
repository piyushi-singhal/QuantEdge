from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, UniqueConstraint
from datetime import datetime
from backend.database import Base


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(username='{self.username}')>"


class TokenMapping(Base):
    __tablename__ = 'token_mappings'
    
    id = Column(Integer, primary_key=True)
    field_name = Column(String(100), nullable=False)
    original_value = Column(Text, nullable=False)
    original_value_truncated = Column(String(100), nullable=True)
    token_value_a = Column(String(255), nullable=False, unique=True)
    token_value_b = Column(String(255), nullable=True)
    risk_score = Column(Integer, default=50)
    risk_tier = Column(String(20), default='Medium')
    expiry_minutes = Column(Integer, default=30)
    expires_at = Column(DateTime, nullable=True)
    blockchain_tx_hash = Column(String(255), nullable=True)
    blockchain_token_id = Column(Integer, nullable=True)
    blockchain_contract = Column(String(255), nullable=True)
    source_dept = Column(String(100), nullable=False)
    dest_dept = Column(String(100), nullable=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_revoked = Column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint('field_name', 'original_value', 'source_dept', 'dest_dept',
                        name='uix_token_mapping'),
    )

    def __repr__(self):
        return f"<TokenMapping(field='{self.field_name}', risk={self.risk_score})>"


class ProcessedFile(Base):
    __tablename__ = 'processed_files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    records_processed = Column(Integer, default=0)
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    fields_tokenized = Column(JSON, nullable=True)
    processing_time = Column(Integer, nullable=True)
    source_dept = Column(String(100), nullable=True)
    dest_dept = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<ProcessedFile(filename='{self.filename}', status='{self.status}')>"
