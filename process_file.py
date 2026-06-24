import os
import pandas as pd
import shutil
from datetime import datetime, timedelta
import logging
import time
from tokenization_rules import TokenizationRules
from quantum_tokenizer import QuantumTokenizer
from risk_scoring import RiskScorer
from backend.blockchain_manager import BlockchainManager
from models import TokenMapping, ProcessedFile
from backend.database import db_session
from sqlalchemy import or_

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class FileProcessor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        self.input_dir = os.path.join(data_dir, 'input')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.archive_dir = os.path.join(data_dir, 'archive')
        self.tokenizer = QuantumTokenizer()
        self.blockchain = BlockchainManager()
        for dir_path in [self.input_dir, self.processed_dir, self.archive_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_or_create_token(self, field, value, source_dept, dest_dept):
        try:
            token_mapping = db_session.query(TokenMapping).filter(
                TokenMapping.field_name == field,
                TokenMapping.original_value == str(value),
                or_(
                    TokenMapping.source_dept == source_dept,
                    TokenMapping.source_dept.is_(None)
                ),
                or_(
                    TokenMapping.dest_dept == dest_dept,
                    TokenMapping.dest_dept.is_(None)
                )
            ).first()

            if token_mapping:
                token_mapping.last_used_at = datetime.utcnow()
                token_mapping.usage_count += 1
                db_session.commit()
                return token_mapping.token_value_a

            # Risk scoring
            risk = RiskScorer.compute_risk(field, value, source_dept)
            risk_score = risk['score']
            risk_tier = risk['tier']
            expiry_min = risk['expiry_minutes']
            expires_at = datetime.utcnow() + timedelta(minutes=expiry_min)

            # Generate token pair
            token_a, token_b = self.tokenizer.generate_token_pair()
            token_value_a = f'QT_{token_a[:12]}'
            token_value_b = f'QT_B_{token_b[:12]}'

            # Store on blockchain
            bc = self.blockchain.store_token(
                token_value_a, field, risk_score, expiry_min,
                source_dept, dest_dept
            )

            new_mapping = TokenMapping(
                field_name=field,
                original_value=str(value),
                original_value_truncated=str(value)[:100],
                token_value_a=token_value_a,
                token_value_b=token_value_b,
                risk_score=risk_score,
                risk_tier=risk_tier,
                expiry_minutes=expiry_min,
                expires_at=expires_at,
                blockchain_tx_hash=bc['tx_hash'],
                blockchain_token_id=bc['token_id'],
                blockchain_contract=bc['contract_address'],
                source_dept=source_dept,
                dest_dept=dest_dept,
                usage_count=1,
                last_used_at=datetime.utcnow(),
            )
            db_session.add(new_mapping)
            db_session.commit()

            return token_value_a

        except Exception as e:
            db_session.rollback()
            logging.error(f"Error in get_or_create_token: {str(e)}")
            raise

    def process_file(self, filename):
        start_time = time.time()
        try:
            file_path = os.path.join(self.input_dir, filename)
            logging.info(f"Processing file: {file_path}")

            processed_file = ProcessedFile(
                filename=filename,
                status='processing',
                created_at=datetime.utcnow()
            )
            db_session.add(processed_file)
            db_session.commit()

            source_dept, dest_dept = TokenizationRules.parse_filename(filename)
            if not source_dept or not dest_dept:
                raise ValueError(f"Invalid filename format: {filename}")

            processed_file.source_dept = source_dept
            processed_file.dest_dept = dest_dept

            rules = TokenizationRules.get_rules(source_dept, dest_dept)
            df = pd.read_csv(file_path)
            logging.info(f"Read {len(df)} rows")

            processed_df = df.copy()
            fields_tokenized = {}

            for field in df.columns:
                if field in rules['tokenize']:
                    logging.info(f"Tokenizing field: {field}")
                    fields_tokenized[field] = 0
                    for value in df[field].unique():
                        if pd.notna(value):
                            try:
                                token = self.get_or_create_token(field, value, source_dept, dest_dept)
                                mask = df[field] == value
                                processed_df.loc[mask, field] = token
                                fields_tokenized[field] += int(mask.sum())
                            except Exception as e:
                                logging.error(f"Error tokenizing {field}={value}: {e}")
                else:
                    logging.info(f"Passing through field: {field}")

            output_path = os.path.join(self.processed_dir, f"processed_{filename}")
            processed_df.to_csv(output_path, index=False)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = os.path.join(self.archive_dir, f"{timestamp}_{filename}")
            shutil.move(file_path, archive_path)

            processed_file.records_processed = len(df)
            processed_file.fields_tokenized = fields_tokenized
            processed_file.status = 'success'
            processed_file.processed_at = datetime.utcnow()
            processed_file.processing_time = int((time.time() - start_time) * 1000)
            db_session.commit()

            return True

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            if 'processed_file' in locals():
                processed_file.status = 'error'
                processed_file.error_message = str(e)
                processed_file.processed_at = datetime.utcnow()
                processed_file.processing_time = int((time.time() - start_time) * 1000)
                db_session.commit()
            return False

if __name__ == "__main__":
    try:
        processor = FileProcessor()
        files = [f for f in os.listdir(processor.input_dir) if f.endswith('.csv')]
        for filename in files:
            processor.process_file(filename)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
