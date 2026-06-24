"""
Real Blockchain Manager — QuantEdge
Uses a local Python EVM (eth-tester + py-evm) with a deployed Solidity smart contract.
Every token operation produces a real transaction with on-chain storage.
"""
import json
import os
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from eth_tester import EthereumTester, PyEVMBackend
from web3 import Web3
from web3.types import TxReceipt, Wei

logger = logging.getLogger(__name__)

CONTRACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'contracts')


class BlockchainManager:
    """Manages on-chain token storage via a local EVM + deployed smart contract.
    
    Every call to store_token() produces a real EVM transaction with:
      - Transaction hash (unique, verifiable)
      - Block number (timestamp-ordered)
      - Gas used (cost measurement)
      - Event emission (TokenStored)
    """

    def __init__(self):
        # Initialize local EVM
        self._tester = EthereumTester(backend=PyEVMBackend())
        self.w3 = Web3(Web3.EthereumTesterProvider(self._tester))
        self.w3.eth.default_account = self.w3.eth.accounts[0]

        # Load contract ABI
        abi_path = os.path.join(CONTRACTS_DIR, 'build', 'TokenStore.abi')
        with open(abi_path) as f:
            self.contract_abi = json.load(f)

        # Deploy or load contract
        self.contract_address = self._deploy_contract()
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi,
        )
        logger.info(f"BlockchainManager initialized. Contract: {self.contract_address}")

    def _deploy_contract(self) -> str:
        """Deploy the TokenStore contract to the local EVM."""
        bin_path = os.path.join(CONTRACTS_DIR, 'build', 'TokenStore.bin')
        with open(bin_path) as f:
            bytecode = f.read().strip()

        Contract = self.w3.eth.contract(abi=self.contract_abi, bytecode=bytecode)
        tx_hash = Contract.constructor().transact()
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        addr = receipt['contractAddress']

        # Save deployment info for reference
        deploy_info = {
            'address': addr,
            'tx_hash': tx_hash.hex(),
            'block': receipt['blockNumber'],
            'gas_used': receipt['gasUsed'],
        }
        deploy_path = os.path.join(CONTRACTS_DIR, 'deployment.json')
        with open(deploy_path, 'w') as f:
            json.dump(deploy_info, f, indent=2)
        logger.info(f"Contract deployed at {addr} (tx: {tx_hash.hex()})")
        return addr

    def _compute_token_hash(self, token_value: str, field_name: str) -> bytes:
        """Compute a deterministic hash for on-chain storage."""
        raw = f"{token_value}::{field_name}".encode()
        return hashlib.sha256(raw).digest()

    def store_token(self, token_value: str, field_name: str, risk_score: int,
                    expiry_minutes: int, source_dept: str, dest_dept: str) -> Dict[str, Any]:
        """Store a token on-chain. Returns transaction details."""
        token_hash = self._compute_token_hash(token_value, field_name)

        tx_hash = self.contract.functions.storeToken(
            token_hash, field_name, risk_score, expiry_minutes,
            source_dept, dest_dept,
        ).transact()

        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        token_id = self.contract.functions.getTotalTokens().call() - 1

        result = {
            'token_id': token_id,
            'tx_hash': tx_hash.hex(),
            'block_number': receipt['blockNumber'],
            'gas_used': receipt['gasUsed'],
            'contract_address': self.contract_address,
        }
        logger.info(f"Token {token_value[:16]}... stored on-chain: tx={tx_hash.hex()[:16]}... id={token_id}")
        return result

    def validate_token(self, token_value: str, field_name: str) -> bool:
        """Check if a token hash exists on-chain."""
        token_hash = self._compute_token_hash(token_value, field_name)
        return self.contract.functions.hashExists(token_hash).call()

    def get_token_record(self, token_id: int) -> Optional[Dict]:
        """Retrieve full token record from chain."""
        try:
            record = self.contract.functions.getToken(token_id).call()
            return {
                'token_hash': record[0].hex(),
                'field_name': record[1],
                'risk_score': record[2],
                'timestamp': record[3],
                'expiry_timestamp': record[4],
                'revoked': record[5],
                'source_dept': record[6],
                'dest_dept': record[7],
            }
        except Exception as e:
            logger.error(f"Error reading token {token_id}: {e}")
            return None

    def get_total_tokens(self) -> int:
        return self.contract.functions.getTotalTokens().call()

    def get_active_tokens(self) -> int:
        return self.contract.functions.getActiveTokenCount().call()

    def get_recent_operations(self, minutes: int = 1440) -> int:
        return self.contract.functions.getRecentOperations(minutes).call()

    def revoke_token(self, token_id: int) -> Optional[Dict]:
        """Revoke a token on-chain."""
        try:
            tx_hash = self.contract.functions.revokeToken(token_id).transact()
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            return {
                'tx_hash': tx_hash.hex(),
                'block_number': receipt['blockNumber'],
                'gas_used': receipt['gasUsed'],
            }
        except Exception as e:
            logger.error(f"Error revoking token {token_id}: {e}")
            return None

    def get_gas_summary(self) -> Dict:
        """Return gas usage statistics for research metrics."""
        return {
            'total_tokens': self.get_total_tokens(),
            'active_tokens': self.get_active_tokens(),
            'operations_24h': self.get_recent_operations(1440),
            'node_type': 'py-evm (local)',
            'consensus': 'proof-of-authority (auto)',
        }
