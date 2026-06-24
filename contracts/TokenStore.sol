// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title TokenStore
 * @notice Stores hashes of tokenized data on-chain for immutability audit trail
 */
contract TokenStore {
    address public owner;
    uint256 public tokenCount;

    struct TokenRecord {
        bytes32 tokenHash;
        string fieldName;
        uint256 riskScore;
        uint256 timestamp;
        uint256 expiryTimestamp;
        bool revoked;
        string sourceDept;
        string destDept;
    }

    mapping(uint256 => TokenRecord) public tokens;
    mapping(bytes32 => bool) public hashExists;

    event TokenStored(uint256 indexed id, bytes32 indexed tokenHash, uint256 timestamp, uint256 riskScore);
    event TokenRevoked(uint256 indexed id, bytes32 indexed tokenHash);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        tokenCount = 0;
    }

    function storeToken(
        bytes32 _tokenHash,
        string memory _fieldName,
        uint256 _riskScore,
        uint256 _expiryMinutes,
        string memory _sourceDept,
        string memory _destDept
    ) public onlyOwner returns (uint256) {
        require(!hashExists[_tokenHash], "Token hash already exists");

        uint256 id = tokenCount;
        tokens[id] = TokenRecord({
            tokenHash: _tokenHash,
            fieldName: _fieldName,
            riskScore: _riskScore,
            timestamp: block.timestamp,
            expiryTimestamp: block.timestamp + (_expiryMinutes * 60),
            revoked: false,
            sourceDept: _sourceDept,
            destDept: _destDept
        });

        hashExists[_tokenHash] = true;
        tokenCount++;

        emit TokenStored(id, _tokenHash, block.timestamp, _riskScore);
        return id;
    }

    function revokeToken(uint256 _id) public onlyOwner {
        require(_id < tokenCount, "Token does not exist");
        require(!tokens[_id].revoked, "Already revoked");

        tokens[_id].revoked = true;
        emit TokenRevoked(_id, tokens[_id].tokenHash);
    }

    function getToken(uint256 _id) public view returns (TokenRecord memory) {
        require(_id < tokenCount, "Token does not exist");
        return tokens[_id];
    }

    function isExpired(uint256 _id) public view returns (bool) {
        require(_id < tokenCount, "Token does not exist");
        return block.timestamp > tokens[_id].expiryTimestamp;
    }

    function getTotalTokens() public view returns (uint256) {
        return tokenCount;
    }

    function getActiveTokenCount() public view returns (uint256) {
        uint256 active = 0;
        for (uint256 i = 0; i < tokenCount; i++) {
            if (!tokens[i].revoked && block.timestamp <= tokens[i].expiryTimestamp) {
                active++;
            }
        }
        return active;
    }

    function getRecentOperations(uint256 _minutes) public view returns (uint256) {
        uint256 count = 0;
        uint256 cutoff = block.timestamp - (_minutes * 60);
        for (uint256 i = 0; i < tokenCount; i++) {
            if (tokens[i].timestamp >= cutoff) {
                count++;
            }
        }
        return count;
    }
}
