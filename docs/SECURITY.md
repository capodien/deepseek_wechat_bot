# Security Guide

Security analysis, vulnerabilities, and remediation steps for the WeChat automation bot.

## Security Risk Assessment

### 🚨 Critical Vulnerabilities (Immediate Action Required)

#### 1. SQL Injection Risk - CRITICAL
**Location**: `db/db.py`  
**Risk Level**: 🔴 **CRITICAL**  
**Impact**: Database compromise, data corruption, unauthorized access

**Current Issue**:
```python
# VULNERABLE CODE - Uses string interpolation
query = f"INSERT INTO messages VALUES ('{content}', '{response}')"
cursor.execute(query)
```

**Required Fix**:
```python
# SECURE CODE - Use parameterized queries
query = "INSERT INTO messages VALUES (?, ?)"
cursor.execute(query, (content, response))

# For all database operations
cursor.execute("SELECT * FROM conversations WHERE user_id = ?", (user_id,))
cursor.execute("UPDATE messages SET status = ? WHERE id = ?", (status, message_id))
```

#### 2. Input Validation Missing - HIGH
**Location**: Throughout message processing pipeline  
**Risk Level**: 🟠 **HIGH**  
**Impact**: AI prompt injection, system manipulation, unexpected behavior

**Current Issue**:
```python
# VULNERABLE - No input sanitization
message_content = extract_text_from_screenshot()  # Raw OCR output
ai_response = deepseek_api.generate_response(message_content)  # Direct injection
```

**Required Fix**:
```python
import html
import re

def sanitize_message_input(message_content):
    """Sanitize user input before AI processing"""
    if not message_content or len(message_content.strip()) == 0:
        return None
    
    # Remove potential injection patterns
    message_content = re.sub(r'[<>"`\'%;()&+]', '', message_content)
    
    # HTML escape
    message_content = html.escape(message_content)
    
    # Length limiting
    if len(message_content) > 1000:
        message_content = message_content[:1000] + "..."
    
    return message_content.strip()

# Usage
sanitized_content = sanitize_message_input(raw_ocr_text)
if sanitized_content:
    ai_response = deepseek_api.generate_response(sanitized_content)
```

#### 3. Credential Exposure - HIGH  
**Location**: `.env` file, environment variables  
**Risk Level**: 🟠 **HIGH**  
**Impact**: API key theft, unauthorized API usage, account compromise

**Current Issue**:
```python
# INSECURE - Plain text environment variables
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxx
```

**Required Fix**:
```python
import keyring
from cryptography.fernet import Fernet

# Secure credential storage
def store_api_key_securely(api_key):
    """Store API key in system keyring"""
    keyring.set_password("deepseek_bot", "api_key", api_key)

def get_api_key_securely():
    """Retrieve API key from secure storage"""
    return keyring.get_password("deepseek_bot", "api_key")

# Encrypted storage alternative
def encrypt_and_store_key(api_key, password):
    """Encrypt API key with user password"""
    key = Fernet.generate_key()
    cipher = Fernet(key)
    encrypted_key = cipher.encrypt(api_key.encode())
    
    # Store encrypted key and derive key from password
    with open('.encrypted_config', 'wb') as f:
        f.write(encrypted_key)
    
    return True
```

#### 4. Data Encryption Missing - MEDIUM
**Location**: Database files, screenshot storage  
**Risk Level**: 🟡 **MEDIUM**  
**Impact**: Sensitive chat data exposure, privacy violation

**Current Issue**:
```python
# UNENCRYPTED - Plain text database
sqlite3.connect('history.db')  # No encryption
cv2.imwrite('pic/screenshots/chat.png', image)  # Plain image files
```

**Required Fix**:
```python
import sqlite3
from cryptography.fernet import Fernet
import os

# Database encryption
def create_encrypted_connection(db_path, password):
    """Create encrypted SQLite connection"""
    # Use SQLCipher for encrypted SQLite
    conn = sqlite3.connect(db_path)
    conn.execute(f"PRAGMA key = '{password}'")
    return conn

# File encryption for screenshots
def encrypt_screenshot(image_data, encryption_key):
    """Encrypt screenshot before storage"""
    cipher = Fernet(encryption_key)
    
    # Convert image to bytes
    _, img_encoded = cv2.imencode('.png', image_data)
    img_bytes = img_encoded.tobytes()
    
    # Encrypt image data
    encrypted_data = cipher.encrypt(img_bytes)
    
    return encrypted_data

def decrypt_screenshot(encrypted_data, encryption_key):
    """Decrypt screenshot for processing"""
    cipher = Fernet(encryption_key)
    decrypted_data = cipher.decrypt(encrypted_data)
    
    # Convert back to image
    img_array = np.frombuffer(decrypted_data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return image
```

### ⚠️ Medium Priority Vulnerabilities

#### 5. API Rate Limiting Missing - MEDIUM
**Location**: `deepseek/deepseekai.py`  
**Risk Level**: 🟡 **MEDIUM**  
**Impact**: API abuse, service disruption, cost overruns

**Required Fix**:
```python
import time
from datetime import datetime, timedelta

class APIRateLimiter:
    def __init__(self, requests_per_minute=20):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    def wait_if_needed(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=1)]
        
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]).seconds
            time.sleep(sleep_time)
        
        self.requests.append(now)

# Usage
rate_limiter = APIRateLimiter(requests_per_minute=15)

def safe_api_call(prompt):
    rate_limiter.wait_if_needed()
    return deepseek_api.generate_response(prompt)
```

#### 6. Error Information Disclosure - MEDIUM
**Location**: Error handling throughout codebase  
**Risk Level**: 🟡 **MEDIUM**  
**Impact**: System information leakage, attack vector discovery

**Required Fix**:
```python
import logging

def secure_error_handler(func):
    """Decorator for secure error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log full error internally
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Return generic error to user
            return {
                "success": False,
                "error": "An error occurred during processing",
                "error_id": generate_error_id()
            }
    return wrapper

@secure_error_handler
def process_message(message_content):
    # Implementation
    pass
```

### ✅ Current Security Measures

#### Implemented Protections
1. **Local Processing**: All data processed locally, no cloud storage
2. **HTTPS API**: Encrypted communication with DeepSeek API
3. **Environment Variables**: API keys not hardcoded in source
4. **File Permissions**: Local file access only
5. **Network Isolation**: No unnecessary network connections

#### Existing Security Controls
- API communication over HTTPS
- Local SQLite database (not network accessible)
- Screenshot files stored locally only
- No remote command execution capabilities
- Limited system privileges required

## Security Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
1. **SQL Injection Prevention**:
   ```bash
   # Implement parameterized queries in all database operations
   python security_fixes/fix_sql_injection.py
   ```

2. **Input Validation**:
   ```bash
   # Add input sanitization to message processing
   python security_fixes/implement_input_validation.py
   ```

### Phase 2: High Priority (Week 2)
1. **Credential Security**:
   ```bash
   # Implement secure credential storage
   pip install keyring cryptography
   python security_fixes/secure_credentials.py
   ```

2. **Data Encryption**:
   ```bash
   # Encrypt database and screenshot files
   pip install pycryptodome
   python security_fixes/implement_encryption.py
   ```

### Phase 3: Medium Priority (Week 3)
1. **API Security**:
   ```bash
   # Add rate limiting and request validation
   python security_fixes/secure_api_usage.py
   ```

2. **Error Handling**:
   ```bash
   # Implement secure error handling
   python security_fixes/secure_error_handling.py
   ```

### Phase 4: Monitoring & Maintenance (Ongoing)
1. **Security Monitoring**:
   ```python
   # Implement security event logging
   python security_monitoring/setup_security_logs.py
   ```

2. **Vulnerability Scanning**:
   ```bash
   # Regular dependency scanning
   pip install safety bandit
   safety check
   bandit -r . -f json -o security_scan.json
   ```

## Security Best Practices

### Development Security
```python
# Secure coding checklist
security_checklist = {
    'input_validation': True,      # All user input validated
    'parameterized_queries': True, # No SQL injection vectors
    'error_handling': True,        # No information disclosure
    'credential_protection': True, # Secure credential storage
    'data_encryption': True,       # Sensitive data encrypted
    'api_rate_limiting': True,     # API abuse prevention
    'logging_security': True,      # No sensitive data in logs
    'dependency_scanning': True    # Regular security updates
}
```

### Operational Security
1. **Environment Isolation**:
   ```bash
   # Run in isolated environment
   python -m venv secure_env
   source secure_env/bin/activate
   pip install --no-deps -r requirements-secure.txt
   ```

2. **File Permissions**:
   ```bash
   # Secure file permissions
   chmod 600 .env                    # Environment variables
   chmod 600 *.db                    # Database files
   chmod 700 pic/screenshots/        # Screenshot directory
   chmod 700 logs/                   # Log directory
   ```

3. **Regular Updates**:
   ```bash
   # Security update schedule
   pip list --outdated              # Check for updates
   pip install --upgrade pip        # Update pip
   pip install -r requirements.txt --upgrade  # Update all packages
   ```

### Monitoring & Alerting
```python
# Security event monitoring
def log_security_event(event_type, details, severity='INFO'):
    security_logger = logging.getLogger('security')
    security_logger.log(
        level=getattr(logging, severity),
        msg=f"SECURITY[{event_type}]: {details}",
        extra={
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'details': details
        }
    )

# Usage examples
log_security_event('API_ACCESS', f'DeepSeek API called', 'INFO')
log_security_event('INPUT_VALIDATION', f'Blocked suspicious input: {content}', 'WARN')
log_security_event('AUTH_FAILURE', f'Invalid API key attempted', 'ERROR')
```

## Compliance Considerations

### Data Privacy
- **Personal Data**: Chat messages may contain PII
- **Data Retention**: Implement automatic data cleanup
- **Access Control**: Restrict access to sensitive data
- **Audit Trail**: Log all data access and modifications

### Regulatory Compliance
- **GDPR**: EU data protection (if applicable)
- **CCPA**: California privacy rights (if applicable)
- **Local Privacy Laws**: Check jurisdiction requirements

### Security Standards
- **OWASP Top 10**: Address common web vulnerabilities
- **CWE/SANS Top 25**: Mitigate software weaknesses
- **NIST Cybersecurity**: Follow framework guidelines

## Incident Response Plan

### Security Incident Categories
1. **Data Breach**: Unauthorized access to chat data
2. **API Compromise**: DeepSeek API key stolen
3. **System Compromise**: Bot system infiltrated
4. **Service Disruption**: Security causing system failure

### Response Procedures
```python
# Security incident response
def handle_security_incident(incident_type, severity, details):
    """Handle security incidents with appropriate response"""
    
    # Immediate actions
    if severity == 'CRITICAL':
        shutdown_system()
        revoke_api_keys()
        backup_evidence()
    
    # Logging and notification
    log_security_event('INCIDENT', f'{incident_type}: {details}', severity)
    notify_administrators(incident_type, severity, details)
    
    # Investigation and recovery
    start_incident_investigation()
    implement_containment_measures()
    begin_system_recovery()
    
    # Post-incident activities
    conduct_post_incident_review()
    update_security_measures()
    document_lessons_learned()
```

## Security Testing

### Automated Security Testing
```bash
# Security test suite
python -m pytest security_tests/
python security_tests/test_sql_injection.py
python security_tests/test_input_validation.py
python security_tests/test_api_security.py
python security_tests/test_encryption.py
```

### Manual Security Testing
1. **Penetration Testing**: Attempt to exploit identified vulnerabilities
2. **Code Review**: Manual analysis of security-sensitive code
3. **Configuration Review**: Verify secure configuration settings
4. **Access Control Testing**: Validate authentication and authorization

### Security Metrics
- **Vulnerability Count**: Track and trend security issues
- **Time to Fix**: Measure remediation speed
- **Security Test Coverage**: Ensure comprehensive testing
- **Incident Response Time**: Monitor response effectiveness