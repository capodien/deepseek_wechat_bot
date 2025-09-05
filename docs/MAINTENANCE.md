# Maintenance & Best Practices Guide

Development workflows, coding standards, and operational maintenance for the WeChat automation bot.

## Development Workflow

### Pre-Development Checklist
- [ ] Review [`process.md`](../process.md) for current workflow understanding
- [ ] Check existing diagnostic tools in [`DIAGNOSTICS.md`](DIAGNOSTICS.md) for similar features
- [ ] Verify WeChat desktop app is running and positioned correctly
- [ ] Test coordinates in `Constants.py` match current setup
- [ ] Ensure development environment is properly configured

### Development Process
1. **Planning Phase**:
   - Define feature requirements and scope
   - Identify affected system components
   - Design diagnostic testing approach
   - Plan performance impact and metrics

2. **Implementation Phase**:
   - Create diagnostic endpoint in `step_diagnostic_server.py`
   - Add frontend button in `step_diagnostic.html`
   - Implement core functionality with error handling
   - Add visual verification for CV features
   - Document performance characteristics

3. **Testing Phase**:
   - Test error handling and edge cases
   - Verify diagnostic tools work end-to-end
   - Validate performance meets benchmarks
   - Test integration with existing systems

4. **Documentation Phase**:
   - Update `process.md` with workflow changes
   - Update relevant documentation files
   - Document new configuration requirements
   - Add troubleshooting guides for new features

### Post-Development Checklist
- [ ] Update `process.md` with workflow changes
- [ ] Move test files to `TestRun/` directory
- [ ] Verify diagnostic tools work end-to-end
- [ ] Clean root directory of temporary files
- [ ] Test integration with existing systems
- [ ] Validate security considerations addressed

## Code Quality Standards

### Error Handling Patterns
```python
import logging
from functools import wraps

def robust_error_handler(operation_name):
    """Decorator for consistent error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                processing_time = (time.time() - start_time) * 1000
                
                logger.info(f"{operation_name} completed successfully in {processing_time:.2f}ms")
                return result
                
            except FileNotFoundError as e:
                logger.error(f"{operation_name} - File not found: {e}")
                return None
            except cv2.error as e:
                logger.error(f"{operation_name} - OpenCV error: {e}")
                return None
            except Exception as e:
                logger.error(f"{operation_name} - Unexpected error: {e}")
                raise
        return wrapper
    return decorator

# Usage
@robust_error_handler("Screenshot Processing")
def process_screenshot(image_path):
    # Implementation
    pass
```

### Performance Monitoring
```python
import time
import psutil
from functools import wraps

def performance_monitor(func):
    """Monitor function performance and resource usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-execution metrics
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            # Post-execution metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            processing_time = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            logger.info(f"{func.__name__} - Time: {processing_time:.2f}ms, Memory: {memory_delta:+.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} failed after {(time.time() - start_time)*1000:.2f}ms: {e}")
            raise
    return wrapper
```

### Resource Management
```python
import gc
import cv2
from contextlib import contextmanager

@contextmanager
def safe_image_processing():
    """Context manager for safe image processing with cleanup"""
    images_to_cleanup = []
    
    try:
        yield images_to_cleanup
    finally:
        # Cleanup OpenCV resources
        for img in images_to_cleanup:
            if img is not None:
                del img
        cv2.destroyAllWindows()
        gc.collect()

# Usage
def process_multiple_images(image_paths):
    with safe_image_processing() as cleanup_list:
        for path in image_paths:
            img = cv2.imread(path)
            cleanup_list.append(img)
            
            # Process image
            result = analyze_image(img)
            
        return results
```

### Input Validation
```python
import re
import html
from typing import Optional, Union

def validate_message_input(content: str) -> Optional[str]:
    """Validate and sanitize message content"""
    if not content or not isinstance(content, str):
        return None
    
    content = content.strip()
    if len(content) == 0:
        return None
    
    # Remove potential injection patterns
    content = re.sub(r'[<>"`\'%;()&+]', '', content)
    
    # HTML escape
    content = html.escape(content)
    
    # Length limiting
    if len(content) > 1000:
        content = content[:1000] + "..."
    
    return content

def validate_coordinates(x: int, y: int, max_x: int = 3000, max_y: int = 2000) -> bool:
    """Validate screen coordinates"""
    return (
        isinstance(x, int) and isinstance(y, int) and
        0 <= x <= max_x and 0 <= y <= max_y
    )

def validate_api_key(api_key: str) -> bool:
    """Validate DeepSeek API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # DeepSeek API key pattern
    return re.match(r'^sk-[a-zA-Z0-9]{32,}$', api_key) is not None
```

## Function Documentation Standards
```python
def detect_contact_cards(image_path: str, size_range: tuple = (25, 120)) -> list:
    """
    Detect WeChat contact avatar cards using OpenCV adaptive thresholding.
    
    This function processes a WeChat screenshot to identify contact avatar cards
    using adaptive thresholding and contour detection. It filters results based
    on size and aspect ratio constraints to eliminate false positives.
    
    Args:
        image_path (str): Path to WeChat screenshot image file
        size_range (tuple): Min/max avatar size in pixels, default (25, 120)
        
    Returns:
        list: List of dict objects with keys:
            - 'center': (x, y) tuple of avatar center coordinates
            - 'click_pos': (x, y) tuple of calculated click position
            - 'confidence': float confidence score (0.0-1.0)
            - 'bbox': (x, y, w, h) tuple of bounding box coordinates
            
    Raises:
        FileNotFoundError: If image file doesn't exist at specified path
        cv2.error: If image processing fails or image is corrupted
        ValueError: If size_range parameters are invalid
        
    Performance:
        Typical execution time: 200-400ms on GPU, 400-800ms on CPU
        Memory usage: ~50-100MB during processing
        
    Example:
        >>> cards = detect_contact_cards('screenshot.png', size_range=(30, 100))
        >>> print(f"Found {len(cards)} contact cards")
        >>> for card in cards:
        ...     print(f"Avatar at {card['center']}, click at {card['click_pos']}")
        
    Notes:
        - Requires WeChat to be in consistent window layout
        - Performance depends on GPU availability for acceleration
        - Accuracy may decrease with UI theme changes or lighting variations
    """
```

## Testing Framework

### Unit Testing Structure
```python
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

class TestAvatarDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image_path = 'TestRun/test_images/sample_wechat.png'
        self.detector = OpenCVAdaptiveDetector()
    
    def tearDown(self):
        """Clean up test resources"""
        # Clean up any temporary files created during testing
        pass
    
    def test_detect_valid_avatars(self):
        """Test detection of valid avatar cards"""
        results = self.detector.detect_contact_cards(self.test_image_path)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn('center', result)
            self.assertIn('click_pos', result)
            self.assertIn('confidence', result)
            self.assertIsInstance(result['center'], tuple)
            self.assertIsInstance(result['click_pos'], tuple)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths"""
        with self.assertRaises(FileNotFoundError):
            self.detector.detect_contact_cards('nonexistent.png')
    
    @patch('cv2.imread')
    def test_corrupted_image_handling(self, mock_imread):
        """Test handling of corrupted image files"""
        mock_imread.return_value = None
        
        with self.assertRaises(cv2.error):
            self.detector.detect_contact_cards('corrupted.png')

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing
```python
import pytest
import requests
import time

class TestDiagnosticIntegration:
    
    @pytest.fixture(scope="session", autouse=True)
    def setup_diagnostic_server(self):
        """Start diagnostic server for testing"""
        import subprocess
        import time
        
        # Start server in background
        server_process = subprocess.Popen(['python', 'step_diagnostic_server.py'])
        time.sleep(2)  # Allow server to start
        
        yield
        
        # Cleanup
        server_process.terminate()
    
    def test_avatar_detection_endpoint(self):
        """Test avatar detection API endpoint"""
        response = requests.post('http://localhost:5001/api/analyze-contact-cards-opencv')
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'success' in data
        if data['success']:
            assert 'data' in data
            assert 'performance_ms' in data
    
    def test_health_check_endpoint(self):
        """Test system health monitoring"""
        response = requests.get('http://localhost:5001/health')
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert 'status' in health_data
        assert 'components' in health_data
        assert isinstance(health_data['components'], dict)
```

## Logging Configuration

### Production Logging Setup
```python
import logging
import logging.handlers
from datetime import datetime
import os

def setup_production_logging():
    """Configure comprehensive logging for production use"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Main application logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/bot_operations.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Error handler for critical issues
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    
    # Performance handler for timing data
    perf_handler = logging.handlers.RotatingFileHandler(
        'logs/performance.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_format = logging.Formatter('%(asctime)s - %(message)s')
    perf_handler.setFormatter(perf_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    return logger

# Usage
logger = setup_production_logging()
perf_logger = logging.getLogger('performance')
```

### Structured Logging
```python
import json
import logging

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'module': record.name,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        if hasattr(record, 'processing_time'):
            log_data['processing_time_ms'] = record.processing_time
        if hasattr(record, 'accuracy'):
            log_data['accuracy'] = record.accuracy
            
        return json.dumps(log_data)

# Usage
def log_operation_result(operation, processing_time, accuracy=None):
    logger.info(
        f"Operation completed: {operation}",
        extra={
            'operation': operation,
            'processing_time': processing_time,
            'accuracy': accuracy
        }
    )
```

## Maintenance Protocols

### Regular Maintenance Schedule

#### Daily Automated Checks
```python
def daily_health_check():
    """Automated daily system health verification"""
    health_status = {
        'wechat_process': check_wechat_running(),
        'disk_space': check_available_disk_space(),
        'log_rotation': verify_log_rotation(),
        'api_connectivity': test_deepseek_connectivity(),
        'coordinate_accuracy': quick_coordinate_verification(),
        'database_integrity': check_database_integrity()
    }
    
    # Report status
    logger.info(f"Daily health check: {health_status}")
    
    # Alert on issues
    issues = [k for k, v in health_status.items() if not v]
    if issues:
        send_maintenance_alert(f"Health check issues: {issues}")
    
    return all(health_status.values())
```

#### Weekly Maintenance Tasks
```python
def weekly_maintenance():
    """Weekly system maintenance and optimization"""
    tasks = [
        cleanup_old_screenshots,
        optimize_database,
        update_performance_baselines,
        review_error_logs,
        check_dependency_updates,
        backup_configuration_files
    ]
    
    results = {}
    for task in tasks:
        try:
            result = task()
            results[task.__name__] = {'success': True, 'result': result}
        except Exception as e:
            results[task.__name__] = {'success': False, 'error': str(e)}
            logger.error(f"Weekly maintenance task {task.__name__} failed: {e}")
    
    return results

def cleanup_old_screenshots():
    """Remove screenshots older than 7 days"""
    import os
    import time
    
    screenshot_dir = 'pic/screenshots'
    cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
    
    cleaned_files = 0
    for filename in os.listdir(screenshot_dir):
        file_path = os.path.join(screenshot_dir, filename)
        if os.path.getctime(file_path) < cutoff_time:
            os.remove(file_path)
            cleaned_files += 1
    
    logger.info(f"Cleaned up {cleaned_files} old screenshot files")
    return cleaned_files
```

#### Monthly Maintenance Tasks
```python
def monthly_maintenance():
    """Monthly comprehensive system maintenance"""
    return {
        'security_scan': run_security_dependency_scan(),
        'performance_analysis': generate_performance_report(),
        'coordinate_recalibration': suggest_coordinate_updates(),
        'database_optimization': optimize_database_performance(),
        'backup_verification': verify_backup_integrity(),
        'update_documentation': check_documentation_currency()
    }

def run_security_dependency_scan():
    """Scan dependencies for security vulnerabilities"""
    import subprocess
    
    try:
        # Run safety check for known vulnerabilities
        result = subprocess.run(['safety', 'check', '--json'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            return {'vulnerabilities': 0, 'status': 'clean'}
        else:
            return {
                'vulnerabilities': len(result.stdout.split('\n')),
                'details': result.stdout,
                'status': 'issues_found'
            }
    except subprocess.SubprocessError as e:
        logger.error(f"Security scan failed: {e}")
        return {'status': 'scan_failed', 'error': str(e)}
```

### Performance Monitoring

#### Real-time Metrics Collection
```python
import psutil
import time
from collections import deque

class PerformanceMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        
    def record_operation(self, operation_name, processing_time, cpu_usage=None, memory_usage=None):
        """Record performance metrics for an operation"""
        timestamp = time.time()
        
        if cpu_usage is None:
            cpu_usage = psutil.cpu_percent()
        if memory_usage is None:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
        self.cpu_history.append((timestamp, cpu_usage))
        self.memory_history.append((timestamp, memory_usage))
        self.processing_times.append((timestamp, operation_name, processing_time))
        
        # Log to performance logger
        perf_logger.info(f"{operation_name}: {processing_time:.2f}ms, CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}MB")
    
    def get_performance_summary(self):
        """Generate performance summary statistics"""
        if not self.processing_times:
            return None
            
        times = [t[2] for t in self.processing_times]
        cpu_values = [c[1] for c in self.cpu_history]
        memory_values = [m[1] for m in self.memory_history]
        
        return {
            'processing_time': {
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            },
            'cpu_usage': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values)
            },
            'memory_usage': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values)
            },
            'operation_count': len(self.processing_times)
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()
```

### Health Monitoring
```python
def comprehensive_health_check():
    """Comprehensive system health assessment"""
    health_checks = {
        'system_resources': check_system_resources(),
        'wechat_integration': check_wechat_integration(),
        'api_connectivity': check_api_connectivity(),
        'computer_vision': check_cv_functionality(),
        'database_health': check_database_health(),
        'file_system': check_file_system_health(),
        'security_status': check_security_status()
    }
    
    # Calculate overall health score
    passed_checks = sum(1 for status in health_checks.values() if status['healthy'])
    health_score = (passed_checks / len(health_checks)) * 100
    
    return {
        'overall_health_score': health_score,
        'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'unhealthy',
        'checks': health_checks,
        'recommendations': generate_health_recommendations(health_checks)
    }

def check_system_resources():
    """Check system resource availability"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    healthy = (
        cpu_percent < 80 and
        memory.percent < 85 and
        disk.percent < 90
    )
    
    return {
        'healthy': healthy,
        'cpu_usage': cpu_percent,
        'memory_usage': memory.percent,
        'disk_usage': disk.percent,
        'available_memory_gb': memory.available / (1024**3),
        'available_disk_gb': disk.free / (1024**3)
    }
```

## Documentation Maintenance

### Documentation Update Matrix

| Change Type | process.md | CLAUDE.md | SETUP.md | ARCHITECTURE.md | DIAGNOSTICS.md | SECURITY.md | MAINTENANCE.md |
|-------------|------------|-----------|----------|-----------------|----------------|-------------|----------------|
| Workflow modifications | ✅ Required | ⚠️ If architecture changes | ❌ Not needed | ❌ Not needed | ❌ Not needed | ❌ Not needed | ⚠️ If affects process |
| New features | ✅ Required | ✅ Required | ⚠️ If setup changes | ⚠️ If architectural | ✅ Required | ⚠️ If security impact | ⚠️ If maintenance impact |
| Performance changes | ✅ Required | ✅ Update benchmarks | ❌ Not needed | ⚠️ If architectural | ⚠️ If diagnostic impact | ❌ Not needed | ✅ Update monitoring |
| Security updates | ✅ Required | ✅ Update security section | ⚠️ If setup impact | ❌ Not needed | ❌ Not needed | ✅ Required | ⚠️ If procedures change |
| Configuration changes | ✅ Required | ⚠️ If setup changes | ✅ Required | ❌ Not needed | ❌ Not needed | ⚠️ If security impact | ⚠️ If maintenance impact |
| Diagnostic tool additions | ❌ Not needed | ❌ Not needed | ❌ Not needed | ❌ Not needed | ✅ Required | ❌ Not needed | ❌ Not needed |

### Documentation Quality Checklist
```python
def validate_documentation_quality():
    """Check documentation quality and completeness"""
    checks = {
        'technical_accuracy': verify_code_examples_work(),
        'performance_data': verify_performance_benchmarks(),
        'security_current': verify_security_recommendations(),
        'examples_functional': test_documentation_examples(),
        'links_valid': check_internal_documentation_links(),
        'completeness': check_documentation_completeness()
    }
    
    return {
        'overall_quality': sum(checks.values()) / len(checks),
        'individual_checks': checks,
        'recommendations': generate_doc_improvement_suggestions(checks)
    }
```

This comprehensive maintenance guide ensures systematic development practices, robust error handling, performance monitoring, and proactive system maintenance for the WeChat automation bot.