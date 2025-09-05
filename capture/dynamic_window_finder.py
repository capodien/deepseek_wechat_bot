#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic WeChat Window Finder
Automatically detects WeChat window location and size across platforms
"""

import platform
import pyautogui
import time
import subprocess
import re
from typing import Optional, Tuple, List, Dict
import json
import os

class DynamicWindowFinder:
    """Cross-platform WeChat window detection system"""
    
    def __init__(self):
        self.system = platform.system()
        self.wechat_window_cache = None
        self.cache_timestamp = 0
        self.cache_duration = 10  # Cache for 10 seconds
        
        # Common WeChat window identifiers
        self.wechat_patterns = {
            'Darwin': ['WeChat', '微信'],  # macOS
            'Windows': ['WeChat', '微信', 'WeChatWin.exe', 'wechat.exe'],
            'Linux': ['WeChat', 'wechat', 'electronic-wechat']
        }
    
    def get_wechat_window(self, use_cache: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """
        Get WeChat window coordinates (left, top, width, height)
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (left, top, width, height) or None if not found
        """
        current_time = time.time()
        
        # Return cached result if valid
        if (use_cache and 
            self.wechat_window_cache and 
            current_time - self.cache_timestamp < self.cache_duration):
            return self.wechat_window_cache
        
        # Try different detection methods
        window_coords = None
        
        if self.system == 'Darwin':
            window_coords = self._find_window_macos()
        elif self.system == 'Windows':
            window_coords = self._find_window_windows()
        else:
            window_coords = self._find_window_linux()
        
        # Fallback methods
        if not window_coords:
            window_coords = self._find_window_by_screenshot()
        
        if not window_coords:
            window_coords = self._find_window_interactive()
        
        # Cache the result
        if window_coords:
            self.wechat_window_cache = window_coords
            self.cache_timestamp = current_time
            
        return window_coords
    
    def _find_window_macos(self) -> Optional[Tuple[int, int, int, int]]:
        """Find WeChat window on macOS using OCR-based detection"""
        try:
            print("🔍 macOS: 使用OCR方式检测WeChat窗口...")
            
            # Take full screen screenshot for analysis
            full_screenshot = pyautogui.screenshot()
            screenshot_path = "temp_fullscreen.png"
            full_screenshot.save(screenshot_path)
            
            # Use OCR to find WeChat UI elements
            import easyocr
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            
            # Look for WeChat-specific text elements
            wechat_indicators = [
                '微信', 'WeChat', '聊天', '通讯录', '发现', '我',
                'Chat', 'Contacts', 'Discover', 'Me',
                '文件传输助手', 'File Transfer'
            ]
            
            results = reader.readtext(screenshot_path, width_ths=0.7, height_ths=0.7)
            
            wechat_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    text_clean = text.strip()
                    for indicator in wechat_indicators:
                        if indicator in text_clean:
                            # Get bounding box coordinates
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            
                            min_x, max_x = min(x_coords), max(x_coords)
                            min_y, max_y = min(y_coords), max(y_coords)
                            
                            wechat_regions.append({
                                'text': text_clean,
                                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                                'confidence': confidence
                            })
                            print(f"✅ 发现WeChat元素: '{text_clean}' 置信度:{confidence:.2f}")
            
            # Clean up temp file
            try:
                os.remove(screenshot_path)
            except:
                pass
            
            if wechat_regions:
                # Estimate window bounds from detected text regions
                window_bounds = self._estimate_window_from_text_regions(wechat_regions, full_screenshot.size)
                return window_bounds
            
            return None
            
        except Exception as e:
            print(f"❌ macOS OCR window detection error: {e}")
            return None
    
    def _find_window_windows(self) -> Optional[Tuple[int, int, int, int]]:
        """Find WeChat window on Windows using OCR-based detection"""
        try:
            print("🔍 Windows: 使用OCR方式检测WeChat窗口...")
            
            # First try process enumeration as it's faster
            try:
                import psutil
                wechat_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'exe']):
                    try:
                        if any(pattern.lower() in proc.info['name'].lower() 
                              for pattern in ['wechat', '微信']):
                            wechat_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if wechat_processes:
                    print(f"✅ 发现 {len(wechat_processes)} 个WeChat进程")
            except ImportError:
                print("⚠️ psutil未安装，跳过进程检测")
            
            # Use OCR to find WeChat window regardless of process detection
            return self._find_window_by_screenshot()
                    
        except Exception as e:
            print(f"❌ Windows window detection error: {e}")
            
        return None
    
    def _find_window_linux(self) -> Optional[Tuple[int, int, int, int]]:
        """Find WeChat window on Linux using xwininfo/wmctrl"""
        try:
            # Try wmctrl first
            try:
                result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if any(pattern.lower() in line.lower() for pattern in self.wechat_patterns['Linux']):
                            window_id = line.split()[0]
                            
                            # Get geometry with xwininfo
                            geo_result = subprocess.run(
                                ['xwininfo', '-id', window_id],
                                capture_output=True, text=True, timeout=5
                            )
                            
                            if geo_result.returncode == 0:
                                coords = self._parse_xwininfo_output(geo_result.stdout)
                                if coords:
                                    return coords
                                    
            except subprocess.SubprocessError:
                pass
            
            # Fallback to xdotool
            try:
                result = subprocess.run(
                    ['xdotool', 'search', '--name', 'WeChat'],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    window_id = result.stdout.strip().split()[0]
                    
                    geo_result = subprocess.run(
                        ['xdotool', 'getwindowgeometry', window_id],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if geo_result.returncode == 0:
                        coords = self._parse_xdotool_output(geo_result.stdout)
                        if coords:
                            return coords
                            
            except subprocess.SubprocessError:
                pass
                
        except Exception as e:
            print(f"❌ Linux window detection error: {e}")
            
        return None
    
    def _parse_xwininfo_output(self, output: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse xwininfo output to extract window coordinates"""
        try:
            lines = output.split('\n')
            x = y = width = height = 0
            
            for line in lines:
                if 'Absolute upper-left X:' in line:
                    x = int(re.search(r': (\d+)', line).group(1))
                elif 'Absolute upper-left Y:' in line:
                    y = int(re.search(r': (\d+)', line).group(1))
                elif 'Width:' in line:
                    width = int(re.search(r': (\d+)', line).group(1))
                elif 'Height:' in line:
                    height = int(re.search(r': (\d+)', line).group(1))
                    
            if all(coord > 0 for coord in [x, y, width, height]):
                return (x, y, width, height)
                
        except (ValueError, AttributeError):
            pass
            
        return None
    
    def _parse_xdotool_output(self, output: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse xdotool output to extract window coordinates"""
        try:
            lines = output.split('\n')
            for line in lines:
                if 'Position:' in line and 'Width:' in line and 'Height:' in line:
                    # Example: Position: 100,200 (screen: 0)  Width: 800  Height: 600
                    pos_match = re.search(r'Position: (\d+),(\d+)', line)
                    size_match = re.search(r'Width: (\d+).*Height: (\d+)', line)
                    
                    if pos_match and size_match:
                        x, y = map(int, pos_match.groups())
                        width, height = map(int, size_match.groups())
                        return (x, y, width, height)
                        
        except (ValueError, AttributeError):
            pass
            
        return None
    
    def _find_window_by_screenshot(self) -> Optional[Tuple[int, int, int, int]]:
        """Fallback: Try to detect WeChat window using OCR on full screenshot"""
        try:
            print("🔍 使用全屏OCR检测WeChat窗口...")
            
            # Take full screen screenshot
            screenshot = pyautogui.screenshot()
            screenshot_path = "temp_fullscreen_fallback.png"
            screenshot.save(screenshot_path)
            
            # Use existing OCR reader or create new one
            try:
                import easyocr
                reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
                
                # Look for WeChat-specific text with broader search
                wechat_indicators = [
                    '微信', 'WeChat', '聊天', '通讯录', '发现', '我',
                    'Chat', 'Contacts', 'Discover', 'Me',
                    '文件传输助手', 'File Transfer',
                    '搜索', 'Search', '设置', 'Settings'
                ]
                
                print("📖 正在进行OCR文字识别...")
                results = reader.readtext(screenshot_path, width_ths=0.5, height_ths=0.5)
                
                wechat_regions = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Lower threshold for fallback
                        text_clean = text.strip()
                        for indicator in wechat_indicators:
                            if indicator in text_clean or text_clean in indicator:
                                # Get bounding box coordinates
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                                
                                min_x, max_x = min(x_coords), max(x_coords)
                                min_y, max_y = min(y_coords), max(y_coords)
                                
                                wechat_regions.append({
                                    'text': text_clean,
                                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                                    'confidence': confidence
                                })
                                print(f"✅ 发现WeChat元素: '{text_clean}' 置信度:{confidence:.2f}")
                                break
                
                # Clean up temp file
                try:
                    os.remove(screenshot_path)
                except:
                    pass
                
                if wechat_regions:
                    # Use the same estimation method as macOS
                    window_bounds = self._estimate_window_from_text_regions(wechat_regions, screenshot.size)
                    if window_bounds:
                        print(f"✅ OCR检测成功: {window_bounds}")
                        return window_bounds
                
                print("❌ OCR未找到足够的WeChat元素")
                return None
                
            except ImportError:
                print("❌ EasyOCR 未安装，无法使用OCR检测")
                return None
                
        except Exception as e:
            print(f"❌ OCR Screenshot detection error: {e}")
            
        return None
    
    def _is_likely_wechat_window(self, image) -> bool:
        """
        Simple heuristic to check if image region looks like WeChat
        This is basic and could be improved with proper template matching
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic checks for WeChat-like characteristics
            pixels = list(image.getdata())
            
            # Check for common WeChat color schemes (white/gray backgrounds, green elements)
            white_count = sum(1 for r, g, b in pixels if r > 240 and g > 240 and b > 240)
            green_count = sum(1 for r, g, b in pixels if g > r and g > b and g > 100)
            
            total_pixels = len(pixels)
            white_ratio = white_count / total_pixels
            green_ratio = green_count / total_pixels
            
            # Simple heuristic: WeChat often has high white content and some green elements
            return white_ratio > 0.3 and green_ratio > 0.01
            
        except Exception:
            return False
    
    def _find_window_interactive(self) -> Optional[Tuple[int, int, int, int]]:
        """Interactive fallback: Ask user to click on WeChat window"""
        try:
            print("\n🔍 自动检测失败，启用交互式检测...")
            print("请确保 WeChat 应用程序可见，然后按照提示操作：")
            print("1. 将鼠标移动到 WeChat 窗口内的任意位置")
            print("2. 3秒后点击鼠标左键")
            
            time.sleep(3)
            print("📍 请现在点击 WeChat 窗口...")
            
            # Wait for click
            start_time = time.time()
            initial_pos = pyautogui.position()
            
            while time.time() - start_time < 10:  # 10 second timeout
                current_pos = pyautogui.position()
                if current_pos != initial_pos:
                    click_x, click_y = current_pos
                    
                    # Try to estimate window bounds from click position
                    estimated_window = self._estimate_window_from_click(click_x, click_y)
                    if estimated_window:
                        print(f"✅ 估算窗口位置: {estimated_window}")
                        return estimated_window
                    
                time.sleep(0.1)
            
            print("⏰ 交互式检测超时")
            return None
            
        except Exception as e:
            print(f"❌ Interactive detection error: {e}")
            return None
    
    def _estimate_window_from_text_regions(self, wechat_regions: list, screen_size: tuple) -> Optional[Tuple[int, int, int, int]]:
        """Estimate WeChat window bounds from OCR detected text regions"""
        if not wechat_regions:
            return None
        
        try:
            screen_width, screen_height = screen_size
            
            # Filter out non-WeChat content before processing
            filtered_regions = self._filter_non_wechat_content(wechat_regions)
            if not filtered_regions:
                print("⚠️ 所有检测到的文本都被过滤，可能包含混合内容")
                return None
            
            # Get all detected region coordinates
            all_x_coords = []
            all_y_coords = []
            
            for region in filtered_regions:
                x, y, w, h = region['bbox']
                all_x_coords.extend([x, x + w])
                all_y_coords.extend([y, y + h])
            
            if not all_x_coords or not all_y_coords:
                return None
            
            # Calculate bounding rectangle of all WeChat elements
            min_x = min(all_x_coords)
            max_x = max(all_x_coords)
            min_y = min(all_y_coords)
            max_y = max(all_y_coords)
            
            # Add padding to estimate actual window bounds
            # WeChat typically has margins around UI elements
            padding_x = 50
            padding_y = 30
            title_bar_height = 30
            
            # Estimate window bounds
            window_left = max(0, min_x - padding_x)
            window_top = max(0, min_y - title_bar_height)
            window_right = min(screen_width, max_x + padding_x)
            window_bottom = min(screen_height, max_y + padding_y)
            
            window_width = window_right - window_left
            window_height = window_bottom - window_top
            
            # QUICK FIX: Width validation to prevent capturing entire screen
            if not self._validate_window_bounds(window_left, window_top, window_width, window_height):
                return None
            
            # Validate reasonable window size
            if window_width < 300 or window_height < 200:
                # If detected region is too small, use common WeChat sizes
                common_sizes = [(1200, 800), (1000, 700), (900, 600)]
                
                # Find the best size that contains our detected elements
                for width, height in common_sizes:
                    # Try to center the window around detected elements
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    
                    new_left = max(0, min(center_x - width // 2, screen_width - width))
                    new_top = max(0, min(center_y - height // 2, screen_height - height))
                    
                    # Check if this window would contain our detected elements
                    if (new_left <= min_x and new_left + width >= max_x and
                        new_top <= min_y and new_top + height >= max_y):
                        return (new_left, new_top, width, height)
                
                # Fallback to first common size
                width, height = common_sizes[0]
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                window_left = max(0, min(center_x - width // 2, screen_width - width))
                window_top = max(0, min(center_y - height // 2, screen_height - height))
                window_width, window_height = width, height
            
            print(f"📐 估算窗口边界: ({window_left}, {window_top}, {window_width}, {window_height})")
            print(f"   基于 {len(wechat_regions)} 个WeChat UI元素")
            
            return (window_left, window_top, window_width, window_height)
            
        except Exception as e:
            print(f"❌ 窗口边界估算失败: {e}")
            return None
    
    def _estimate_window_from_click(self, click_x: int, click_y: int) -> Optional[Tuple[int, int, int, int]]:
        """Estimate window bounds from a click position inside the window"""
        try:
            # Take screenshot around click area to analyze
            screen_width, screen_height = pyautogui.size()
            
            # Search outward from click point to find window edges
            # This is a simplified approach - could be enhanced
            
            # Common WeChat window sizes
            common_sizes = [(1200, 800), (1000, 700), (900, 600), (800, 600)]
            
            for width, height in common_sizes:
                # Try centering window around click point
                left = max(0, click_x - width // 2)
                top = max(0, click_y - height // 2)
                
                # Adjust if window goes off screen
                if left + width > screen_width:
                    left = screen_width - width
                if top + height > screen_height:
                    top = screen_height - height
                    
                # Ensure click is within estimated window
                if (left <= click_x <= left + width and 
                    top <= click_y <= top + height):
                    return (left, top, width, height)
            
            # Fallback: use default size centered on click
            width, height = 1000, 700
            left = max(0, min(click_x - width // 2, screen_width - width))
            top = max(0, min(click_y - height // 2, screen_height - height))
            
            return (left, top, width, height)
            
        except Exception:
            return None
    
    def test_window_detection(self) -> bool:
        """Test the window detection and take a screenshot with OCR validation"""
        window_coords = self.get_wechat_window(use_cache=False)
        
        if not window_coords:
            print("❌ 无法检测到 WeChat 窗口")
            return False
        
        left, top, width, height = window_coords
        print(f"✅ 检测到 WeChat 窗口: ({left}, {top}, {width}, {height})")
        
        # Take test screenshot
        try:
            test_screenshot = pyautogui.screenshot(region=(left, top, width, height))
            test_path = "wechat_window_test_dynamic.png"
            test_screenshot.save(test_path)
            print(f"📷 测试截图已保存: {test_path}")
            
            # Validate screenshot contains WeChat content using OCR
            if self._validate_wechat_window_ocr(test_path):
                print("✅ OCR验证通过：截图包含WeChat内容")
                return True
            else:
                print("⚠️ OCR验证失败：截图可能不包含WeChat窗口")
                return False
            
        except Exception as e:
            print(f"❌ 截图测试失败: {e}")
            return False
    
    def _validate_wechat_window_ocr(self, screenshot_path: str) -> bool:
        """Use OCR to validate that screenshot contains WeChat content"""
        try:
            import easyocr
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            
            # Read text from screenshot
            results = reader.readtext(screenshot_path, width_ths=0.5, height_ths=0.5)
            
            # Check for WeChat-specific content
            wechat_indicators = [
                '微信', 'WeChat', '聊天', '通讯录', '发现', '我',
                'Chat', 'Contacts', 'Discover', 'Me',
                '文件传输助手', 'File Transfer', '搜索', 'Search'
            ]
            
            found_indicators = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    text_clean = text.strip()
                    for indicator in wechat_indicators:
                        if indicator in text_clean or text_clean in indicator:
                            found_indicators.append((text_clean, confidence))
                            break
            
            if found_indicators:
                print(f"📝 验证发现WeChat元素: {found_indicators}")
                return True
            else:
                print("📝 验证未发现WeChat特征文字")
                return False
                
        except ImportError:
            print("⚠️ EasyOCR 未安装，跳过OCR验证")
            return True  # Assume valid if can't verify
        except Exception as e:
            print(f"⚠️ OCR验证失败: {e}")
            return True  # Assume valid if verification fails
    
    def _validate_window_bounds(self, left: int, top: int, width: int, height: int) -> bool:
        """Validate detected window bounds are reasonable for WeChat"""
        MAX_WECHAT_WIDTH = 1500   # WeChat rarely exceeds this width
        MIN_WECHAT_WIDTH = 600    # Minimum reasonable WeChat width
        MAX_WECHAT_HEIGHT = 1200  # Maximum reasonable WeChat height
        MIN_WECHAT_HEIGHT = 400   # Minimum reasonable WeChat height
        
        # Width validation (primary fix for mixed content issue)
        if width > MAX_WECHAT_WIDTH:
            print(f"⚠️ 窗口宽度过大 ({width}px > {MAX_WECHAT_WIDTH}px)，可能包含混合内容，回退到静态坐标")
            return False
            
        if width < MIN_WECHAT_WIDTH:
            print(f"⚠️ 窗口宽度过小 ({width}px < {MIN_WECHAT_WIDTH}px)，回退到静态坐标")
            return False
        
        # Height validation
        if height > MAX_WECHAT_HEIGHT:
            print(f"⚠️ 窗口高度过大 ({height}px > {MAX_WECHAT_HEIGHT}px)，回退到静态坐标")
            return False
            
        if height < MIN_WECHAT_HEIGHT:
            print(f"⚠️ 窗口高度过小 ({height}px < {MIN_WECHAT_HEIGHT}px)，回退到静态坐标")
            return False
        
        # Aspect ratio validation (WeChat is typically wider than it is tall)
        aspect_ratio = width / height
        if aspect_ratio < 0.8 or aspect_ratio > 3.0:
            print(f"⚠️ 窗口宽高比异常 ({aspect_ratio:.2f})，回退到静态坐标")
            return False
        
        print(f"✅ 窗口边界验证通过: {width}x{height}px, 比例: {aspect_ratio:.2f}")
        return True
    
    def _filter_non_wechat_content(self, wechat_regions: list) -> list:
        """Filter out non-WeChat content from detected text regions"""
        EXCLUDE_KEYWORDS = [
            # Diagnostic panel content
            'diagnostic', 'Current Metrics', 'Performance Optimization',
            'Screenshot', 'Process', 'The bot is detecting', 'OCR processing',
            'reading diagnostic panel', 'Message Change Detection',
            'Verify screenshots', 'contain only WeChat', 'WeChat Window',
            'boundaries', 'Performance', 'Optimization', 'Screenshots',
            
            # Desktop/system content
            'Desktop', 'Finder', 'Safari', 'Chrome', 'Terminal', 'Activity Monitor',
            'System Preferences', 'Applications', 'Documents', 'Downloads',
            
            # Code/development content
            'function', 'return', 'import', 'class', 'def ', 'var ', 'const',
            'python', 'javascript', 'html', 'css', '#!/usr', 'encoding',
            
            # Long diagnostic text patterns (more than 30 chars usually not WeChat UI)
            # Will be checked by length below
        ]
        
        filtered_regions = []
        
        for region in wechat_regions:
            text = region['text'].strip()
            confidence = region['confidence']
            
            # Skip very long text (likely diagnostic content)
            if len(text) > 50:
                print(f"⚠️ 过滤长文本: '{text[:30]}...' (长度: {len(text)})")
                continue
            
            # Skip low confidence with suspicious keywords
            should_exclude = False
            for keyword in EXCLUDE_KEYWORDS:
                if keyword.lower() in text.lower():
                    print(f"⚠️ 过滤关键词匹配: '{text}' (匹配: {keyword})")
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
            
            # Skip text that looks like code or technical content
            if self._looks_like_technical_content(text):
                print(f"⚠️ 过滤技术内容: '{text}'")
                continue
            
            # Only keep text that looks like legitimate WeChat UI elements
            if confidence >= 0.6 or self._is_likely_wechat_ui_text(text):
                filtered_regions.append(region)
                print(f"✅ 保留WeChat元素: '{text}' 置信度:{confidence:.2f}")
            else:
                print(f"⚠️ 过滤低置信度文本: '{text}' 置信度:{confidence:.2f}")
        
        return filtered_regions
    
    def _looks_like_technical_content(self, text: str) -> bool:
        """Check if text looks like technical/code content"""
        technical_indicators = [
            # Programming patterns
            '()', '{}', '[]', '->', '=>', '==', '!=', '&&', '||',
            # File extensions
            '.py', '.js', '.html', '.css', '.json', '.png', '.jpg',
            # Technical terms
            'API', 'HTTP', 'URL', 'JSON', 'XML', 'SQL', 'CSS', 'HTML',
            'screenshot', 'config', 'debug', 'error', 'warning',
            # Coordinate patterns
            'px', 'ms', 'KB', 'MB', 'GB'
        ]
        
        # Check for multiple technical indicators
        indicator_count = sum(1 for indicator in technical_indicators if indicator in text)
        return indicator_count >= 2
    
    def _is_likely_wechat_ui_text(self, text: str) -> bool:
        """Check if text looks like legitimate WeChat UI element"""
        wechat_ui_patterns = [
            # Core WeChat elements
            '微信', 'WeChat', '聊天', '通讯录', '发现', '我',
            'Chat', 'Contacts', 'Discover', 'Me',
            '文件传输助手', 'File Transfer',
            
            # Common WeChat interface elements
            '搜索', 'Search', '设置', 'Settings',
            '消息', 'Messages', '群聊', 'Group',
            '朋友圈', 'Moments', '收藏', 'Favorites',
            
            # Time patterns (common in WeChat)
            ':', '：', '今天', '昨天', 'Today', 'Yesterday',
            
            # Chinese names/text patterns
            '小', '老', '大'  # Common Chinese name prefixes
        ]
        
        return any(pattern in text for pattern in wechat_ui_patterns)
    
    def save_window_config(self, config_file: str = "wechat_window_config.json") -> bool:
        """Save detected window configuration to file"""
        window_coords = self.get_wechat_window()
        
        if not window_coords:
            return False
        
        try:
            config = {
                "wechat_window": window_coords,
                "timestamp": time.time(),
                "system": self.system
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
            print(f"✅ 窗口配置已保存到: {config_file}")
            return True
            
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
            return False
    
    def load_window_config(self, config_file: str = "wechat_window_config.json") -> Optional[Tuple[int, int, int, int]]:
        """Load window configuration from file"""
        try:
            if not os.path.exists(config_file):
                return None
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check if config is recent (within 1 hour)
            if time.time() - config.get("timestamp", 0) < 3600:
                window_coords = tuple(config["wechat_window"])
                print(f"✅ 从配置文件加载窗口: {window_coords}")
                return window_coords
            else:
                print("⚠️ 配置文件过旧，重新检测...")
                
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            
        return None


def main():
    """Test the dynamic window finder"""
    print("🚀 动态 WeChat 窗口检测测试")
    print("=" * 50)
    
    finder = DynamicWindowFinder()
    
    # Try to load existing config first
    window_coords = finder.load_window_config()
    
    if not window_coords:
        # Detect window
        print("🔍 开始检测 WeChat 窗口...")
        window_coords = finder.get_wechat_window(use_cache=False)
    
    if window_coords:
        left, top, width, height = window_coords
        print(f"\n✅ WeChat 窗口坐标: ({left}, {top}, {width}, {height})")
        
        # Test screenshot
        if finder.test_window_detection():
            # Save config for future use
            finder.save_window_config()
            
            print(f"\n📝 Constants.py 更新代码:")
            print(f"   WECHAT_WINDOW = ({left}, {top}, {width}, {height})")
        else:
            print("❌ 窗口检测失败")
    else:
        print("❌ 未能检测到 WeChat 窗口")
        print("请确保:")
        print("1. WeChat 桌面版已打开且可见")
        print("2. WeChat 窗口没有被其他窗口完全遮挡")
        print("3. 运行此脚本时具有必要的系统权限")


if __name__ == "__main__":
    main()