#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed OCR Debug Tool
"""

import os
import glob
import cv2
import numpy as np
import easyocr
from PIL import Image

def debug_ocr_detailed():
    """Detailed OCR debug with color analysis"""
    # Get the latest screenshot
    screenshot_dirs = ['pic/screenshots/', 'pic/message/']
    latest_file = None
    latest_time = 0
    
    for dir_path in screenshot_dirs:
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, '*.png'))
            for file_path in files:
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path
    
    if not latest_file:
        print("❌ 没有找到截图文件")
        return
    
    print(f"📷 分析最新截图: {latest_file}")
    
    # Load image
    image = cv2.imread(latest_file)
    if image is None:
        print("❌ 无法加载图片")
        return
        
    height, width = image.shape[:2]
    print(f"📐 图片尺寸: {width}x{height}")
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define white color range (for message bubbles)
    lower_white = np.array([0, 0, 200])  # Very light colors
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Define green color range (for sent messages) 
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Analyze color distribution
    white_pixels = np.sum(white_mask == 255)
    green_pixels = np.sum(green_mask == 255)
    total_pixels = height * width
    
    print(f"🎨 颜色分析:")
    print(f"  白色像素: {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    print(f"  绿色像素: {green_pixels} ({green_pixels/total_pixels*100:.1f}%)")
    
    # Find white regions (potential message bubbles)
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_white_regions = [c for c in contours_white if cv2.contourArea(c) > 500]  # Filter small noise
    
    print(f"📝 检测到 {len(large_white_regions)} 个大白色区域 (可能的消息气泡)")
    
    # Try OCR on the full image
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    
    print(f"\n🔤 全图OCR分析:")
    try:
        results = reader.readtext(latest_file)
        print(f"  检测到 {len(results)} 个文本区域")
        
        for i, (bbox, text, confidence) in enumerate(results):
            if confidence > 0.5 and len(text.strip()) > 0:
                print(f"  {i+1}. '{text}' (置信度: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ OCR失败: {e}")
    
    # Save debug image with detected regions
    debug_image = image.copy()
    cv2.drawContours(debug_image, large_white_regions, -1, (0, 255, 0), 2)
    
    debug_path = latest_file.replace('.png', '_debug.png')
    cv2.imwrite(debug_path, debug_image)
    print(f"\n💾 调试图片已保存: {debug_path}")
    print("   (绿色轮廓标记检测到的白色区域)")

if __name__ == "__main__":
    debug_ocr_detailed()