#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Debug Tool
Test OCR on the latest screenshot to debug message content extraction
"""

import os
import glob
from capture.deal_chatbox import get_chat_messages
from pprint import pprint

def debug_latest_screenshot():
    """Debug OCR on the latest screenshot"""
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
    print(f"🕐 文件时间: {os.path.getmtime(latest_file)}")
    
    # Test OCR
    try:
        result = get_chat_messages(latest_file)
        print(f"\n🔍 OCR 结果:")
        pprint(result, width=80, depth=3)
        
        if result and 'white' in result:
            print(f"\n✅ 提取到 {len(result['white'])} 条白色消息:")
            for i, msg in enumerate(result['white']):
                print(f"  {i+1}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
        else:
            print("\n❌ 没有提取到白色消息 (用户消息)")
            
        if result and 'green' in result:
            print(f"\n✅ 提取到 {len(result['green'])} 条绿色消息:")
            for i, msg in enumerate(result['green']):
                print(f"  {i+1}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
        
    except Exception as e:
        print(f"❌ OCR 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_latest_screenshot()