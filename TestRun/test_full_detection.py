#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Full Message Detection Flow
Simulate the complete bot detection -> processing flow
"""

from capture.text_change_monitor import detect_new_message_by_text_change
from capture.monitor_new_message import recognize_message
from capture.deal_chatbox import get_message_area_screenshot, get_chat_messages
from capture.get_name_free import get_friend_name
import glob
import os
import pyautogui
import time

def test_full_detection_flow():
    """Test the complete detection and processing flow"""
    print("🤖 测试完整消息检测流程")
    print("=" * 50)
    
    # Step 1: Get latest screenshot (simulating bot monitoring)
    files = glob.glob('pic/screenshots/*.png')
    latest_file = max(files, key=os.path.getmtime)
    print(f"📷 最新截图: {os.path.basename(latest_file)}")
    
    # Step 2: Test new message detection
    print("\n🔍 步骤 1: 检测新消息指示器")
    
    # Try text change detection first
    x, y = detect_new_message_by_text_change(latest_file)
    if x is None and y is None:
        print("   📝 文本变化检测: 未发现新消息")
        
        # Fallback to red dot detection
        print("   🔴 尝试红点检测...")
        red_dot_result = recognize_message(latest_file)
        if red_dot_result and len(red_dot_result) == 2:
            x, y = red_dot_result
            print(f"   ✅ 红点检测成功: 位置 ({x}, {y})")
        else:
            print("   ❌ 红点检测失败")
            return
    else:
        print(f"   ✅ 文本变化检测成功: 位置 ({x}, {y})")
    
    # Step 3: Simulate clicking on the detected message
    print(f"\n🖱️  步骤 2: 模拟点击检测到的位置 ({x}, {y})")
    print("   注意: 这里只是模拟，不会实际点击")
    # pyautogui.click(x, y)  # Commented out for safety
    
    # Step 4: Get friend name
    print("\n👤 步骤 3: 获取联系人名称")
    try:
        # Take a new screenshot after clicking (simulated)
        time.sleep(0.5)  # Brief pause
        name_screenshot = get_message_area_screenshot()
        name = get_friend_name(x, y, name_screenshot)
        print(f"   检测到联系人: {name}")
        
        # Step 5: Extract chat messages
        print(f"\n💬 步骤 4: 提取聊天消息")
        final_result = get_chat_messages(name_screenshot)
        
        if final_result and 'white' in final_result and final_result['white']:
            latest_msg = final_result['white'][-1]
            print(f"   ✅ 提取到最新消息: '{latest_msg}'")
            print(f"   来自: {name}")
            
            # Step 6: Check if this matches monitored contacts
            print(f"\n✔️  步骤 5: 验证监控联系人")
            with open('names.txt', 'r', encoding='utf-8') as f:
                monitored_names = [line.strip() for line in f if line.strip()]
            
            if name in monitored_names:
                print(f"   ✅ '{name}' 在监控列表中")
                print(f"   📤 准备生成AI回复...")
                print(f"   🔒 安全模式: 会输入回复但不自动发送")
                return True
            else:
                print(f"   ❌ '{name}' 不在监控列表中")
                print(f"   监控列表: {monitored_names}")
                return False
        else:
            print("   ❌ 未能提取到有效消息")
            return False
            
    except Exception as e:
        print(f"   ❌ 处理过程出错: {e}")
        return False

if __name__ == "__main__":
    success = test_full_detection_flow()
    print(f"\n{'='*50}")
    if success:
        print("🎉 完整检测流程测试成功! 所有组件都正常工作")
    else:
        print("⚠️  检测流程存在问题，需要调试")