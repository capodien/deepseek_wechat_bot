#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Window Finder
Help locate the correct WeChat window coordinates
"""

import pyautogui
import time

def find_wechat_window():
    """Find WeChat window coordinates interactively"""
    print("🔍 WeChat 窗口坐标定位工具")
    print("=" * 50)
    print("请按照以下步骤操作：")
    print("1. 确保 WeChat 桌面版已打开且可见")
    print("2. 将 WeChat 窗口放在屏幕上合适位置")
    print("3. 打开一个有消息的对话（比如 Rio_Old）")
    print("4. 确保聊天界面完全可见")
    print()
    
    input("✅ 准备完毕后按 Enter 继续...")
    print()
    
    print("请按照提示点击 WeChat 窗口的各个位置：")
    print()
    
    # Get WeChat window corners
    coordinates = {}
    
    try:
        # Top-left corner
        print("📍 请点击 WeChat 窗口的 左上角（窗口标题栏左侧）")
        print("   3秒后开始监听鼠标点击...")
        time.sleep(3)
        
        print("👆 请现在点击左上角...")
        while True:
            if pyautogui.onScreen(*pyautogui.position()):
                pos = pyautogui.position()
                print(f"   检测到点击: {pos}")
                coordinates['top_left'] = pos
                break
            time.sleep(0.1)
        
        input("✅ 确认左上角位置正确？按 Enter 继续...")
        
        # Bottom-right corner
        print("\n📍 请点击 WeChat 窗口的 右下角")
        print("   3秒后开始监听鼠标点击...")
        time.sleep(3)
        
        print("👆 请现在点击右下角...")
        while True:
            if pyautogui.onScreen(*pyautogui.position()):
                pos = pyautogui.position()
                print(f"   检测到点击: {pos}")
                coordinates['bottom_right'] = pos
                break
            time.sleep(0.1)
            
        input("✅ 确认右下角位置正确？按 Enter 继续...")
        
        # Calculate window dimensions
        left, top = coordinates['top_left']
        right, bottom = coordinates['bottom_right']
        width = right - left
        height = bottom - top
        
        print(f"\n🎯 检测到的 WeChat 窗口坐标:")
        print(f"   左上角: ({left}, {top})")
        print(f"   右下角: ({right}, {bottom})")
        print(f"   尺寸: {width} x {height}")
        print(f"\n📝 请将以下配置更新到 Constants.py:")
        print(f"   WECHAT_WINDOW = ({left}, {top}, {width}, {height})")
        
        # Take a test screenshot
        test_screenshot = pyautogui.screenshot(region=(left, top, width, height))
        test_path = "wechat_window_test.png"
        test_screenshot.save(test_path)
        print(f"\n📷 测试截图已保存: {test_path}")
        print("   请检查截图是否正确捕获了 WeChat 聊天界面")
        
    except KeyboardInterrupt:
        print("\n❌ 操作被取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    find_wechat_window()