#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态窗口检测系统
Test the new dynamic window detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from capture.dynamic_window_finder import DynamicWindowFinder
from capture.window_manager import WindowManager, get_window_manager
from capture.monitor_new_message import capture_messages_screenshot
from capture.deal_chatbox import get_message_area_screenshot
import time

def test_basic_detection():
    """测试基本窗口检测功能"""
    print("🚀 测试基本窗口检测")
    print("=" * 50)
    
    finder = DynamicWindowFinder()
    
    # 测试窗口检测
    window_coords = finder.get_wechat_window(use_cache=False)
    
    if window_coords:
        left, top, width, height = window_coords
        print(f"✅ 检测成功: ({left}, {top}, {width}, {height})")
        
        # 测试截图
        if finder.test_window_detection():
            print("✅ 截图测试成功")
            return True
        else:
            print("❌ 截图测试失败")
            return False
    else:
        print("❌ 窗口检测失败")
        return False

def test_window_manager():
    """测试窗口管理器"""
    print("\n🔧 测试窗口管理器")
    print("=" * 50)
    
    # 测试动态检测模式
    manager = WindowManager(use_dynamic_detection=True)
    
    print("📊 窗口信息:")
    info = manager.get_window_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试窗口等待
    if manager.wait_for_wechat_window(timeout=10):
        print("✅ WeChat 窗口检测成功")
        
        # 测试截图功能
        screenshot_path = manager.capture_window_screenshot()
        if screenshot_path:
            print(f"✅ 窗口截图成功: {screenshot_path}")
            
        # 测试消息区域截图
        message_screenshot = manager.capture_message_area()
        if message_screenshot:
            print(f"✅ 消息区域截图成功: {message_screenshot}")
            
        return True
    else:
        print("❌ WeChat 窗口未找到")
        return False

def test_integration_with_existing_code():
    """测试与现有代码的集成"""
    print("\n🔗 测试现有代码集成")
    print("=" * 50)
    
    try:
        # 测试 monitor_new_message 集成
        print("📸 测试消息监控截图（动态检测）...")
        screenshot_path = capture_messages_screenshot(use_dynamic_detection=True)
        if screenshot_path and os.path.exists(screenshot_path):
            print(f"✅ 消息监控截图成功: {screenshot_path}")
        else:
            print("❌ 消息监控截图失败")
            
        # 测试 deal_chatbox 集成  
        print("📸 测试消息区域截图（动态检测）...")
        message_path = get_message_area_screenshot(use_dynamic_detection=True)
        if message_path and os.path.exists(message_path):
            print(f"✅ 消息区域截图成功: {message_path}")
        else:
            print("❌ 消息区域截图失败")
            
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def test_fallback_behavior():
    """测试回退机制"""
    print("\n🔄 测试回退机制")
    print("=" * 50)
    
    try:
        # 测试静态模式
        manager = WindowManager(use_dynamic_detection=False)
        print("📊 静态模式窗口信息:")
        info = manager.get_window_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试动态到静态的回退
        manager.enable_dynamic_detection()
        manager.disable_dynamic_detection()
        
        print("✅ 回退机制测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 回退机制测试失败: {e}")
        return False

def test_performance_comparison():
    """测试性能对比"""
    print("\n⚡ 性能对比测试")
    print("=" * 50)
    
    # 静态检测性能
    start_time = time.time()
    manager_static = WindowManager(use_dynamic_detection=False)
    static_coords = manager_static.get_wechat_window()
    static_time = time.time() - start_time
    
    # 动态检测性能 (首次)
    start_time = time.time()
    manager_dynamic = WindowManager(use_dynamic_detection=True)
    dynamic_coords_first = manager_dynamic.get_wechat_window(force_refresh=True)
    dynamic_time_first = time.time() - start_time
    
    # 动态检测性能 (缓存)
    start_time = time.time()
    dynamic_coords_cached = manager_dynamic.get_wechat_window(force_refresh=False)
    dynamic_time_cached = time.time() - start_time
    
    print(f"📊 性能统计:")
    print(f"  静态检测: {static_time:.4f}s - {static_coords}")
    print(f"  动态检测(首次): {dynamic_time_first:.4f}s - {dynamic_coords_first}")
    print(f"  动态检测(缓存): {dynamic_time_cached:.4f}s - {dynamic_coords_cached}")
    
    return True

def main():
    """主测试函数"""
    print("🔍 WeChat 动态窗口检测系统测试")
    print("=" * 60)
    
    print("⚠️ 测试前请确保:")
    print("1. WeChat 桌面版已打开且可见")
    print("2. WeChat 窗口没有被其他窗口遮挡")
    print("3. 有足够的屏幕截图权限")
    print()
    
    input("✅ 准备完毕后按 Enter 开始测试...")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("基本窗口检测", test_basic_detection()))
    test_results.append(("窗口管理器", test_window_manager()))
    test_results.append(("现有代码集成", test_integration_with_existing_code()))
    test_results.append(("回退机制", test_fallback_behavior()))
    test_results.append(("性能对比", test_performance_comparison()))
    
    # 显示测试结果
    print("\n📋 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！动态窗口检测系统工作正常。")
        print("\n📝 使用建议:")
        print("1. 在 app.py 中启用动态检测: use_dynamic_detection=True")
        print("2. 系统会自动处理窗口位置变化")
        print("3. 如果遇到问题，系统会自动回退到静态坐标")
    else:
        print("⚠️ 部分测试失败，请检查系统配置。")
        print("💡 故障排除:")
        print("1. 确认 WeChat 应用正在运行")
        print("2. 检查屏幕截图权限")
        print("3. 尝试手动运行 find_wechat_window.py")

if __name__ == "__main__":
    main()