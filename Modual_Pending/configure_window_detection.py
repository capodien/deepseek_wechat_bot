#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat 窗口检测配置工具
Configuration utility for WeChat window detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from capture.dynamic_window_finder import DynamicWindowFinder
from capture.window_manager import get_window_manager
import json

def display_menu():
    """显示配置菜单"""
    print("\n🔧 WeChat 窗口检测配置工具")
    print("=" * 50)
    print("1. 🔍 测试动态窗口检测")
    print("2. 📏 手动配置窗口坐标")
    print("3. 💾 保存当前窗口配置")
    print("4. 📂 加载保存的配置")
    print("5. 📊 显示当前配置状态")
    print("6. 🧪 运行完整测试")
    print("7. ❓ 帮助信息")
    print("0. 🚪 退出")
    print("=" * 50)

def test_dynamic_detection():
    """测试动态检测"""
    print("\n🔍 测试动态窗口检测")
    print("-" * 30)
    
    finder = DynamicWindowFinder()
    window_coords = finder.get_wechat_window(use_cache=False)
    
    if window_coords:
        left, top, width, height = window_coords
        print(f"✅ 检测成功!")
        print(f"   窗口坐标: ({left}, {top})")
        print(f"   窗口尺寸: {width} x {height}")
        
        # 保存测试截图
        if finder.test_window_detection():
            print("📷 测试截图已保存: wechat_window_test_dynamic.png")
            return True
    else:
        print("❌ 未能检测到 WeChat 窗口")
        print("💡 请确保:")
        print("   - WeChat 桌面版正在运行")
        print("   - WeChat 窗口可见且未被遮挡")
        print("   - 您有必要的屏幕访问权限")
        return False

def manual_configure():
    """手动配置窗口"""
    print("\n📏 手动配置窗口坐标")
    print("-" * 30)
    
    try:
        print("请输入 WeChat 窗口坐标 (左, 上, 宽, 高):")
        left = int(input("左边距 (Left): "))
        top = int(input("上边距 (Top): "))
        width = int(input("宽度 (Width): "))
        height = int(input("高度 (Height): "))
        
        # 验证输入
        if all(coord > 0 for coord in [left, top, width, height]):
            config = {
                "wechat_window": (left, top, width, height),
                "manual_config": True,
                "timestamp": __import__('time').time()
            }
            
            with open('wechat_window_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
            print(f"✅ 配置已保存: ({left}, {top}, {width}, {height})")
            
            # 测试配置
            import pyautogui
            try:
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                screenshot.save("manual_config_test.png")
                print("📷 测试截图已保存: manual_config_test.png")
            except Exception as e:
                print(f"❌ 测试截图失败: {e}")
                
        else:
            print("❌ 无效的坐标值")
            
    except ValueError:
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 配置失败: {e}")

def save_current_config():
    """保存当前配置"""
    print("\n💾 保存当前窗口配置")
    print("-" * 30)
    
    finder = DynamicWindowFinder()
    if finder.save_window_config():
        print("✅ 配置保存成功")
    else:
        print("❌ 配置保存失败")

def load_saved_config():
    """加载保存的配置"""
    print("\n📂 加载保存的配置")
    print("-" * 30)
    
    finder = DynamicWindowFinder()
    config = finder.load_window_config()
    
    if config:
        left, top, width, height = config
        print(f"✅ 配置加载成功: ({left}, {top}, {width}, {height})")
    else:
        print("❌ 无可用的保存配置")

def show_current_status():
    """显示当前配置状态"""
    print("\n📊 当前配置状态")
    print("-" * 30)
    
    try:
        # 检查动态检测
        window_manager = get_window_manager(use_dynamic=True)
        info = window_manager.get_window_info()
        
        print("🔧 窗口管理器状态:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 检查配置文件
        if os.path.exists('wechat_window_config.json'):
            print("\n📄 配置文件状态:")
            with open('wechat_window_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in config.items():
                print(f"   {key}: {value}")
        else:
            print("\n📄 配置文件: 不存在")
            
    except Exception as e:
        print(f"❌ 状态检查失败: {e}")

def run_full_test():
    """运行完整测试"""
    print("\n🧪 运行完整测试")
    print("-" * 30)
    
    try:
        os.system(f"{sys.executable} test_dynamic_detection.py")
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")

def show_help():
    """显示帮助信息"""
    print("\n❓ 帮助信息")
    print("-" * 30)
    
    help_text = """
🔍 动态窗口检测说明:
   - 自动找到 WeChat 窗口位置和尺寸
   - 支持窗口移动和尺寸变化
   - 跨平台支持 (macOS, Windows, Linux)
   
💡 使用建议:
   1. 首先运行"测试动态窗口检测"确保系统工作正常
   2. 如果动态检测失败，可以使用"手动配置窗口坐标"
   3. 保存配置以便下次快速加载
   
🔧 集成到现有代码:
   在 app.py 中，修改截图调用:
   - capture_messages_screenshot(use_dynamic_detection=True)
   - get_message_area_screenshot(use_dynamic_detection=True)
   
🚨 故障排除:
   - 确保 WeChat 桌面版正在运行
   - 检查屏幕访问权限设置
   - WeChat 窗口应该可见且未被完全遮挡
   - 在 macOS 上可能需要在系统偏好设置中允许屏幕录制
    """
    
    print(help_text)

def main():
    """主配置界面"""
    print("🎯 WeChat 动态窗口检测配置向导")
    
    while True:
        display_menu()
        
        try:
            choice = input("\n请选择操作 (0-7): ").strip()
            
            if choice == '1':
                test_dynamic_detection()
            elif choice == '2':
                manual_configure()
            elif choice == '3':
                save_current_config()
            elif choice == '4':
                load_saved_config()
            elif choice == '5':
                show_current_status()
            elif choice == '6':
                run_full_test()
            elif choice == '7':
                show_help()
            elif choice == '0':
                print("👋 配置工具退出")
                break
            else:
                print("❌ 无效选择，请输入 0-7")
                
        except KeyboardInterrupt:
            print("\n\n👋 配置工具退出")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")

if __name__ == "__main__":
    main()