#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OCR-based WeChat窗口检测
Test OCR-based WeChat window detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from capture.dynamic_window_finder import DynamicWindowFinder
import time
import pyautogui

def test_ocr_detection():
    """测试OCR检测功能"""
    print("🚀 OCR-based WeChat窗口检测测试")
    print("=" * 60)
    
    print("📋 测试前准备:")
    print("1. 确保 WeChat 桌面版已打开且可见")
    print("2. WeChat 窗口包含可识别的中文或英文文字")
    print("3. 窗口没有被其他程序完全遮挡")
    print("4. 系统已安装 EasyOCR (在 requirements.txt 中)")
    
    input("\n✅ 准备完毕后按 Enter 开始测试...")
    
    # 创建检测器实例
    finder = DynamicWindowFinder()
    
    print("\n🔍 开始OCR检测...")
    start_time = time.time()
    
    # 强制刷新缓存进行检测
    window_coords = finder.get_wechat_window(use_cache=False)
    
    detection_time = time.time() - start_time
    
    if window_coords:
        left, top, width, height = window_coords
        print(f"\n✅ 检测成功! (耗时: {detection_time:.2f}秒)")
        print(f"   窗口位置: ({left}, {top})")
        print(f"   窗口尺寸: {width} x {height}")
        
        # 进行详细验证测试
        print("\n🧪 进行验证测试...")
        
        if finder.test_window_detection():
            print("✅ 窗口验证通过")
            
            # 测试截图质量
            print("\n📷 测试截图质量...")
            test_message_area(left, top, width, height)
            
            return True
        else:
            print("❌ 窗口验证失败")
            return False
            
    else:
        print(f"\n❌ 检测失败 (耗时: {detection_time:.2f}秒)")
        print("\n💡 故障排除建议:")
        print("1. 确认 WeChat 应用正在运行且可见")
        print("2. 检查 WeChat 窗口是否包含可识别的中文文字")
        print("3. 尝试切换到 WeChat 窗口使其获得焦点")
        print("4. 确认系统有屏幕截图权限")
        
        return False

def test_message_area(window_left, window_top, window_width, window_height):
    """测试消息区域截图"""
    try:
        # 计算消息区域 (类似现有代码的逻辑)
        msg_area = (
            window_left + 304,  # 左侧栏宽度偏移
            window_top,
            window_width - 304,  # 减去左侧栏宽度
            min(800, window_height)  # 限制高度
        )
        
        print(f"📱 消息区域坐标: {msg_area}")
        
        # 截图测试
        screenshot = pyautogui.screenshot(region=msg_area)
        screenshot_path = "test_message_area_ocr.png"
        screenshot.save(screenshot_path)
        
        print(f"📷 消息区域截图已保存: {screenshot_path}")
        
        # 用OCR验证消息区域内容
        print("🔍 验证消息区域内容...")
        
        import easyocr
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        results = reader.readtext(screenshot_path)
        
        if results:
            print("✅ 消息区域包含文字内容:")
            for (bbox, text, confidence) in results[:5]:  # 显示前5个检测结果
                if confidence > 0.3:
                    print(f"   '{text.strip()}' (置信度: {confidence:.2f})")
        else:
            print("⚠️ 消息区域未检测到文字内容")
            
    except Exception as e:
        print(f"❌ 消息区域测试失败: {e}")

def test_different_positions():
    """测试不同位置的WeChat窗口"""
    print("\n📐 测试窗口位置适应性")
    print("-" * 40)
    
    print("请将 WeChat 窗口移动到屏幕的不同位置进行测试:")
    positions = [
        "左上角", "右上角", "左下角", "右下角", "屏幕中央"
    ]
    
    finder = DynamicWindowFinder()
    results = []
    
    for i, position in enumerate(positions, 1):
        print(f"\n{i}. 请将 WeChat 移动到 {position}")
        input("   移动完成后按 Enter 继续...")
        
        start_time = time.time()
        window_coords = finder.get_wechat_window(use_cache=False)
        detection_time = time.time() - start_time
        
        if window_coords:
            left, top, width, height = window_coords
            print(f"   ✅ 检测成功: ({left}, {top}, {width}, {height}) 耗时:{detection_time:.2f}s")
            results.append((position, True, detection_time, window_coords))
        else:
            print(f"   ❌ 检测失败 耗时:{detection_time:.2f}s")
            results.append((position, False, detection_time, None))
    
    # 汇总结果
    print(f"\n📊 位置适应性测试结果:")
    success_count = 0
    total_time = 0
    
    for position, success, time_taken, coords in results:
        status = "✅" if success else "❌"
        print(f"   {position:<8} {status} {time_taken:.2f}s")
        if success:
            success_count += 1
        total_time += time_taken
    
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"平均耗时: {total_time/len(results):.2f}秒")

def test_performance_comparison():
    """性能对比测试"""
    print("\n⚡ OCR检测性能测试")
    print("-" * 40)
    
    finder = DynamicWindowFinder()
    
    # 多次检测测量平均性能
    times = []
    successes = 0
    
    for i in range(5):
        print(f"第 {i+1} 次检测...")
        start_time = time.time()
        
        window_coords = finder.get_wechat_window(use_cache=False)
        
        detection_time = time.time() - start_time
        times.append(detection_time)
        
        if window_coords:
            successes += 1
            print(f"  ✅ 成功 - {detection_time:.2f}秒")
        else:
            print(f"  ❌ 失败 - {detection_time:.2f}秒")
    
    avg_time = sum(times) / len(times)
    success_rate = successes / len(times) * 100
    
    print(f"\n📈 性能统计:")
    print(f"   成功率: {success_rate:.1f}% ({successes}/{len(times)})")
    print(f"   平均耗时: {avg_time:.2f}秒")
    print(f"   最快: {min(times):.2f}秒")
    print(f"   最慢: {max(times):.2f}秒")

def main():
    """主测试函数"""
    print("🔍 OCR-based WeChat 窗口检测系统测试")
    print("=" * 60)
    
    # 基础检测测试
    if not test_ocr_detection():
        print("\n❌ 基础检测失败，无法继续进行其他测试")
        return
    
    print("\n" + "="*60)
    
    # 询问是否继续进行扩展测试
    choice = input("\n是否进行扩展测试？包含位置适应性和性能测试 (y/N): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        # 位置适应性测试
        test_different_positions()
        
        print("\n" + "="*60)
        
        # 性能测试
        test_performance_comparison()
    
    print("\n✅ 测试完成!")
    print("\n💡 使用提示:")
    print("1. 动态检测现在基于OCR，适用于所有平台")
    print("2. 检测速度取决于屏幕内容复杂度和OCR处理能力")
    print("3. 建议在WeChat窗口包含清晰文字时使用")
    print("4. 如遇到检测问题，可以回退到静态坐标模式")

if __name__ == "__main__":
    main()