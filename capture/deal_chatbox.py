# 在原有导入部分新增
import os
import time
from pprint import pprint

import pyautogui

from Constants import Constants
from .window_manager import fget_window_manager

# 电脑版微信全屏状态的窗口区域（保持向后兼容）
WECHAT_WINDOW = Constants.WECHAT_WINDOW
import easyocr
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=True)  # 添加gpu=True参数启用GPU加速


def fextract_text_by_color_flow(image,target_color , tolerance=1):
    """
    修改说明：
    1. 增加区域垂直位置判断逻辑
    2. 返回最下方符合条件的文本区域
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower = np.array([max(0, c - tolerance) for c in target_color])
    upper = np.array([min(255, c + tolerance) for c in target_color])
    mask = cv2.inRange(image, lower, upper)

    # 优化轮廓查找参数
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_bottom = -1
    target_contour = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        current_bottom = y + h  # 计算区域底部Y坐标

        # 过滤过小区域（根据实际场景调整）
        if w > 50 and h > 20:  # 增加最小宽高限制
            if current_bottom > max_bottom:
                max_bottom = current_bottom
                target_contour = (x, y, w, h)

    return target_contour if target_contour is not None else (0, 0, 0, 0)


import cv2
import numpy as np

# 预定义常量（根据实际场景校准）
GREEN_LOWER = np.array([117, 229, 164])  # BGR颜色下限
GREEN_UPPER = np.array([127, 239, 174])  # BGR颜色上限
X_START = 320  # 水平起始坐标
X_END = 1469  # 水平终止坐标
MIN_Y = 43  # 垂直方向最小检测起点
ROI_HEIGHT = 800  # 感兴趣区域高度


def frecognize_green_bottom(image_path):
    """
    性能优化版绿色区域底部检测
    返回：最下方绿色区域的底部Y坐标（全局坐标系），未检测到返回None
    """
    # 闪电加载图像（灰度模式提升读取速度）
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    try:
        # ROI区域裁剪（减少处理面积）
        h, w = image.shape[:2]
        roi_y1 = max(MIN_Y, 0)
        roi_y2 = min(roi_y1 + ROI_HEIGHT, h)

        # 二次校验防止越界
        if roi_y1 >= h or X_END >= w:
            return None

        roi = image[roi_y1:roi_y2, X_START:X_END]

        # 快速颜色阈值处理
        mask = cv2.inRange(roi, GREEN_LOWER, GREEN_UPPER)

        # 垂直方向投影分析
        vertical_projection = np.any(mask, axis=1)
        y_coords = np.where(vertical_projection)[0]

        if y_coords.size == 0:
            return None

        # 计算全局坐标系Y坐标
        bottom_in_roi = y_coords[-1]  # ROI内的相对Y坐标
        global_y = roi_y1 + bottom_in_roi

        # 有效性验证
        if global_y > h:
            return None

        return int(global_y)

    except Exception as e:
        print(f"检测异常: {str(e)}")
        return None

# 内存缓存优化（减少磁盘IO）
from io import BytesIO

def fget_message_area_screenshot_bytes(use_dynamic_detection=True):
    """获取消息区域截图并返回BytesIO对象"""
    # 使用动态窗口检测获取当前窗口坐标
    if use_dynamic_detection:
        try:
            window_manager = fget_window_manager(use_dynamic=True)
            window_coords = window_manager.get_wechat_window()
            
            # 基于动态窗口计算消息区域
            msg_area = (
                window_coords[0] + 304,  # 左边距（微信左侧栏宽度）
                window_coords[1],        # 顶部对齐
                window_coords[2] - 304,  # 宽度减去左侧栏
                min(800, window_coords[3])  # 高度限制或窗口高度
            )
        except Exception as e:
            print(f"❌ 动态检测失败: {e}，回退到静态坐标")
            # 回退到静态坐标
            msg_area = (
                WECHAT_WINDOW[0] + 304,
                WECHAT_WINDOW[1],
                1175,  # 修正宽度 (1479-304)
                800
            )
    else:
        # 使用静态坐标
        msg_area = (
            WECHAT_WINDOW[0] + 304,
            WECHAT_WINDOW[1],
            1175,  # 修正宽度
            800
        )
    
    screenshot = pyautogui.screenshot(region=msg_area)
    # 直接返回BytesIO对象供后续处理
    img_byte_arr = BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def fget_message_area_screenshot(use_dynamic_detection=True):
    """获取消息区域截图，支持动态窗口检测"""
    # 使用动态窗口检测获取当前窗口坐标
    if use_dynamic_detection:
        try:
            window_manager = fget_window_manager(use_dynamic=True)
            window_coords = window_manager.get_wechat_window()
            
            # 基于动态窗口计算消息区域
            msg_area = (
                window_coords[0] + 304,  # 左边距（微信左侧栏宽度）
                window_coords[1],        # 顶部对齐
                window_coords[2] - 304,  # 宽度减去左侧栏
                min(800, window_coords[3])  # 高度限制或窗口高度
            )
            print(f"🔍 动态消息区域: {msg_area}")
        except Exception as e:
            print(f"❌ 动态检测失败: {e}，回退到静态坐标")
            # 回退到静态坐标
            msg_area = (
                WECHAT_WINDOW[0] + 304,
                WECHAT_WINDOW[1],
                1175,  # 修正宽度 (1479-304)
                800
            )
    else:
        # 使用静态坐标
        msg_area = (
            WECHAT_WINDOW[0] + 304,
            WECHAT_WINDOW[1],
            1175,  # 修正宽度
            800
        )
    
    os.makedirs(Constants.MESSAGES_DIR, exist_ok=True)

    screenshot = pyautogui.screenshot(region=msg_area)
    # 生成时间戳文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(
        Constants.MESSAGES_DIR,
        f"{Constants.MESSAGE_PREFIX}{timestamp}.png"
    )
    screenshot.save(screenshot_path)
    return screenshot_path

def fpreprocess_for_ocr(image):
    """OCR预处理管道"""
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    # 降噪处理
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    return denoised



def fdetect_wechat_theme(image):
    """
    检测微信主题模式（深色/浅色）
    Returns: 'dark' or 'light'
    """
    height, width = image.shape[:2]
    
    # 取样多个背景区域来判断主题
    sample_regions = [
        (int(width * 0.4), int(height * 0.2), 50, 50),  # 上方中央
        (int(width * 0.6), int(height * 0.5), 50, 50),  # 中间右侧
        (int(width * 0.5), int(height * 0.8), 50, 50),  # 下方中央
    ]
    
    avg_brightness = 0
    sample_count = 0
    
    for x, y, w, h in sample_regions:
        if x + w < width and y + h < height:
            region = image[y:y+h, x:x+w]
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            avg_brightness += np.mean(gray_region)
            sample_count += 1
    
    if sample_count > 0:
        avg_brightness /= sample_count
        
    # 阈值判断深色/浅色模式
    theme = 'dark' if avg_brightness < 100 else 'light'
    print(f"🎨 检测到主题: {theme} 模式 (亮度: {avg_brightness:.1f})")
    return theme

def fget_theme_colors(theme):
    """根据主题返回消息气泡颜色"""
    if theme == 'dark':
        # 深色模式颜色
        incoming_colors = [
            (45, 45, 45),    # 深灰色气泡
            (55, 55, 55),    # 稍亮的深灰
            (65, 65, 65),    # 另一种深灰变体
            (40, 40, 40),    # 更深的灰色
        ]
        outgoing_color = (76, 148, 83)  # 绿色气泡（深浅模式基本相同）
        
    else:  # light mode
        incoming_colors = [
            (255, 255, 255), # 白色气泡
            (245, 245, 245), # 浅灰气泡
            (250, 250, 250), # 偏白气泡
        ]
        outgoing_color = (169, 234, 122)  # 浅绿气泡
    
    return incoming_colors, outgoing_color

def fextract_messages_by_theme(image, theme='light', tolerance=30):
    """
    根据微信主题提取消息区域
    Returns: (incoming_regions, outgoing_regions)
    """
    incoming_colors, outgoing_color = get_theme_colors(theme)
    
    incoming_regions = []
    outgoing_regions = []
    
    # 查找接收消息区域（深色模式：深灰，浅色模式：白色）
    for target_color in incoming_colors:
        regions = find_color_regions(image, target_color, tolerance)
        incoming_regions.extend(regions)
    
    # 查找发送消息区域（绿色气泡）
    outgoing_regions = find_color_regions(image, outgoing_color, tolerance)
    
    return incoming_regions, outgoing_regions

def ffind_color_regions(image, target_color, tolerance=30):
    """
    在图像中查找特定颜色的区域
    Returns: list of (x, y, w, h) bounding boxes
    """
    target_color = np.array(target_color)
    
    # 创建颜色掩码
    lower = np.array([max(0, c - tolerance) for c in target_color])
    upper = np.array([min(255, c + tolerance) for c in target_color])
    
    mask = cv2.inRange(image, lower, upper)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 过滤小区域（噪声）
        if w > 50 and h > 20:
            regions.append((x, y, w, h))
    
    return regions

def fget_chat_messages(screenshot_path):
    """捕获并解析微信消息（支持深色/浅色模式）"""
    total_start = time.time()
    time_stats = {
        'total': 0,
        'image_load': 0,
        'theme_detect': 0,
        'region_detect': 0,
        'ocr_process': 0,
        'text_filter': 0
    }
    result = {'white': []}  # 保持原始格式兼容性
    
    try:
        # 图像加载耗时
        img_load_start = time.time()
        image = cv2.imread('./' + screenshot_path)
        if image is None:
            print(f"❌ 无法加载图像: {screenshot_path}")
            return result

        time_stats['image_load'] = time.time() - img_load_start

        # 主题检测耗时
        theme_start = time.time()
        theme = detect_wechat_theme(image)
        time_stats['theme_detect'] = time.time() - theme_start

        # 消息区域检测耗时
        region_start = time.time()
        incoming_regions, outgoing_regions = extract_messages_by_theme(image, theme)
        time_stats['region_detect'] = time.time() - region_start
        
        print(f"📱 找到 {len(incoming_regions)} 个接收消息区域, {len(outgoing_regions)} 个发送消息区域")

        # OCR处理耗时
        ocr_start = time.time()
        
        # 处理接收消息（这是我们主要关心的）
        clean_texts = []
        for i, (x, y, w, h) in enumerate(incoming_regions):
            try:
                # 裁剪消息区域
                message_region = image[y:y+h, x:x+w]
                
                # 预处理
                processed_region = preprocess_for_ocr(message_region)
                
                # OCR识别
                words_result = OCR_READER.readtext(processed_region)
                
                # 提取文本
                region_text = ''
                for detection in words_result:
                    text = detection[1].strip()
                    confidence = detection[2]
                    
                    if text and confidence > 0.5:  # 过滤低置信度
                        region_text += text
                        
                if region_text:
                    clean_texts.append(region_text)
                    print(f"📝 区域 {i+1}: '{region_text[:30]}{'...' if len(region_text) > 30 else ''}'")
            
            except Exception as e:
                print(f"❌ 区域 {i+1} OCR错误: {e}")
        
        # 取最新（最下方）的消息
        if clean_texts:
            # 假设最后一个区域是最新消息
            result['white'] = [clean_texts[-1]]
            print(f"✅ 提取到最新消息: '{clean_texts[-1][:50]}{'...' if len(clean_texts[-1]) > 50 else ''}'")
        
        time_stats['ocr_process'] = time.time() - ocr_start
        time_stats['total'] = time.time() - total_start

        # 打印耗时分析
        print("\n[增强性能分析]")
        print(f"总耗时: {time_stats['total']:.3f}s")
        print(f"图像加载: {time_stats['image_load'] * 1000:.1f}ms ({time_stats['image_load'] / time_stats['total']:.1%})")
        print(f"主题检测: {time_stats['theme_detect'] * 1000:.1f}ms")
        print(f"区域检测: {time_stats['region_detect'] * 1000:.1f}ms")
        print(f"OCR处理: {time_stats['ocr_process'] * 1000:.1f}ms ({time_stats['ocr_process'] / time_stats['total']:.1%})")

        return result

    except Exception as e:
        print(f"消息捕获失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return result


# ========== 主程序 ==========
if __name__ == "__main__":
    total_start = time.time()
    image_path = '../pic/screenshots/wechat_20250224_224615.png'
    image = cv2.imread(image_path)
    print(image.shape )
    result = get_chat_messages(image_path)
    # y = recognize_green_bottom(image_path)
    pprint(result)
    total= time.time() - total_start
    pprint(total)
