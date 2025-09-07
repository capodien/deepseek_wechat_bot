# 标准库导入
import os
import platform
import random
import threading
import time
from pprint import pprint

# 第三方库导入
import pyautogui
from pynput import keyboard

# 本地模块导入
from capture.deal_chatbox import fget_message_area_screenshot, fget_chat_messages
from capture.get_name_free import fget_friend_name
from capture.monitor_new_message import frecognize_message, frecognize_message_forwin
from capture.text_change_monitor import fdetect_new_message_by_text_change
from modules.m_screenshot_processor import fcapture_messages_screenshot
from db import db
from deepseek import deepseekai

# 配置常量
WORK_MODE = "chat"  # 模式切换：chat/forward
FORWARD_PREFIX = "[自动转发] "  # 转发模式前缀
SAFE_MODE = True  # 安全模式：只输入不发送消息

# 全局停止标志 - 使用 threading.Event 以便诊断面板控制
stop_bot = threading.Event()

def fon_key_press(key):
    """ESC键监听器"""
    global stop_bot
    try:
        if key == keyboard.Key.esc:
            print("\n🛑 检测到ESC键，正在停止机器人...")
            stop_bot.set()
            return False  # 停止监听器
    except AttributeError:
        pass

def fstart_key_listener():
    """启动键盘监听器"""
    listener = keyboard.Listener(on_press=fon_key_press)
    listener.daemon = True  # 设置为守护线程
    listener.start()
    return listener


def fload_config():
    """加载配置文件"""
    try:
        with open('config.cfg', 'r', encoding='utf-8') as f:
            config = {}
            for line in f:
                # 处理带注释的情况
                line = line.split('#')[0].strip()  # 去除注释
                if '=' in line:
                    # 使用split的maxsplit参数防止值含等号
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
            return config.get('mode', 'chat').lower()
    except FileNotFoundError:
        print("⚠️ 配置文件不存在，使用默认chat模式")
        return 'chat'
    except Exception as e:
        print(f"❗ 配置读取失败：{str(e)}，使用默认chat模式")
        return 'chat'


from PIL import Image
import io
import platform


def fcopy_image_to_clipboard(image_path):
    system = platform.system()
    img = Image.open(image_path)

    if system == "Darwin":  # macOS
        from AppKit import NSPasteboard, NSImage
        nsimage = NSImage.alloc().initWithContentsOfFile_(image_path)
        NSPasteboard.generalPasteboard().clearContents()
        NSPasteboard.generalPasteboard().writeObjects_([nsimage])

    elif system == "Windows":  # Windows
        import win32clipboard
        output = io.BytesIO()
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # 去除BMP头
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    elif system == "Linux":  # Linux
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk, Gdk
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_image(Gtk.Image.new_from_file(image_path).get_pixbuf())
        clipboard.store()


def fsend_image(image_path):
    """发送图片文件（适用于微信桌面端）"""
    try:
        import pyperclip
        # 保存原始剪贴板内容
        original = pyperclip.paste()

        # 激活附件按钮（坐标需根据实际界面调整）
        # pyautogui.click(x=130, y=680)  # 微信附件按钮坐标

        time.sleep(1)  # 等待文件选择框打开

        # 输入绝对路径（需要处理不同操作系统路径格式）
        if platform.system() == 'Windows':
            image_path = os.path.abspath(image_path).replace('/', '\\')
        else:
            image_path = os.path.abspath(image_path)

        # pyperclip.copy(image_path)
        fcopy_image_to_clipboard(image_path)
        time.sleep(0.5)

        # 粘贴路径并确认（Windows/Mac不同热键）
        if platform.system() == 'Darwin':
            pyautogui.hotkey('command', 'v')
            time.sleep(0.5)
            pyautogui.press('enter')
        else:
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.5)
            pyautogui.press('enter', presses=2)  # 需要两次回车

        time.sleep(1)
        # 恢复剪贴板内容
        pyautogui.copy(original)
    except Exception as e:
        print(f"图片发送失败: {str(e)}")


def fsend_reply(text, safe_mode=True):
    """发送消息（回车发送方案）- 支持安全模式"""
    try:
        import pyperclip

        if platform.system() == 'Darwin':
            print('masos')
            pyperclip.copy(text)
            # pyautogui.typewrite(text, interval=0.1)  # 模拟打字
            pyautogui.hotkey('command', 'a')
            pyautogui.hotkey('command', 'v')
        elif platform.system() == 'Windows':
            print('windows')
            pyperclip.copy(text)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.hotkey('ctrl', 'v')
        else:
            raise Exception("Unsupported OS")
        
        time.sleep(0.1)
        
        if not safe_mode:
            # 只有在非安全模式下才发送
            pyautogui.press('enter')
            print(f"✅ 消息已发送: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            # 安全模式：只输入不发送
            print(f"🔒 安全模式 - 消息已输入但未发送: {text[:50]}{'...' if len(text) > 50 else ''}")
            print("💡 请手动按回车键发送，或按ESC取消")
        
        time.sleep(0.3)

    except Exception as e:
        print(f"操作失败: {str(e)}")


def fcleanup_old_screenshots(max_files=100, max_days=7):
    """清理旧截图文件"""
    import glob
    from datetime import datetime, timedelta
    
    try:
        # 获取所有截图文件
        screenshot_dirs = ['pic/screenshots/', 'pic/message/']
        total_cleaned = 0
        
        for dir_path in screenshot_dirs:
            if not os.path.exists(dir_path):
                continue
                
            files = glob.glob(os.path.join(dir_path, '*.png'))
            files.sort(key=os.path.getmtime)  # 按修改时间排序
            
            # 删除超过数量限制的旧文件
            if len(files) > max_files:
                files_to_delete = files[:-max_files]  # 保留最新的max_files个
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        total_cleaned += 1
                    except Exception:
                        pass
            
            # 删除超过时间限制的文件
            cutoff_time = datetime.now() - timedelta(days=max_days)
            remaining_files = glob.glob(os.path.join(dir_path, '*.png'))
            
            for file_path in remaining_files:
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        total_cleaned += 1
                except Exception:
                    pass
        
        if total_cleaned > 0:
            print(f"🧹 清理了 {total_cleaned} 个旧截图文件")
        
        return total_cleaned
        
    except Exception as e:
        print(f"截图清理失败: {str(e)}")
        return 0

def fget_screenshot_stats():
    """获取截图统计信息"""
    import glob
    
    try:
        total_files = 0
        total_size = 0
        
        screenshot_dirs = ['pic/screenshots/', 'pic/message/']
        for dir_path in screenshot_dirs:
            if os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, '*.png'))
                total_files += len(files)
                for file_path in files:
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        pass
        
        size_mb = total_size / (1024 * 1024)
        return total_files, size_mb
        
    except Exception as e:
        print(f"获取截图统计失败: {str(e)}")
        return 0, 0

def fload_contacts():
    """加载监听名单"""
    try:
        with open('names.txt', 'r', encoding='utf-8') as f:
            return [name.strip() for name in f.readlines() if name.strip()]
    except Exception as e:
        print(f"加载联系人失败: {str(e)}")
        return []


if __name__ == "__main__":
    # 初始化配置
    WORK_MODE = fload_config()
    listen_list = fload_contacts()
    db.create_db()
    db.create_messagesdb()

    print(f"当前模式：{WORK_MODE.upper()} 模式")
    if SAFE_MODE:
        print("🔒 安全模式：已启用 - 只输入消息不自动发送")
        print("💡 机器人会在微信输入框中输入回复，您需要手动按回车发送")
    else:
        print("⚠️  自动发送模式：已启用 - 会自动发送消息")
    print("🔥 按ESC键可随时停止机器人")
    print("🌐 诊断面板: http://localhost:5001")
    
    # 清理旧截图并显示统计信息
    files_count, size_mb = fget_screenshot_stats()
    print(f"📷 当前截图文件: {files_count} 个，占用空间: {size_mb:.1f} MB")
    
    if files_count > 200:  # 如果文件太多就自动清理
        print("🧹 检测到截图文件过多，开始自动清理...")
        fcleanup_old_screenshots(max_files=100, max_days=3)
        files_count, size_mb = fget_screenshot_stats()
        print(f"📷 清理后: {files_count} 个文件，{size_mb:.1f} MB")
    
    # 诊断服务器已移除 - 使用独立的 step_diagnostic_server.py
    
    # 启动键盘监听器
    key_listener = fstart_key_listener()
    
    # 初始化状态
    for name in listen_list:
        deepseekai.add_user(name)
        print(f"已监听: {name}")
        
    mode_status = "安全模式" if SAFE_MODE else "自动发送模式"

    while not stop_bot.is_set():
        try:
            # Reset process steps at the beginning of each cycle
            
            # Step 1: Screenshot Capture
            screenshot_start = time.time()
            screenshot_path = fcapture_messages_screenshot()
            screenshot_duration = (time.time() - screenshot_start) * 1000
            
            # Update current screenshot for dashboard
            if screenshot_path:
                pass  # Screenshot captured successfully

            try:
                # Step 2: Message Detection
                detection_start = time.time()
                
                # Try new text change detection method first
                x, y = fdetect_new_message_by_text_change(screenshot_path)
                
                # Fallback to red dot detection if text change method fails
                if x is None or y is None:
                    print("[调试] 文本变化检测未发现新消息，尝试红点检测")
                    if platform.system() == 'Darwin':
                        x, y = frecognize_message(screenshot_path)
                    elif platform.system() == 'Windows':
                        x, y = frecognize_message_forwin(screenshot_path)
                else:
                    print(f"[调试] 通过文本变化检测到新消息！位置: ({x}, {y})")
                
                detection_duration = (time.time() - detection_start) * 1000
                
                if x is not None and y is not None:
                    print(f"检测到新消息，点击位置: ({x}, {y}) 路径：" + screenshot_path)
                    
                    pyautogui.moveTo(x, y, duration=random.uniform(0.2, 0.5))  # 随机移动速度
                    pyautogui.click(x, y)
                    
                    # Step 3: Name Recognition
                    name_start = time.time()
                    screenshot_path = fcapture_messages_screenshot()
                    if screenshot_path:
                        pass  # Screenshot captured for name extraction
                    name = fget_friend_name(x, y, screenshot_path)
                    name_duration = (time.time() - name_start) * 1000
                    

                    if name not in listen_list:
                        print(f"{name}不在监听列表，跳过处理")
                        pyautogui.moveTo(30, 50, duration=random.uniform(0.2, 0.5))  # 随机移动速度
                        pyautogui.click(30, 50)
                        continue
                    no_message_count = 0
                    while not stop_bot.is_set():
                        try:
                            # Step 4: OCR Processing
                            ocr_start = time.time()
                            screenshot_path = fget_message_area_screenshot()
                            if screenshot_path:
                                pass  # Screenshot captured for OCR
                            final_result = fget_chat_messages(screenshot_path)
                            ocr_duration = (time.time() - ocr_start) * 1000
                            pprint(final_result)
                            if final_result['white']:
                                latest_msg = final_result['white'][-1]
                                print(f'来自 {name} 的消息：{latest_msg}')
                                
                                start_time = time.time()
                                
                                # Step 5: AI Generation
                                ai_start = time.time()
                                
                                # 模式判断逻辑
                                if WORK_MODE == "chat":
                                    reply = deepseekai.reply(name, latest_msg, safe_mode=SAFE_MODE)
                                    ai_duration = (time.time() - ai_start) * 1000
                                    
                                    # Step 6: Message Input
                                    input_start = time.time()
                                    fsend_reply(reply, safe_mode=SAFE_MODE)
                                    input_duration = (time.time() - input_start) * 1000
                                elif WORK_MODE == "forward":
                                    reply = f"{FORWARD_PREFIX}{latest_msg}"
                                    ai_duration = (time.time() - ai_start) * 1000
                                    
                                    # Step 6: Message Input (Forward mode)
                                    input_start = time.time()
                                    pyautogui.click(118, 117)
                                    fsend_image(screenshot_path)
                                    fsend_reply(reply, safe_mode=SAFE_MODE)
                                    input_duration = (time.time() - input_start) * 1000
                                
                                processing_time = (time.time() - start_time) * 1000
                                
                                # 发送和存储逻辑
                                db.save_message(name, latest_msg, reply)
                                if SAFE_MODE:
                                    print(f"🔒 安全模式 - 已准备回复：{reply}")
                                else:
                                    print(f"已发送：{reply}")
                                
                                # Message processing completed
                                
                                no_message_count = 0
                            else:
                                no_message_count += 1
                                print(f"空消息计数：{no_message_count}/5")

                            if no_message_count >= 5:
                                print("连续5次空消息，退出对话")
                                pyautogui.click(30, 50)
                                break
                            time.sleep(5)
                        except Exception as e:
                            print(f"消息处理异常：{str(e)}")
                            import traceback
                            traceback.print_exc()
                            break
            except Exception as e:
                print(f"消息循环异常：{str(e)}")
                import traceback
                traceback.print_exc()
        except KeyboardInterrupt:
            print("\n程序已终止")
            break
        except Exception as e:
            print(f"运行时错误: {str(e)}")
            time.sleep(5)
    
    # 清理状态
    print("✅ 机器人已安全停止")
