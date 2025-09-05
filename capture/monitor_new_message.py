

import os
import cv2
import numpy as np
from datetime import datetime
import pyautogui

from Constants import Constants
from .window_manager import get_window_manager


# Screenshot capture function moved to modules/m_ScreenShot_WeChatWindow.py
# This file now only contains message recognition functions

def recognize_message_forwin(image_path):
    """定位微信新消息红点坐标，返回(x, y)元组"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[预警] 图像加载失败: {image_path}")
        return (None, None)

    # 微信红点特征参数（BGR色彩空间）
    TARGET_COLOR = np.array([81, 81, 255])  # 精确匹配微信消息提示红点
    COLOR_TOLERANCE = 10  # 颜色容差范围
    X_RANGE = (66, 380)  # 有效区域水平坐标范围

    # 生成坐标网格（性能优化关键！）
    x_coords, y_coords = np.meshgrid(
        np.arange(image.shape[1]),
        np.arange(image.shape[0])
    )

    # 构建三维色彩矩阵（比逐像素遍历快100倍）
    lower_bound = TARGET_COLOR - COLOR_TOLERANCE
    upper_bound = TARGET_COLOR + COLOR_TOLERANCE
    color_mask = np.all((lower_bound <= image) & (image <= upper_bound), axis=-1)

    # 区域智能过滤（排除头像区域和侧边栏干扰）
    region_mask = (x_coords >= X_RANGE[0]) & (x_coords <= X_RANGE[1])

    # 获取所有候选坐标（已自动过滤无效区域）
    matched_points = np.column_stack((
        x_coords[color_mask & region_mask],
        y_coords[color_mask & region_mask]
    ))

    # 智能选择策略：优先取最下方的红点（最新消息）
    if matched_points.size > 0:
        # 按垂直坐标降序排序
        sorted_points = matched_points[np.argsort(-matched_points[:, 1])]
        # 返回首个有效坐标（精确到像素级）
        return tuple(sorted_points[0].astype(int))

    print("[调试] 未检测到有效消息提示")
    return (None, None)


def recognize_message(image_path):
    """定位微信新消息红点坐标，返回(x, y)元组"""
    # 闪电加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[预警] 图像加载失败: {image_path}")
        return (None, None)

    # 微信红点特征参数（BGR色彩空间）
    TARGET_COLOR = np.array([84, 98, 227])  # 精确匹配微信消息提示红点 - 更新为实际检测到的颜色
    COLOR_TOLERANCE = 15  # 添加颜色容差以适应不同显示设备
    X_RANGE = (60, 320)  # 有效区域水平坐标范围

    # 生成坐标网格（性能优化关键！）
    x_coords, y_coords = np.meshgrid(
        np.arange(image.shape[1]),
        np.arange(image.shape[0])
    )

    # 构建三维色彩矩阵（比逐像素遍历快100倍）
    lower_bound = TARGET_COLOR - COLOR_TOLERANCE
    upper_bound = TARGET_COLOR + COLOR_TOLERANCE
    color_mask = np.all((lower_bound <= image) & (image <= upper_bound), axis=-1)
    # 区域智能过滤（排除头像区域和侧边栏干扰）
    region_mask = (x_coords >= X_RANGE[0]) & (x_coords <= X_RANGE[1])

    # 获取所有候选坐标（已自动过滤无效区域）
    matched_points = np.column_stack((
        x_coords[color_mask & region_mask],
        y_coords[color_mask & region_mask]
    ))

    # 智能选择策略：优先取最下方的红点（最新消息）
    if matched_points.size > 0:
        # 按垂直坐标降序排序
        sorted_points = matched_points[np.argsort(-matched_points[:, 1])]
        # 返回首个有效坐标（精确到像素级）
        return tuple(sorted_points[0].astype(int))

    print("[调试] 未检测到有效消息提示")
    return (None, None)






