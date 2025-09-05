#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screenshot Cleanup Utility
Standalone script to clean up old screenshot files
"""

import os
import glob
import sys
from datetime import datetime, timedelta

def cleanup_screenshots(max_files=100, max_days=7, dry_run=False):
    """清理截图文件"""
    screenshot_dirs = ['pic/screenshots/', 'pic/message/']
    total_cleaned = 0
    total_size_cleaned = 0
    
    print(f"📷 Screenshot Cleanup Utility")
    print(f"🎯 Keeping max {max_files} files per directory")
    print(f"🕐 Removing files older than {max_days} days")
    print(f"{'🔍 DRY RUN MODE - No files will be deleted' if dry_run else '⚠️  LIVE MODE - Files will be permanently deleted'}")
    print("-" * 50)
    
    for dir_path in screenshot_dirs:
        if not os.path.exists(dir_path):
            print(f"⏭️  Skipping {dir_path} (directory not found)")
            continue
            
        print(f"\n📁 Processing {dir_path}...")
        
        files = glob.glob(os.path.join(dir_path, '*.png'))
        files.sort(key=os.path.getmtime)  # 按修改时间排序
        
        print(f"   Found {len(files)} PNG files")
        
        # 删除超过数量限制的旧文件
        if len(files) > max_files:
            files_to_delete = files[:-max_files]  # 保留最新的max_files个
            print(f"   🗂️  Removing {len(files_to_delete)} files (count limit exceeded)")
            
            for file_path in files_to_delete:
                try:
                    file_size = os.path.getsize(file_path)
                    if not dry_run:
                        os.remove(file_path)
                    total_cleaned += 1
                    total_size_cleaned += file_size
                    if dry_run:
                        print(f"      [DRY] Would delete: {os.path.basename(file_path)} ({file_size/1024:.1f} KB)")
                    else:
                        print(f"      ✅ Deleted: {os.path.basename(file_path)} ({file_size/1024:.1f} KB)")
                except Exception as e:
                    print(f"      ❌ Failed to delete {os.path.basename(file_path)}: {e}")
        
        # 删除超过时间限制的文件
        cutoff_time = datetime.now() - timedelta(days=max_days)
        remaining_files = glob.glob(os.path.join(dir_path, '*.png'))
        old_files = []
        
        for file_path in remaining_files:
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    old_files.append(file_path)
            except Exception:
                pass
        
        if old_files:
            print(f"   📅 Removing {len(old_files)} files (older than {max_days} days)")
            
            for file_path in old_files:
                try:
                    file_size = os.path.getsize(file_path)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if not dry_run:
                        os.remove(file_path)
                    total_cleaned += 1
                    total_size_cleaned += file_size
                    if dry_run:
                        print(f"      [DRY] Would delete: {os.path.basename(file_path)} (from {file_time.strftime('%Y-%m-%d')})")
                    else:
                        print(f"      ✅ Deleted: {os.path.basename(file_path)} (from {file_time.strftime('%Y-%m-%d')})")
                except Exception as e:
                    print(f"      ❌ Failed to delete {os.path.basename(file_path)}: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"🏁 Cleanup Summary:")
    print(f"   Files processed: {total_cleaned}")
    print(f"   Space freed: {total_size_cleaned/1024/1024:.1f} MB")
    
    return total_cleaned, total_size_cleaned

def get_stats():
    """获取截图统计信息"""
    screenshot_dirs = ['pic/screenshots/', 'pic/message/']
    total_files = 0
    total_size = 0
    
    for dir_path in screenshot_dirs:
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, '*.png'))
            dir_size = 0
            for file_path in files:
                try:
                    file_size = os.path.getsize(file_path)
                    dir_size += file_size
                except Exception:
                    pass
            
            print(f"📁 {dir_path}: {len(files)} files, {dir_size/1024/1024:.1f} MB")
            total_files += len(files)
            total_size += dir_size
    
    print(f"📊 Total: {total_files} files, {total_size/1024/1024:.1f} MB")
    return total_files, total_size

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'stats':
            print("📷 Screenshot Statistics:")
            get_stats()
            sys.exit(0)
        elif sys.argv[1] == 'help':
            print("""
Screenshot Cleanup Utility

Usage:
    python3 cleanup_screenshots.py [command] [options]

Commands:
    stats                    - Show current screenshot statistics
    cleanup [--dry-run]      - Clean up old screenshots
    aggressive [--dry-run]   - Aggressive cleanup (keep only 50 files, 3 days)
    help                     - Show this help

Examples:
    python3 cleanup_screenshots.py stats
    python3 cleanup_screenshots.py cleanup --dry-run
    python3 cleanup_screenshots.py aggressive
            """)
            sys.exit(0)
        elif sys.argv[1] == 'aggressive':
            dry_run = '--dry-run' in sys.argv
            cleanup_screenshots(max_files=50, max_days=3, dry_run=dry_run)
        elif sys.argv[1] == 'cleanup':
            dry_run = '--dry-run' in sys.argv
            cleanup_screenshots(max_files=100, max_days=7, dry_run=dry_run)
    else:
        # Default: show stats and ask
        print("📷 Current screenshot statistics:")
        files, size = get_stats()
        
        if files > 500:
            print(f"\n⚠️  You have {files} screenshot files taking up {size/1024/1024:.1f} MB")
            print("💡 Consider running cleanup:")
            print("   python3 cleanup_screenshots.py cleanup --dry-run  (preview)")
            print("   python3 cleanup_screenshots.py cleanup            (execute)")
        else:
            print(f"\n✅ Screenshot storage is reasonable ({files} files, {size/1024/1024:.1f} MB)")