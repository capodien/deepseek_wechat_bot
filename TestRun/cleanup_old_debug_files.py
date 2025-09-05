#!/usr/bin/env python3
"""
Cleanup utility for managing debug screenshot files

This script helps manage the pic/screenshots/ directory by removing old debug files
while preserving important screenshots.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_debug_files(screenshots_dir: str = "pic/screenshots", 
                           keep_days: int = 7, 
                           dry_run: bool = True):
    """
    Clean up old debug files while preserving important screenshots
    
    Args:
        screenshots_dir: Directory containing screenshots
        keep_days: Number of days to keep files (default: 7)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    
    if not os.path.exists(screenshots_dir):
        print(f"❌ Directory not found: {screenshots_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    print(f"🧹 Cleanup Analysis for {screenshots_dir}")
    print(f"   Keep files newer than: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'ACTUAL DELETION'}")
    print("=" * 60)
    
    # Categories for cleanup
    debug_patterns = [
        r".*horizontal_differences.*\.png$",
        r".*EnhancedDualBoundary.*\.png$", 
        r".*SimpleWidth.*\.png$",
        r".*advanced_avatar_detection.*\.png$",
        r".*Enhanced_Card_Avatar_Boundaries.*\.png$",
        r".*ContactName_Boundaries.*\.png$",
        r".*Complete_Analysis.*\.png$",
        r".*Debug_TimeDetection.*\.png$"
    ]
    
    preserved_patterns = [
        r".*_WeChat\.png$"  # Keep original WeChat screenshots
    ]
    
    files_to_delete = []
    files_to_preserve = []
    total_size_to_delete = 0
    total_files_scanned = 0
    
    for filename in os.listdir(screenshots_dir):
        if not filename.endswith('.png'):
            continue
            
        filepath = os.path.join(screenshots_dir, filename)
        file_stat = os.stat(filepath)
        file_date = datetime.fromtimestamp(file_stat.st_mtime)
        file_size = file_stat.st_size
        total_files_scanned += 1
        
        # Check if file should be preserved
        is_preserved = any(re.match(pattern, filename) for pattern in preserved_patterns)
        if is_preserved:
            files_to_preserve.append((filename, file_date, file_size))
            continue
            
        # Check if file is a debug file
        is_debug = any(re.match(pattern, filename) for pattern in debug_patterns)
        
        # Delete old debug files or very old files
        should_delete = False
        reason = ""
        
        if is_debug and file_date < cutoff_date:
            should_delete = True
            reason = f"Old debug file ({file_date.strftime('%Y-%m-%d')})"
        elif not is_debug and file_date < cutoff_date - timedelta(days=30):
            should_delete = True  
            reason = f"Very old file ({file_date.strftime('%Y-%m-%d')})"
        
        if should_delete:
            files_to_delete.append((filename, file_date, file_size, reason))
            total_size_to_delete += file_size
        else:
            files_to_preserve.append((filename, file_date, file_size))
    
    # Summary
    print(f"\n📊 CLEANUP SUMMARY")
    print(f"   Total files scanned: {total_files_scanned}")
    print(f"   Files to preserve: {len(files_to_preserve)}")
    print(f"   Files to delete: {len(files_to_delete)}")
    print(f"   Space to free: {total_size_to_delete / (1024*1024):.1f} MB")
    
    if files_to_delete:
        print(f"\n🗑️  FILES TO DELETE:")
        for filename, file_date, file_size, reason in sorted(files_to_delete):
            size_mb = file_size / (1024*1024)
            print(f"   📄 {filename}")
            print(f"      Date: {file_date.strftime('%Y-%m-%d %H:%M')}, Size: {size_mb:.1f}MB, Reason: {reason}")
        
        if not dry_run:
            print(f"\n⚠️  PERFORMING ACTUAL DELETION...")
            deleted_count = 0
            for filename, _, _, _ in files_to_delete:
                try:
                    filepath = os.path.join(screenshots_dir, filename)
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {filename}: {e}")
            
            print(f"✅ Deleted {deleted_count}/{len(files_to_delete)} files")
            print(f"💾 Freed {total_size_to_delete / (1024*1024):.1f} MB of space")
        else:
            print(f"\n💡 This was a DRY RUN. Run with dry_run=False to actually delete files.")
    
    if files_to_preserve:
        print(f"\n📁 PRESERVED FILES ({len(files_to_preserve)}):")
        preserved_size = sum(size for _, _, size in files_to_preserve)
        print(f"   Total preserved size: {preserved_size / (1024*1024):.1f} MB")
        
        # Show newest preserved files
        newest_preserved = sorted(files_to_preserve, key=lambda x: x[1], reverse=True)[:5]
        for filename, file_date, file_size in newest_preserved:
            size_mb = file_size / (1024*1024)
            print(f"   📄 {filename} ({file_date.strftime('%Y-%m-%d')}, {size_mb:.1f}MB)")
    
    print(f"\n✅ Cleanup analysis complete!")

def interactive_cleanup():
    """Interactive cleanup with user prompts"""
    
    print("🧹 INTERACTIVE DEBUG FILE CLEANUP")
    print("=" * 40)
    
    screenshots_dir = "pic/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        return
    
    # First, show current status
    print("📊 Current directory status:")
    total_files = len([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])
    total_size = sum(os.path.getsize(os.path.join(screenshots_dir, f)) 
                    for f in os.listdir(screenshots_dir) if f.endswith('.png'))
    print(f"   Total PNG files: {total_files}")
    print(f"   Total size: {total_size / (1024*1024):.1f} MB")
    
    # Ask user preferences
    print(f"\nCleanup Options:")
    print(f"1. Conservative (keep 7 days)")
    print(f"2. Moderate (keep 3 days)")
    print(f"3. Aggressive (keep 1 day)")
    print(f"4. Custom days")
    print(f"5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    keep_days_map = {'1': 7, '2': 3, '3': 1}
    
    if choice in keep_days_map:
        keep_days = keep_days_map[choice]
    elif choice == '4':
        try:
            keep_days = int(input("Enter days to keep: ").strip())
            if keep_days < 0:
                print("❌ Invalid number of days")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    elif choice == '5':
        print("👋 Cleanup cancelled")
        return
    else:
        print("❌ Invalid choice")
        return
    
    # Run dry run first
    print(f"\n🔍 Running analysis...")
    cleanup_old_debug_files(screenshots_dir, keep_days, dry_run=True)
    
    # Ask for confirmation
    confirm = input(f"\nProceed with actual deletion? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        cleanup_old_debug_files(screenshots_dir, keep_days, dry_run=False)
    else:
        print("👋 Cleanup cancelled")

if __name__ == "__main__":
    interactive_cleanup()