#!/usr/bin/env python3
"""
Log Viewer utility for run_system logs
Provides easy access to view and filter system logs
"""

import os
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path

def list_log_files(log_dir="logs"):
    """List all available run_system log files"""
    log_pattern = os.path.join(log_dir, "run_system.log.*")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("No run_system log files found")
        return []
    
    # Sort by date (newest first)
    log_files.sort(reverse=True)
    
    print("Available run_system log files:")
    print("-" * 50)
    for i, log_file in enumerate(log_files, 1):
        filename = os.path.basename(log_file)
        date_str = filename.replace("run_system.log.", "")
        size = os.path.getsize(log_file) / 1024  # KB
        print(f"{i:2d}. {filename} ({size:.1f} KB)")
    
    return log_files

def view_log(log_file, lines=None, filter_text=None, follow=False):
    """View log file with optional filtering"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Viewing: {log_file}")
    print("=" * 80)
    
    try:
        if follow:
            # Follow mode (like tail -f)
            import subprocess
            cmd = ['tail', '-f', log_file]
            if filter_text:
                cmd = ['tail', '-f', log_file, '|', 'grep', filter_text]
            subprocess.run(cmd)
        else:
            with open(log_file, 'r', encoding='utf-8') as f:
                content_lines = f.readlines()
            
            # Apply filter if specified
            if filter_text:
                content_lines = [line for line in content_lines if filter_text.lower() in line.lower()]
            
            # Apply line limit if specified
            if lines:
                if lines > 0:
                    content_lines = content_lines[-lines:]  # Last N lines
                else:
                    content_lines = content_lines[:abs(lines)]  # First N lines
            
            # Display content
            for line in content_lines:
                print(line.rstrip())
    
    except Exception as e:
        print(f"Error reading log file: {e}")

def get_log_stats(log_dir="logs"):
    """Get statistics about log files"""
    log_pattern = os.path.join(log_dir, "run_system.log.*")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("No log files found")
        return
    
    total_size = 0
    oldest_date = None
    newest_date = None
    
    print("Log File Statistics:")
    print("-" * 50)
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        date_str = filename.replace("run_system.log.", "")
        size = os.path.getsize(log_file)
        total_size += size
        
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if oldest_date is None or file_date < oldest_date:
                oldest_date = file_date
            if newest_date is None or file_date > newest_date:
                newest_date = file_date
        except ValueError:
            continue
    
    print(f"Total files: {len(log_files)}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    if oldest_date and newest_date:
        print(f"Date range: {oldest_date.strftime('%Y-%m-%d')} to {newest_date.strftime('%Y-%m-%d')}")
        print(f"Retention: {(newest_date - oldest_date).days + 1} days")

def cleanup_old_logs(log_dir="logs", retention_days=5, dry_run=True):
    """Clean up old log files"""
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    log_pattern = os.path.join(log_dir, "run_system.log.*")
    log_files = glob.glob(log_pattern)
    
    files_to_delete = []
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        date_str = filename.replace("run_system.log.", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff_date:
                files_to_delete.append((log_file, file_date))
        except ValueError:
            continue
    
    if not files_to_delete:
        print(f"No log files older than {retention_days} days found")
        return
    
    print(f"Files older than {retention_days} days:")
    print("-" * 50)
    
    total_size = 0
    for log_file, file_date in files_to_delete:
        size = os.path.getsize(log_file) / 1024  # KB
        total_size += size
        print(f"  {os.path.basename(log_file)} ({size:.1f} KB) - {file_date.strftime('%Y-%m-%d')}")
    
    print(f"\nTotal: {len(files_to_delete)} files, {total_size:.1f} KB")
    
    if dry_run:
        print("\n[DRY RUN] Use --execute to actually delete these files")
    else:
        confirm = input("\nDelete these files? (y/N): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for log_file, _ in files_to_delete:
                try:
                    os.remove(log_file)
                    deleted_count += 1
                    print(f"Deleted: {os.path.basename(log_file)}")
                except Exception as e:
                    print(f"Error deleting {log_file}: {e}")
            print(f"\nDeleted {deleted_count} files")
        else:
            print("Cleanup cancelled")

def main():
    parser = argparse.ArgumentParser(description='Log Viewer for run_system logs')
    parser.add_argument('--log-dir', default='logs', help='Log directory (default: logs)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available log files')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View log file')
    view_parser.add_argument('file', nargs='?', help='Log file to view (or number from list)')
    view_parser.add_argument('--lines', '-n', type=int, help='Number of lines to show (negative for first N lines)')
    view_parser.add_argument('--filter', '-f', help='Filter lines containing text')
    view_parser.add_argument('--follow', action='store_true', help='Follow log file (like tail -f)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show log file statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old log files')
    cleanup_parser.add_argument('--days', type=int, default=5, help='Retention days (default: 5)')
    cleanup_parser.add_argument('--execute', action='store_true', help='Actually delete files (default is dry run)')
    
    # Today command (shortcut to view today's log)
    today_parser = subparsers.add_parser('today', help='View today\'s log')
    today_parser.add_argument('--lines', '-n', type=int, help='Number of lines to show')
    today_parser.add_argument('--filter', '-f', help='Filter lines containing text')
    today_parser.add_argument('--follow', action='store_true', help='Follow log file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_log_files(args.log_dir)
    
    elif args.command == 'view':
        log_files = list_log_files(args.log_dir)
        if not log_files:
            return
        
        if args.file:
            # Check if it's a number (index) or filename
            try:
                file_index = int(args.file) - 1
                if 0 <= file_index < len(log_files):
                    log_file = log_files[file_index]
                else:
                    print(f"Invalid file number: {args.file}")
                    return
            except ValueError:
                # It's a filename
                log_file = os.path.join(args.log_dir, args.file)
                if not args.file.startswith('run_system.log.'):
                    log_file = os.path.join(args.log_dir, f'run_system.log.{args.file}')
        else:
            # Default to most recent log
            log_file = log_files[0]
        
        view_log(log_file, args.lines, args.filter, args.follow)
    
    elif args.command == 'today':
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(args.log_dir, f'run_system.log.{today}')
        view_log(log_file, args.lines, args.filter, args.follow)
    
    elif args.command == 'stats':
        get_log_stats(args.log_dir)
    
    elif args.command == 'cleanup':
        cleanup_old_logs(args.log_dir, args.days, not args.execute)

if __name__ == "__main__":
    main() 