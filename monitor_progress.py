#!/usr/bin/env python3
"""
Data Collection Progress Monitor
Real-time monitoring of the comprehensive data collection process
"""

import json
import time
from pathlib import Path
from datetime import datetime
import os

def monitor_progress():
    """Monitor the progress of data collection"""
    output_dir = Path("historical_data")
    progress_file = output_dir / "progress.json"
    
    print("📊 DATA COLLECTION PROGRESS MONITOR")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring\n")
    
    last_completed = 0
    start_time = datetime.now()
    
    try:
        while True:
            # Check if progress file exists
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    completed = sum(1 for v in progress.values() if v.get('status') == 'success')
                    failed = sum(1 for v in progress.values() if v.get('status') == 'failed')
                    total_processed = completed + failed
                    
                    # Calculate statistics
                    if total_processed > 0:
                        success_rate = (completed / total_processed) * 100
                    else:
                        success_rate = 0
                    
                    # Calculate speed
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 0:
                        rate = total_processed / elapsed * 60  # per minute
                    else:
                        rate = 0
                    
                    # Calculate ETA (assuming 304 total tickers)
                    total_tickers = 304
                    remaining = total_tickers - total_processed
                    if rate > 0:
                        eta_minutes = remaining / (rate / 60)
                        eta_hours = eta_minutes / 60
                    else:
                        eta_minutes = 0
                        eta_hours = 0
                    
                    # Clear screen and display progress
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("📊 DATA COLLECTION PROGRESS MONITOR")
                    print("=" * 50)
                    print(f"🕐 Started: {start_time.strftime('%H:%M:%S')}")
                    print(f"⏱️ Elapsed: {elapsed/60:.1f} minutes")
                    print(f"📈 Progress: {total_processed}/304 tickers ({total_processed/304*100:.1f}%)")
                    print(f"✅ Successful: {completed}")
                    print(f"❌ Failed: {failed}")
                    print(f"🎯 Success Rate: {success_rate:.1f}%")
                    print(f"⚡ Rate: {rate:.1f} tickers/minute")
                    print(f"🕒 ETA: {eta_hours:.1f} hours ({eta_minutes:.1f} minutes)")
                    
                    # Show progress bar
                    progress_percent = total_processed / 304 * 100
                    bar_length = 40
                    filled_length = int(bar_length * progress_percent / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\n[{bar}] {progress_percent:.1f}%")
                    
                    # Show recent activity
                    if completed > last_completed:
                        recent_completed = [k for k, v in progress.items() 
                                          if v.get('status') == 'success'][-5:]
                        print(f"\n🔄 Recent completions: {', '.join(recent_completed)}")
                    
                    last_completed = completed
                    
                except json.JSONDecodeError:
                    print("⚠️ Progress file is being updated...")
                except Exception as e:
                    print(f"❌ Error reading progress: {e}")
                    
            else:
                print("⏳ Waiting for progress file to be created...")
            
            # Check for completion
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    if len(progress) >= 304:
                        print("\n🎉 DATA COLLECTION COMPLETED!")
                        break
                except:
                    pass
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring stopped by user")

def show_current_status():
    """Show a quick status snapshot"""
    output_dir = Path("historical_data")
    progress_file = output_dir / "progress.json"
    
    if not progress_file.exists():
        print("❌ No progress file found. Data collection may not have started.")
        return
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        completed = sum(1 for v in progress.values() if v.get('status') == 'success')
        failed = sum(1 for v in progress.values() if v.get('status') == 'failed')
        total_processed = completed + failed
        
        print(f"📊 Current Status:")
        print(f"  Processed: {total_processed}/304 ({total_processed/304*100:.1f}%)")
        print(f"  Successful: {completed}")
        print(f"  Failed: {failed}")
        
        if failed > 0:
            failed_tickers = [k for k, v in progress.items() if v.get('status') == 'failed']
            print(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
            if len(failed_tickers) > 10:
                print(f"    ... and {len(failed_tickers) - 10} more")
                
    except Exception as e:
        print(f"❌ Error reading status: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_current_status()
    else:
        monitor_progress()
