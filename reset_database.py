#!/usr/bin/env python3
"""
Simple script to reset the Sentosa database files
Useful when parquet files become corrupted
"""

import os
import sys
from database import ParquetDatabase

def main():
    """Reset all database files to empty state"""
    print("🔄 Resetting Sentosa database files...")
    
    try:
        db = ParquetDatabase()
        db.reset_database_files()
        print("✅ Database reset completed successfully!")
        print("📁 The following files have been reset:")
        print("   - users.parquet")
        print("   - queries.parquet") 
        print("   - answers.parquet")
        print("\n💡 You can now restart the API server and run tests.")
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 