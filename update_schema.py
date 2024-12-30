#!/usr/bin/env python3
"""
Script to update database schema with missing columns
"""

import sqlite3
import sys
from datetime import datetime

def update_schema(db_path="khcc_papers.db"):
    """Add missing columns and set default values"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(papers)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Get current timestamp as string
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add missing columns
            if 'last_updated' not in columns:
                print("Adding last_updated column...")
                conn.execute("ALTER TABLE papers ADD COLUMN last_updated TIMESTAMP")
                conn.execute(f"UPDATE papers SET last_updated = '{current_time}'")
            
            if 'date_added' not in columns:
                print("Adding date_added column...")
                conn.execute("ALTER TABLE papers ADD COLUMN date_added TIMESTAMP")
                conn.execute(f"UPDATE papers SET date_added = '{current_time}'")
            
            # Add other columns without timestamp defaults
            simple_columns = {
                'jif': 'REAL',
                'jif5years': 'REAL',
                'quartile': 'TEXT',
                'categories': 'TEXT',
                'journal_matched': 'TEXT',
                'paper_hash': 'TEXT',
                'pubmed_citations': 'INTEGER',
                'scholar_citations': 'INTEGER',
                'last_citation_update': 'TIMESTAMP'
            }
            
            for column, dtype in simple_columns.items():
                if column not in columns:
                    print(f"Adding column {column}...")
                    conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {dtype}")
            
            # Create indices
            print("Creating indices...")
            indices = {
                'idx_doi': 'doi',
                'idx_pmid': 'pmid',
                'idx_issn': 'issn',
                'idx_paper_hash': 'paper_hash'
            }
            
            for idx_name, column in indices.items():
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON papers({column})")
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        raise
            
            print("Database schema updated successfully")
            
    except Exception as e:
        print(f"Error updating schema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update KHCC papers database schema')
    parser.add_argument('--db', default='khcc_papers.db', help='Database file path')
    args = parser.parse_args()
    
    update_schema(args.db)