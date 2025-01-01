#!/usr/bin/env python3
"""
KHCC Paper Tracker for Databricks
A tool to track research papers from King Hussein Cancer Center using OpenAlex and PubMed.
"""

# Databricks imports
from databricks.sdk.runtime import *

# Standard library imports
import os
from datetime import datetime
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging

# Third-party imports
import pandas as pd
import numpy as np
from Bio import Entrez
from openai import AzureOpenAI
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Import helper functions for SQL operations
from aidi_functions import append_to_sql, write_to_sql

# Import the Paper class and DatabaseManager from openAlexSearch.py
from openAlexSearch import Paper, PaperSearcher

class DatabricksManager:
    """Manages Databricks-specific database operations"""
    
    def __init__(self, catalog_name: str = "AIDI-DB",
                 schema_name: str = "dbo",
                 table_name: str = "GOLD_Bibliometry",
                 impact_file: str = "/dbfs/FileStore/khcc_paper_tracker/impact_factor/CopyofImpactFactor2024.xlsx"):
        """Initialize database manager"""
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.table_name = table_name
        self.impact_file = impact_file
        self.impact_df = None
        
        # Field length limits matching SQL definition
        self.length_limits = {
            'title': 500,
            'authors': 1000,
            'abstract': 8000,
            'journal': 255,
            'doi': 255,
            'pmid': 50,
            'issn': 20,
            'source': 255,
            'affiliation': 500,
            'url': 1000,
            'keywords': 500,
            'paper_hash': 255,
            'journal_name': 255,
            'quartile': 50,
            'categories': 500
        }
        
        # Load impact factors
        self._load_impact_factors()

    def _load_impact_factors(self):
        """Load impact factor data from Excel file in DBFS"""
        try:
            self.impact_df = pd.read_excel(self.impact_file)
            print(f"Loaded impact factors for {len(self.impact_df)} journals")
        except Exception as e:
            print(f"Error loading impact factors: {str(e)}")
            self.impact_df = None

    def save_papers_batch(self, papers: List[Paper]):
        """Save batch of papers to Databricks SQL database"""
        try:
            from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType
            
            # Convert papers to DataFrame
            papers_data = [paper.to_dict() for paper in papers]
            df = pd.DataFrame(papers_data)
            
            # Generate paper_hash if not present
            if 'paper_hash' not in df.columns:
                df['paper_hash'] = [p.generate_hash() for p in papers]
            
            # Process impact factors if available
            if self.impact_df is not None:
                df = process_impact_factors(df, self.impact_df)
            
            # Ensure all string columns
            string_columns = [
                'title', 'authors', 'abstract', 'abstract_summary', 'journal',
                'doi', 'pmid', 'issn', 'source', 'affiliation', 'url',
                'keywords', 'paper_hash', 'journal_name', 'impact_factor',
                'quartile', 'categories'
            ]
            
            for column in string_columns:
                if column in df.columns:
                    df[column] = df[column].astype(str)
                    df[column] = df[column].replace('nan', None)
                    df[column] = df[column].replace('None', None)
                    if column in self.length_limits:
                        df[column] = df[column].apply(lambda x: self._truncate_string(x, self.length_limits[column]))
            
            # Handle dates
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            df.loc[df['publication_date'].isna(), 'publication_date'] = pd.Timestamp.now()
            
            # Handle numeric columns
            df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
            
            # Define schema
            schema = StructType([
                StructField("title", StringType(), True),
                StructField("authors", StringType(), True),
                StructField("abstract", StringType(), True),
                StructField("abstract_summary", StringType(), True),
                StructField("publication_date", DateType(), True),
                StructField("journal", StringType(), True),
                StructField("doi", StringType(), True),
                StructField("pmid", StringType(), True),
                StructField("issn", StringType(), True),
                StructField("citations", IntegerType(), True),
                StructField("source", StringType(), True),
                StructField("affiliation", StringType(), True),
                StructField("url", StringType(), True),
                StructField("keywords", StringType(), True),
                StructField("paper_hash", StringType(), True),
                StructField("journal_name", StringType(), True),
                StructField("impact_factor", StringType(), True),
                StructField("quartile", StringType(), True),
                StructField("categories", StringType(), True)
            ])
            
            # Convert to Spark DataFrame
            df_spark = spark.createDataFrame(df, schema=schema)
            
            # Write to SQL database
            write_to_sql(
                catalog_name=self.catalog_name,
                table_name=self.table_name,
                df=df_spark,
                schema_name=self.schema_name,
                jdbc_url="jdbc:sqlserver://aidi-db-server.database.windows.net:1433;databaseName=AIDI-DB"
            )
            
            print(f"Successfully processed {len(papers)} papers")
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            raise

    def _truncate_string(self, s: str, max_length: int = 500) -> str:
        """Truncate string to specified length if necessary"""
        if not s:
            return s
        return str(s)[:max_length] if len(str(s)) > max_length else str(s)

class DatabricksPaperTracker:
    """Main class for tracking papers in Databricks environment"""
    
    def __init__(self):
        """Initialize tracker with API keys from Databricks secrets"""
        email = dbutils.secrets.get(scope="key-vault-secrrt", key="EMAIL")
        self.searcher = PaperSearcher(email=email)
        self.db = DatabricksManager()
        
        # Setup logging to DBFS
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to DBFS"""
        log_dir = "/dbfs/FileStore/khcc_paper_tracker/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, "paper_tracker.log")
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def run(self, start_date: str, end_date: str):
        """Run the paper tracking process"""
        try:
            # Search both sources
            papers = self.searcher.search_papers(start_date, end_date)
            
            if papers:
                print(f"\nSaving {len(papers)} papers to database...")
                self.db.save_papers_batch(papers)
                print("\nProcess completed successfully!")
            else:
                print("\nNo papers found in the date range.")
                
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            logging.error(f"Error during execution: {str(e)}")
            raise

def main(start_date: str, end_date: str):
    """Main entry point for Databricks notebook"""
    try:
        # Validate dates
        start = datetime.strptime(start_date, '%Y/%m/%d')
        end = datetime.strptime(end_date, '%Y/%m/%d')
        
        if end < start:
            raise ValueError("End date must be after start date")
        
        if end > datetime.now():
            raise ValueError("End date cannot be in the future")
        
        # Initialize and run tracker
        tracker = DatabricksPaperTracker()
        tracker.run(start_date, end_date)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # This section won't be used in Databricks, but kept for completeness
    import argparse
    parser = argparse.ArgumentParser(description='Track KHCC research papers')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    args = parser.parse_args()
    
    main(args.start_date, args.end_date) 

    #