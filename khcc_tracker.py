#!/usr/bin/env python3
"""
KHCC Paper Tracker: A tool to track and analyze research papers from KHCC using OpenAlex and PubMed
"""
import os
import sqlite3
from datetime import datetime
import json
import re
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import traceback
import argparse
import sys

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from Bio import Entrez

def reconstruct_abstract(inverted_index: Dict) -> Optional[str]:
    """Reconstruct abstract text from OpenAlex inverted index format"""
    if not inverted_index:
        return None
    
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort()
    return ' '.join(word for _, word in word_positions)

class JournalMetrics:
    """Handle journal impact factors and quartiles"""
    def __init__(self, excel_file: Optional[str] = None):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        if excel_file:
            self.load_metrics(excel_file)

    def load_metrics(self, file_path: str):
        """Load metrics from Excel file"""
        try:
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                issn = str(row['ISSN']).strip()
                if issn and issn != 'nan':
                    # Handle special case for impact factors
                    impact_factor = row['JIF']
                    if isinstance(impact_factor, str) and impact_factor.startswith('<'):
                        impact_factor = float(impact_factor.replace('<', ''))

                    self.metrics[issn] = {
                        'impact_factor': impact_factor if pd.notna(impact_factor) else None,
                        'quartile': row['Category'].split('|')[1] if '|' in str(row['Category']) else None
                    }
        except Exception as e:
            self.logger.error(f"Error loading journal metrics: {e}")

    def get_metrics(self, issn_list: List[str]) -> Tuple[Optional[float], Optional[str]]:
        """Get impact factor and quartile for a journal"""
        if not issn_list:
            return None, None

        # Try to match second ISSN first (print ISSN), then first
        for issn in issn_list:
            issn = self._normalize_issn(issn)
            metrics = self.metrics.get(issn)
            if metrics:
                return metrics['impact_factor'], metrics['quartile']
        
        return None, None

    @staticmethod
    def _normalize_issn(issn: str) -> str:
        """Normalize ISSN format"""
        issn = issn.strip().replace('X', 'x')
        if '-' not in issn and len(issn) == 8:
            issn = f"{issn[:4]}-{issn[4:]}"
        return issn

class Database:
    """Handle SQLite database operations"""
    def __init__(self, db_path: str = "khcc_papers.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        self.cursor.executescript('''
            DROP TABLE IF EXISTS papers;
            
            CREATE TABLE papers (
                -- Core identifiers
                paper_id TEXT PRIMARY KEY,
                doi TEXT,
                pmid TEXT,
                openalex_id TEXT,
                
                -- Basic metadata
                title TEXT,
                publication_date TEXT,
                publication_year INTEGER,
                
                -- Journal information
                journal_name TEXT,
                journal_issn TEXT,
                impact_factor REAL,
                quartile TEXT,
                
                -- Authors and institutions
                authors TEXT,
                authorships TEXT,
                institutions TEXT,
                
                -- Content
                abstract TEXT,
                abstract_summary TEXT,
                keywords TEXT,
                
                -- Metrics
                citations INTEGER,
                referenced_works_count INTEGER,
                
                -- Classification
                type TEXT,
                concepts TEXT,
                mesh TEXT,
                topics TEXT,
                
                -- Access information
                is_open_access BOOLEAN,
                pdf_url TEXT,
                
                -- Bibliographic
                volume TEXT,
                issue TEXT,
                pages TEXT,
                
                -- Additional metadata
                is_retracted BOOLEAN,
                language TEXT,
                source TEXT,
                raw_data TEXT,
                
                -- Timestamps and tracking
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi);
            CREATE INDEX IF NOT EXISTS idx_pmid ON papers(pmid);
            CREATE INDEX IF NOT EXISTS idx_date ON papers(publication_date);
        ''')
        self.conn.commit()
        self.logger.info(f"Initialized database at {self.db_path}")

    def save_paper(self, paper: Dict, journal_metrics: Optional[JournalMetrics] = None):
        """Save paper to database"""
        try:
            # Extract journal ISSNs and get metrics
            issns = self._extract_issns(paper)
            impact_factor = None
            quartile = None
            if journal_metrics and issns:
                impact_factor, quartile = journal_metrics.get_metrics(issns)

            # Prepare data
            data = self._prepare_paper_data(paper, impact_factor, quartile)
            
            # Insert or update
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = list(data.values())

            self.cursor.execute(f'''
                INSERT OR REPLACE INTO papers ({columns})
                VALUES ({placeholders})
            ''', values)
            
            self.conn.commit()

        except Exception as e:
            self.logger.error(f"Error saving paper {paper.get('id')}: {e}")
            traceback.print_exc()

    def _extract_issns(self, paper: Dict) -> List[str]:
        """Extract ISSNs from paper data"""
        try:
            source = paper.get('primary_location', {}).get('source', {})
            return source.get('issn', [])
        except Exception:
            return []

    def _prepare_paper_data(self, paper: Dict, impact_factor: Optional[float], quartile: Optional[str]) -> Dict:
        # Get nested objects with safe fallbacks
        location = paper.get('primary_location', {}) or {}
        source = location.get('source', {}) or {}
        biblio = paper.get('biblio', {}) or {}
        
        # Handle ISSN list safely
        issns = source.get('issn', [])
        issn_str = ', '.join(issns) if isinstance(issns, list) else str(issns)
            
        return {
            'paper_id': paper.get('id'),
            'doi': paper.get('doi'),
            'pmid': paper.get('ids', {}).get('pmid', '').replace('https://pubmed.ncbi.nlm.nih.gov/', '').replace('/', ''),
            'openalex_id': paper.get('id'),
            'title': paper.get('title'),
            'publication_date': paper.get('publication_date'),
            'publication_year': paper.get('publication_year'),
            'journal_name': source.get('display_name'),
            'journal_issn': issn_str,
            'impact_factor': impact_factor,
            'quartile': quartile,
            'authors': json.dumps([a.get('author', {}).get('display_name') for a in paper.get('authorships', [])]),
            'authorships': json.dumps(paper.get('authorships', [])),
            'institutions': json.dumps([i.get('display_name') for a in paper.get('authorships', []) 
                                     for i in a.get('institutions', [])]),
            'abstract': paper.get('abstract_text'),
            'abstract_summary': paper.get('abstract_summary'),
            'keywords': json.dumps([k.get('display_name') for k in paper.get('keywords', [])]),
            'citations': paper.get('cited_by_count'),
            'referenced_works_count': paper.get('referenced_works_count'),
            'type': paper.get('type'),
            'concepts': json.dumps(paper.get('concepts', [])),
            'mesh': json.dumps(paper.get('mesh', [])),
            'topics': json.dumps(paper.get('topics', [])),
            'is_open_access': paper.get('open_access', {}).get('is_oa'),
            'pdf_url': location.get('pdf_url'),
            'volume': biblio.get('volume'),
            'issue': biblio.get('issue'),
            'pages': f"{biblio.get('first_page')}-{biblio.get('last_page')}" if biblio.get('first_page') else None,
            'is_retracted': paper.get('is_retracted'),
            'language': paper.get('language'),
            'source': 'openalex',
            'raw_data': json.dumps(paper)
        }

    def close(self):
        """Close database connection"""
        self.conn.close()

class PaperTracker:
    """Main class for tracking and processing KHCC papers"""
    def __init__(self, email: str, openai_api_key: Optional[str] = None, 
                 impact_factor_file: Optional[str] = None, db_path: str = "khcc_papers.sqlite"):
        self.email = email
        Entrez.email = email
        self.logger = self._setup_logger()
        
        # Initialize components
        self.db = Database(db_path)
        self.journal_metrics = JournalMetrics(impact_factor_file)
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # KHCC identification
        self.khcc_id = 'I2799468983'  # OpenAlex ID for KHCC
        self.khcc_regex = re.compile(r'king\s+hussein\s+cancer\s+(center|centre|foundation)|khcc', re.IGNORECASE)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/khcc_tracker.log")
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def search_date_range(self, start_date: str, end_date: str):
        """Search for papers within date range"""
        self.logger.info(f"Starting search from {start_date} to {end_date}")
        
        # Search OpenAlex first
        openalex_papers = self.search_openalex(start_date, end_date)
        self.logger.info(f"Found {len(openalex_papers)} papers in OpenAlex")
        
        # Get existing PMIDs
        existing_pmids = {
            p.get('ids', {}).get('pmid', '').replace('https://pubmed.ncbi.nlm.nih.gov/', '').replace('/', '')
            for p in openalex_papers if p.get('ids', {}).get('pmid')
        }
        
        # Find additional papers from PubMed
        pubmed_pmids = set(self.search_pubmed(start_date, end_date))
        new_pmids = pubmed_pmids - existing_pmids
        self.logger.info(f"Found {len(new_pmids)} additional papers in PubMed")
        
        # Retrieve missing papers
        additional_papers = []
        for pmid in tqdm(new_pmids, desc="Retrieving additional papers"):
            paper = self.get_paper_by_pmid(pmid)
            if paper and self.verify_khcc_affiliation(paper):
                additional_papers.append(paper)
        
        # Combine all papers
        all_papers = openalex_papers + additional_papers
        self.logger.info(f"Total papers found: {len(all_papers)}")
        
        # Generate summaries if OpenAI is available
        if self.openai_client:
            self.generate_summaries(all_papers)
        
        # Save to database
        for paper in tqdm(all_papers, desc="Saving papers"):
            self.db.save_paper(paper, self.journal_metrics)

    def search_openalex(self, start_date: str, end_date: str) -> List[Dict]:
        """Search OpenAlex for KHCC papers"""
        papers = []
        
        # Format dates
        start = datetime.strptime(start_date, '%Y/%m/%d').strftime('%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y/%m/%d').strftime('%Y-%m-%d')
        
        params = {
            'filter': f'institutions.id:{self.khcc_id},from_publication_date:{start},to_publication_date:{end}',
            'per_page': 200,
            'cursor': '*'
        }
        
        with tqdm(desc="Fetching OpenAlex papers") as pbar:
            while True:
                try:
                    response = requests.get(
                        "https://api.openalex.org/works",
                        params=params,
                        headers={'User-Agent': f'KHCC Paper Tracker ({self.email})'}
                    )
                    
                    if response.status_code != 200:
                        self.logger.error(f"OpenAlex API error: {response.text}")
                        break
                    
                    data = response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        break
                    
                    for result in results:
                        if result:
                            # Reconstruct abstract if needed
                            if 'abstract_inverted_index' in result:
                                result['abstract_text'] = reconstruct_abstract(result['abstract_inverted_index'])
                            
                            papers.append(result)
                            pbar.update(1)
                    
                    # Get next page
                    next_cursor = data.get('meta', {}).get('next_cursor')
                    if not next_cursor:
                        break
                        
                    params['cursor'] = next_cursor
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error fetching OpenAlex results: {str(e)}")
                    break
        
        return papers

    def search_pubmed(self, start_date: str, end_date: str) -> List[str]:
        try:
            khcc_query = '("KHCC"[Affiliation] OR "King Hussein Cancer Center"[Affiliation] OR ' \
                        '"King Hussein Cancer Centre"[Affiliation] OR "King Hussein Cancer Foundation"[Affiliation])'
            date_query = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            query = f"{khcc_query} AND {date_query}"
            
            handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)
            record = Entrez.read(handle)
            handle.close()
            
            return record["IdList"]
            
        except Exception as e:
            self.logger.error(f"Error in PubMed search: {str(e)}")
            return []

    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Retrieve paper from OpenAlex using PMID"""
        try:
            url = f"https://api.openalex.org/works/pmid:{pmid}"
            response = requests.get(
                url,
                headers={'User-Agent': f'KHCC Paper Tracker ({self.email})'}
            )
            
            if response.status_code == 200:
                paper = response.json()
                
                # Reconstruct abstract if needed
                if 'abstract_inverted_index' in paper:
                    paper['abstract_text'] = reconstruct_abstract(paper['abstract_inverted_index'])
                
                return paper
            else:
                self.logger.error(f"Error retrieving paper {pmid} from OpenAlex: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving paper {pmid} from OpenAlex: {str(e)}")
            return None

    def verify_khcc_affiliation(self, paper: Dict) -> bool:
        """Verify KHCC affiliation in paper"""
        # Check OpenAlex institution ID
        for authorship in paper.get('authorships', []):
            for institution in authorship.get('institutions', []):
                if institution.get('id') == self.khcc_id:
                    return True
        
        # Check affiliation text
        for authorship in paper.get('authorships', []):
            for institution in authorship.get('institutions', []):
                if self.khcc_regex.search(institution.get('display_name', '')):
                    return True
        
        return False

    def generate_summaries(self, papers: List[Dict]):
        """Generate paper summaries using GPT-4"""
        self.logger.info("Generating abstract summaries...")
        
        for paper in tqdm(papers, desc="Generating summaries"):
            if not paper.get('abstract_text') or paper.get('abstract_summary'):
                continue
                
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Create a one-sentence summary capturing the main research objective and key findings of this abstract."},
                        {"role": "user", "content": paper['abstract_text']}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                paper['abstract_summary'] = response.choices[0].message.content.strip()
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error generating summary for paper {paper.get('title', '')}: {str(e)}")
                continue

def setup_logging():
    """Setup root logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/khcc_tracker.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='KHCC Paper Tracker')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    parser.add_argument('--impact-factor-file', default='impact_factor/CopyofimpactFactor2024.xlsx', 
                       help='Path to impact factor Excel file')
    parser.add_argument('--db-path', default='khcc_papers.sqlite',
                       help='Path to SQLite database file')
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load environment variables
        if os.path.exists(args.env_file):
            load_dotenv(args.env_file)
            logger.info(f"Loaded environment from {args.env_file}")
        else:
            logger.warning(f"Environment file {args.env_file} not found")
        
        # Get required environment variables
        email = os.getenv('EMAIL')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not email:
            raise ValueError("Missing required EMAIL environment variable")
        
        # Validate dates
        try:
            start_date = datetime.strptime(args.start_date, '%Y/%m/%d')
            end_date = datetime.strptime(args.end_date, '%Y/%m/%d')
            
            if end_date < start_date:
                raise ValueError("End date must be after start date")
            
            if end_date > datetime.now():
                raise ValueError("End date cannot be in the future")
                
        except ValueError as e:
            if "time data" in str(e):
                logger.error("Dates must be in YYYY/MM/DD format")
            else:
                logger.error(f"Date validation error: {str(e)}")
            return 1
        
        # Validate impact factor file
        if not os.path.exists(args.impact_factor_file):
            logger.warning(f"Impact factor file {args.impact_factor_file} not found")
            proceed = input("Continue without impact factors? (y/n): ")
            if proceed.lower() != 'y':
                return 1
        
        # Initialize tracker
        tracker = PaperTracker(
            email=email,
            openai_api_key=openai_api_key,
            impact_factor_file=args.impact_factor_file,
            db_path=args.db_path
        )
        
        logger.info(f"Starting paper tracking from {args.start_date} to {args.end_date}")
        logger.info(f"Using impact factor file: {args.impact_factor_file}")
        logger.info(f"Using database: {args.db_path}")
        
        # Run the search
        tracker.search_date_range(args.start_date, args.end_date)
        
        logger.info("Process completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
