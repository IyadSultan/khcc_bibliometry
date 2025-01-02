#!/usr/bin/env python3
"""
File uses pubmed and serpapi to search for papers from KHCC (pubmed) and scholar (serpapi)
KHCC Paper Tracker
A tool to track research papers from King Hussein Cancer Center using PubMed and Google Scholar.
"""

import os
from datetime import datetime
import sqlite3
import pandas as pd
from Bio import Entrez
from impact_factor.core import Factor
import openai
import requests
from typing import Dict, List, Optional
import json
import hashlib
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

@dataclass
class Paper:
    """Data class representing a research paper"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    journal: str
    doi: Optional[str]
    pmid: Optional[str]
    issn: Optional[str]
    citations: Optional[int]
    source: str  # 'pubmed' or 'scholar'
    affiliation: str
    url: Optional[str]
    keywords: Optional[List[str]]

    def generate_hash(self) -> str:
        """Generate a unique hash for this paper"""
        content = f"{self.title}{self.publication_date}{self.journal}"
        return hashlib.md5(content.encode()).hexdigest()
def process_impact_factors(papers_df: pd.DataFrame, impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link papers with impact factor data using ISSN matching and extract quartile info
    
    Args:
        papers_df: DataFrame with papers data including ISSN
        impact_df: DataFrame with impact factor data
    """
    # Clean and prepare ISSN data
    impact_df['ISSN'] = impact_df['ISSN'].astype(str).str.strip()
    impact_df['EISSN'] = impact_df['EISSN'].astype(str).str.strip()
    
    # Create a mapping of ISSN/EISSN to impact factor data
    issn_mapping = {}
    
    for _, row in impact_df.iterrows():
        # Process each category string
        categories = row['Category'].split(';') if pd.notna(row['Category']) else []
        quartiles = []
        
        for cat in categories:
            if '|' in cat:
                cat_parts = cat.split('|')
                if len(cat_parts) >= 2:
                    quartiles.append(cat_parts[1])  # Get Q1, Q2, etc.
        
        # Take the highest quartile (Q1 is highest)
        best_quartile = min(quartiles) if quartiles else None
        
        impact_data = {
            'journal_name': row['Name'],
            'journal_abbr': row['Abbr Name'],
            'impact_factor': row['JIF'],
            'impact_factor_5y': row['JIF5Years'],
            'quartile': best_quartile,
            'categories': row['Category']
        }
        
        # Map both ISSN and EISSN to the same data
        if pd.notna(row['ISSN']) and row['ISSN'] != 'nan':
            issn_mapping[row['ISSN']] = impact_data
        if pd.notna(row['EISSN']) and row['EISSN'] != 'nan':
            issn_mapping[row['EISSN']] = impact_data
    
    # Function to get impact data for a paper
    def get_impact_data(issn):
        if pd.isna(issn) or issn == 'nan':
            return pd.Series({
                'journal_name_matched': None,
                'journal_abbr_matched': None,
                'impact_factor_matched': None,
                'impact_factor_5y_matched': None,
                'quartile_matched': None,
                'categories_matched': None
            })
        
        issn = str(issn).strip()
        impact_data = issn_mapping.get(issn)
        
        if impact_data:
            return pd.Series({
                'journal_name_matched': impact_data['journal_name'],
                'journal_abbr_matched': impact_data['journal_abbr'],
                'impact_factor_matched': impact_data['impact_factor'],
                'impact_factor_5y_matched': impact_data['impact_factor_5y'],
                'quartile_matched': impact_data['quartile'],
                'categories_matched': impact_data['categories']
            })
        else:
            return pd.Series({
                'journal_name_matched': None,
                'journal_abbr_matched': None,
                'impact_factor_matched': None,
                'impact_factor_5y_matched': None,
                'quartile_matched': None,
                'categories_matched': None
            })
    
    # Add impact factor data to papers DataFrame
    impact_columns = papers_df['issn'].apply(get_impact_data)
    result_df = pd.concat([papers_df, impact_columns], axis=1)
    
    # Add a flag for whether impact factor was found
    result_df['has_impact_data'] = result_df['impact_factor_matched'].notna()
    
    return result_df

class DatabaseManager:
    """Manages SQLite database operations for paper storage"""
    
    def __init__(self, db_path: str = "khcc_papers.db", 
                 impact_file: str = "impact_factor/CopyofImpactFactor2024.xlsx"):
        """
        Initialize database manager
        Args:
            db_path: Path to SQLite database
            impact_file: Path to impact factor Excel file
        """
        self.db_path = db_path
        self.impact_file = impact_file
        self.impact_df = None
        self.issn_mapping = {}
        
        # Setup database and load impact factors
        self.setup_database()
        self.load_impact_factors()

    def setup_database(self):
        """Create or update the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # First create the base table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    abstract TEXT,
                    publication_date TEXT,
                    journal TEXT,
                    doi TEXT,
                    pmid TEXT,
                    issn TEXT,
                    citations INTEGER,
                    pubmed_citations INTEGER,
                    scholar_citations INTEGER,
                    last_citation_update TIMESTAMP,
                    source TEXT,
                    affiliation TEXT,
                    url TEXT,
                    keywords TEXT,
                    impact_factor REAL,
                    verified_khcc BOOLEAN,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    paper_hash TEXT UNIQUE
                )
            """)
            
            # Check if impact_factor_5y column exists
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(papers)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add new columns if they don't exist
            if 'impact_factor_5y' not in columns:
                print("Adding impact_factor_5y column...")
                conn.execute("ALTER TABLE papers ADD COLUMN impact_factor_5y REAL")
            
            if 'quartile' not in columns:
                print("Adding quartile column...")
                conn.execute("ALTER TABLE papers ADD COLUMN quartile TEXT")
                
            if 'categories' not in columns:
                print("Adding categories column...")
                conn.execute("ALTER TABLE papers ADD COLUMN categories TEXT")
                
            if 'journal_matched' not in columns:
                print("Adding journal_matched column...")
                conn.execute("ALTER TABLE papers ADD COLUMN journal_matched TEXT")
                
            # Create indices for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pmid ON papers(pmid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_issn ON papers(issn)")
            
            print("Database schema updated successfully")

    def load_impact_factors(self):
        """Load and process impact factor data"""
        try:
            self.impact_df = pd.read_excel(self.impact_file)
            
            # Clean ISSN data
            self.impact_df['ISSN'] = self.impact_df['ISSN'].astype(str).str.strip()
            self.impact_df['EISSN'] = self.impact_df['EISSN'].astype(str).str.strip()
            
            # Create ISSN to impact factor mapping
            for _, row in self.impact_df.iterrows():
                # Process categories and get best quartile
                categories = row['Category'].split(';') if pd.notna(row['Category']) else []
                quartiles = []
                
                for cat in categories:
                    if '|' in cat:
                        cat_parts = cat.split('|')
                        if len(cat_parts) >= 2:
                            quartiles.append(cat_parts[1])
                
                # Take the highest quartile
                best_quartile = min(quartiles) if quartiles else None
                
                impact_data = {
                    'journal_name': row['Name'],
                    'journal_abbr': row['Abbr Name'],
                    'impact_factor': row['JIF'],
                    'impact_factor_5y': row['JIF5Years'],
                    'quartile': best_quartile,
                    'categories': row['Category']
                }
                
                # Map both ISSN and EISSN
                if pd.notna(row['ISSN']) and row['ISSN'] != 'nan':
                    self.issn_mapping[row['ISSN']] = impact_data
                if pd.notna(row['EISSN']) and row['EISSN'] != 'nan':
                    self.issn_mapping[row['EISSN']] = impact_data
            
            print(f"Loaded impact factors for {len(self.issn_mapping)} unique journals")
            
        except Exception as e:
            print(f"Error loading impact factors: {str(e)}")
            self.impact_df = None
            self.issn_mapping = {}

    def get_impact_data(self, issn: str) -> Dict:
        """Get impact factor data for a given ISSN"""
        if not issn or pd.isna(issn):
            return None
        
        issn = str(issn).strip()
        return self.issn_mapping.get(issn)

    def save_paper(self, paper: Paper, impact_data: Dict, verified: bool):
        """Save or update a paper with impact factor data"""
        paper_hash = paper.generate_hash()
        
        # Get impact factor data
        if paper.issn:
            journal_impact = self.get_impact_data(paper.issn)
        else:
            journal_impact = None
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if paper exists
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM papers WHERE paper_hash = ?", (paper_hash,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing paper
                cursor.execute("""
                    UPDATE papers SET
                        title = ?, authors = ?, abstract = ?, publication_date = ?,
                        journal = ?, doi = ?, pmid = ?, issn = ?, citations = ?,
                        source = ?, affiliation = ?, url = ?, keywords = ?,
                        impact_factor = ?, impact_factor_5y = ?, quartile = ?,
                        categories = ?, journal_matched = ?, verified_khcc = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE paper_hash = ?
                """, (
                    paper.title,
                    json.dumps(paper.authors),
                    paper.abstract,
                    paper.publication_date,
                    paper.journal,
                    paper.doi,
                    paper.pmid,
                    paper.issn,
                    paper.citations,
                    paper.source,
                    paper.affiliation,
                    paper.url,
                    json.dumps(paper.keywords) if paper.keywords else None,
                    journal_impact['impact_factor'] if journal_impact else None,
                    journal_impact['impact_factor_5y'] if journal_impact else None,
                    journal_impact['quartile'] if journal_impact else None,
                    journal_impact['categories'] if journal_impact else None,
                    journal_impact['journal_name'] if journal_impact else None,
                    verified,
                    paper_hash
                ))
                print(f"Updated paper: {paper.title}")
                if journal_impact:
                    print(f"Impact Factor: {journal_impact['impact_factor']}, "
                          f"Quartile: {journal_impact['quartile']}")
            else:
                # Insert new paper
                cursor.execute("""
                    INSERT INTO papers (
                        title, authors, abstract, publication_date, journal,
                        doi, pmid, issn, citations, source, affiliation,
                        url, keywords, impact_factor, impact_factor_5y, 
                        quartile, categories, journal_matched, verified_khcc,
                        paper_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.title,
                    json.dumps(paper.authors),
                    paper.abstract,
                    paper.publication_date,
                    paper.journal,
                    paper.doi,
                    paper.pmid,
                    paper.issn,
                    paper.citations,
                    paper.source,
                    paper.affiliation,
                    paper.url,
                    json.dumps(paper.keywords) if paper.keywords else None,
                    journal_impact['impact_factor'] if journal_impact else None,
                    journal_impact['impact_factor_5y'] if journal_impact else None,
                    journal_impact['quartile'] if journal_impact else None,
                    journal_impact['categories'] if journal_impact else None,
                    journal_impact['journal_name'] if journal_impact else None,
                    verified,
                    paper_hash
                ))
                print(f"Inserted new paper: {paper.title}")
                if journal_impact:
                    print(f"Impact Factor: {journal_impact['impact_factor']}, "
                          f"Quartile: {journal_impact['quartile']}")

    def get_unmatched_journals(self) -> pd.DataFrame:
        """Get list of papers without impact factor matches"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("""
                SELECT title, journal, issn
                FROM papers
                WHERE impact_factor IS NULL
                ORDER BY journal
            """, conn)

    def get_impact_summary(self) -> pd.DataFrame:
        """Get summary of impact factors by quartile"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("""
                SELECT 
                    quartile,
                    COUNT(*) as paper_count,
                    ROUND(AVG(impact_factor), 2) as avg_impact_factor,
                    ROUND(MIN(impact_factor), 2) as min_impact_factor,
                    ROUND(MAX(impact_factor), 2) as max_impact_factor
                FROM papers
                WHERE impact_factor IS NOT NULL
                GROUP BY quartile
                ORDER BY quartile
            """, conn)

class PaperSearcher:
    """Handles searching for papers across different sources"""

    def __init__(self, email: str, openai_api_key: str, serpapi_key: str):
        """Initialize searcher with required API keys"""
        self.email = email
        Entrez.email = email
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.serpapi_key = serpapi_key
        self.factor = Factor()
        self.db = DatabaseManager()

    def search_pubmed(self, start_date: str, end_date: str) -> List[Paper]:
        """Search PubMed for papers within date range"""
        print("\nSearching PubMed...")
        query = f'("King Hussein Cancer Center"[Affiliation]) AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=1000, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        total_papers = len(record["IdList"])
        print(f"Found {total_papers} papers in PubMed")
        
        papers = []
        with tqdm(total=total_papers, desc="Fetching PubMed papers") as pbar:
            for pmid in record["IdList"]:
                try:
                    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="xml")
                    record = Entrez.read(handle)['PubmedArticle'][0]
                    handle.close()
                    
                    paper = self._parse_pubmed_record(record)
                    if paper:
                        papers.append(paper)
                    
                    pbar.update(1)
                    time.sleep(0.5)  # Respect NCBI rate limits
                    
                except Exception as e:
                    print(f"Error processing PMID {pmid}: {str(e)}")
                    continue
        
        return papers

    def search_google_scholar(self, start_date: str, end_date: str) -> List[Paper]:
        """Search Google Scholar using SerpAPI"""
        print("\nSearching Google Scholar...")
        papers = []
        
        try:
            # Verify API key first
            if not self.test_serpapi_key():
                print("Invalid SerpAPI key. Please check your API key at https://serpapi.com")
                return papers
                
            # Convert dates to year for Google Scholar search
            start_year = datetime.strptime(start_date, '%Y/%m/%d').year
            end_year = datetime.strptime(end_date, '%Y/%m/%d').year
            
            # Parameters for SerpAPI
            params = {
                "engine": "google_scholar",
                "q": '"King Hussein Cancer Center"',
                "api_key": self.serpapi_key,
                "as_ylo": start_year,
                "as_yhi": end_year,
                "num": 20,
                "start": 0
            }
            
            print(f"Searching with params: {params}")
            
            max_pages = 5  # Limit pages to avoid excessive API calls
            
            with tqdm(desc="Fetching Google Scholar papers") as pbar:
                for page in range(max_pages):
                    try:
                        response = requests.get("https://serpapi.com/search", params=params)
                        print(f"\nAPI Response Status: {response.status_code}")
                        
                        if response.status_code != 200:
                            print(f"Error response: {response.text}")
                            break
                        
                        data = response.json()
                        
                        if "error" in data:
                            print(f"API Error: {data['error']}")
                            break
                            
                        if "organic_results" not in data:
                            print(f"No results found. Response keys: {data.keys()}")
                            break
                            
                        results = data["organic_results"]
                        if not results:
                            break
                        
                        for result in results:
                            paper = self._parse_serpapi_result(result)
                            if paper:
                                papers.append(paper)
                                pbar.update(1)
                        
                        params["start"] = (page + 1) * 20
                        time.sleep(2)  # Respect API rate limits
                        
                    except Exception as e:
                        print(f"\nError fetching page {page}: {str(e)}")
                        break
                        
        except Exception as e:
            print(f"\nError in Google Scholar search: {str(e)}")
        
        print(f"Found {len(papers)} papers in Google Scholar")
        return papers

    def test_serpapi_key(self) -> bool:
        """Test if the SerpAPI key is valid"""
        try:
            response = requests.get(
                "https://serpapi.com/account", 
                params={"api_key": self.serpapi_key}
            )
            return response.status_code == 200
        except:
            return False

    def verify_khcc_affiliation(self, paper: Paper) -> bool:
        """Verify KHCC affiliation using OpenAI"""
        prompt = f"""
        Please analyze if this paper is affiliated with King Hussein Cancer Center (KHCC):
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Affiliation: {paper.affiliation}
        Abstract: {paper.abstract}
        
        Return only 'true' if it's affiliated with KHCC, or 'false' if not.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip().lower() == 'true'

    def get_impact_factor(self, paper: Paper) -> Dict:
        """Get journal impact factor"""
        search_term = paper.issn if paper.issn else paper.journal
        result = self.factor.search(search_term)
        
        if result:
            return {
                'jcr': result[0].get('jcr'),
                'factor': result[0].get('factor')
            }
        return {'jcr': None, 'factor': None}

    def _parse_pubmed_record(self, record) -> Optional[Paper]:
        """Parse PubMed record into Paper object"""
        try:
            article = record['MedlineCitation']['Article']
            
            # Extract authors
            authors = []
            for author in article.get('AuthorList', []):
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['LastName']} {author['ForeName']}")
                elif 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])
            
            # Extract date
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')
            publication_date = f"{year}/{month}/{day}"
            
            # Extract affiliations
            affiliations = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'AffiliationInfo' in author:
                        for affiliation in author['AffiliationInfo']:
                            if 'Affiliation' in affiliation:
                                affiliations.append(affiliation['Affiliation'])
            
            # Extract DOI
            doi = None
            for id_obj in article.get('ELocationID', []):
                if id_obj.attributes.get('EIdType') == 'doi':
                    doi = str(id_obj)
                    break
            
            return Paper(
                title=article.get('ArticleTitle', ''),
                authors=authors,
                abstract=article.get('Abstract', {}).get('AbstractText', [''])[0],
                publication_date=publication_date,
                journal=article.get('Journal', {}).get('Title', ''),
                doi=doi,
                pmid=record['MedlineCitation'].get('PMID', ''),
                issn=article.get('Journal', {}).get('ISSN', ''),
                citations=None,
                source='pubmed',
                affiliation='; '.join(affiliations),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{record['MedlineCitation'].get('PMID', '')}/",
                keywords=article.get('KeywordList', [[]])[0] if 'KeywordList' in article else None
            )
            
        except Exception as e:
            print(f"Error parsing PubMed record: {str(e)}")
            return None

    def _parse_serpapi_result(self, result: Dict) -> Optional[Paper]:
        """Parse SerpAPI result into Paper object"""
        try:
            # Extract publication info
            pub_info = result.get("publication_info", {}).get("summary", "")
            
            # Extract authors
            authors = []
            if "..." in pub_info:
                authors = pub_info.split("-")[0].strip().split(", ")
            else:
                authors = [pub_info.split("-")[0].strip()]
            
            # Extract year
            year = ""
            if "year" in result.get("publication_info", {}):
                year = result["publication_info"]["year"]
            elif ", " in pub_info:
                year = pub_info.split(", ")[-1].split("-")[0].strip()
            
            # Extract journal
            journal = ""
            if "journal" in result.get("publication_info", {}):
                journal = result["publication_info"]["journal"]
            elif " - " in pub_info:
                journal = pub_info.split(" - ")[-1].strip()
            
            # Extract citations
            citations = None
            if "inline_links" in result and "cited_by" in result["inline_links"]:
                citations = result["inline_links"]["cited_by"].get("total")
            
            return Paper(
                title=result.get("title", ""),
                authors=authors,
                abstract=result.get("snippet", ""),
                publication_date=f"{year}/01/01" if year else None,
                journal=journal,
                doi=None,
                pmid=None,
                issn=None,
                citations=citations,
                source='scholar',
                affiliation="",
                url=result.get("link"),
                keywords=None
            )
            
        except Exception as e:
            print(f"Error parsing Scholar result: {str(e)}")
            return None

class KHCCPaperTracker:
    """Main class for tracking KHCC research papers"""

    def __init__(self, email: str, openai_api_key: str, serpapi_key: str):
        """Initialize tracker with required API keys"""
        self.searcher = PaperSearcher(email, openai_api_key, serpapi_key)

    def run(self, start_date: str, end_date: str):
        """Run the complete paper tracking process"""
        # 1. Search both sources
        pubmed_papers = self.searcher.search_pubmed(start_date, end_date)
        scholar_papers = self.searcher.search_google_scholar(start_date, end_date)
        
        # 2. Merge and deduplicate
        all_papers = self._merge_and_deduplicate(pubmed_papers + scholar_papers)
        
        if not all_papers:
            print("No papers found to process.")
            return
            
        # 3. Process each paper
        with ThreadPoolExecutor() as executor:
            # Verify KHCC affiliation
            print("\nVerifying KHCC affiliations...")
            verification_futures = {
                executor.submit(self.searcher.verify_khcc_affiliation, paper): paper 
                for paper in all_papers
            }
            
            verified_papers = []
            for future in tqdm(verification_futures, desc="Verifying papers"):
                try:
                    paper = verification_futures[future]
                    if future.result():
                        verified_papers.append(paper)
                except Exception as e:
                    print(f"Error verifying paper '{paper.title}': {str(e)}")
                    continue
            
            # Get impact factors
            print(f"\nFetching impact factors for {len(verified_papers)} papers...")
            for paper in tqdm(verified_papers, desc="Processing papers"):
                try:
                    impact_data = executor.submit(self.searcher.get_impact_factor, paper).result()
                    self.searcher.db.save_paper(paper, impact_data, True)
                except Exception as e:
                    print(f"Error processing paper '{paper.title}': {str(e)}")
                    continue
        
        print(f"\nSuccessfully processed {len(verified_papers)} verified KHCC papers")
    
    def _merge_and_deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """Merge and deduplicate papers from multiple sources"""
        unique_papers = {}
        
        for paper in papers:
            if not paper:
                continue
            
            paper_hash = paper.generate_hash()
            
            if paper_hash in unique_papers:
                existing = unique_papers[paper_hash]
                # Prefer PubMed source over Scholar
                if paper.source == 'pubmed' and existing.source == 'scholar':
                    unique_papers[paper_hash] = paper
                # If both are from same source, keep the one with more information
                elif paper.source == existing.source:
                    if len(str(paper.abstract or '')) > len(str(existing.abstract or '')):
                        unique_papers[paper_hash] = paper
            else:
                unique_papers[paper_hash] = paper
        
        return list(unique_papers.values())

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Track KHCC research papers')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    parser.add_argument('--output', default='khcc_papers.db', help='Output database file')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    args = parser.parse_args()
    
    try:
        # Load environment variables
        if os.path.exists(args.env_file):
            load_dotenv(args.env_file)
        else:
            print(f"Warning: Environment file {args.env_file} not found")
        
        # Get credentials from environment variables
        email = os.getenv('EMAIL')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        serpapi_key = os.getenv('SERPAPI_KEY')
        
        if not all([email, openai_api_key, serpapi_key]):
            raise ValueError(
                "Missing required environment variables. Please set EMAIL, "
                "OPENAI_API_KEY, and SERPAPI_KEY in your .env file"
            )
        
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
                print("Error: Dates must be in YYYY/MM/DD format")
            else:
                print(f"Error: {str(e)}")
            return
        
        # Initialize tracker
        tracker = KHCCPaperTracker(
            email=email,
            openai_api_key=openai_api_key,
            serpapi_key=serpapi_key
        )
        
        # Run the search
        print(f"\nSearching for papers from {args.start_date} to {args.end_date}")
        tracker.run(args.start_date, args.end_date)
        print(f"\nResults saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    main()