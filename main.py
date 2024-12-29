import os
from datetime import datetime
import sqlite3
import pandas as pd
from Bio import Entrez
from scholarly import scholarly
from impact_factor.core import Factor
import openai
from typing import Dict, List, Optional
import json
import hashlib
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

@dataclass
class Paper:
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
    
class DatabaseManager:
    def __init__(self, db_path: str = "khcc_papers.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
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
                    source TEXT,
                    affiliation TEXT,
                    url TEXT,
                    keywords TEXT,
                    impact_factor REAL,
                    jcr TEXT,
                    verified_khcc BOOLEAN,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    paper_hash TEXT UNIQUE
                )
            """)

class PaperSearcher:
    def __init__(self, email: str, api_key: str):
        Entrez.email = email
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.factor = Factor()
        self.db = DatabaseManager()
        
    def search_pubmed(self, start_date: str, end_date: str) -> List[Paper]:
        print("Searching PubMed...")
        query = f'("King Hussein Cancer Center"[Affiliation]) AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        handle = Entrez.esearch(db="pubmed", term=query, retmax=1000, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        total_papers = len(record["IdList"])
        print(f"Found {total_papers} papers in PubMed")
        
        papers = []
        for pmid in tqdm(record["IdList"], desc="Fetching PubMed papers"):
            try:
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="xml")
                record = Entrez.read(handle)['PubmedArticle'][0]
                handle.close()
                
                # Extract paper details
                paper = self._parse_pubmed_record(record)
                if paper:  # Only append if parsing was successful
                    papers.append(paper)
                time.sleep(0.5)  # Respect NCBI rate limits
                
            except Exception as e:
                print(f"Error processing PMID {pmid}: {str(e)}")
                continue
            
        return papers
    
    def search_google_scholar(self, start_date: str, end_date: str) -> List[Paper]:
        print("\nSearching Google Scholar...")
        
        # Use scholarly's search_keyword method
        search_query = scholarly.search_keyword("pip install --upgrade scholarly")
        papers = []
        
        try:
            # Convert dates to datetime objects for comparison
            start_dt = datetime.strptime(start_date, '%Y/%m/%d')
            end_dt = datetime.strptime(end_date, '%Y/%m/%d')
            
            with tqdm(desc="Fetching Google Scholar papers") as pbar:
                count = 0
                for result in search_query:
                    try:
                        # Fetch detailed information
                        pub = result
                        if hasattr(pub, 'fill'):
                            pub = pub.fill()
                        
                        # Try to parse publication date
                        try:
                            pub_year = int(pub.bib.get('year', 0))
                            pub_date = datetime(pub_year, 1, 1)  # Default to January 1st if only year is available
                        except (ValueError, TypeError):
                            continue
                        
                        # Check if publication is within date range
                        if not (start_dt <= pub_date <= end_dt):
                            continue
                            
                        paper = self._parse_scholar_record(pub)
                        if paper:
                            papers.append(paper)
                        
                        count += 1
                        pbar.update(1)
                        
                        # Limit to prevent timeout (adjust as needed)
                        if count >= 100:
                            break
                            
                        time.sleep(2)  # Respect Google Scholar rate limits
                        
                    except Exception as e:
                        print(f"\nError processing Google Scholar record: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"\nError in Google Scholar search: {str(e)}")
            
        print(f"Found {len(papers)} papers in Google Scholar")
        return papers
    
    def verify_khcc_affiliation(self, paper: Paper) -> bool:
        prompt = f"""
        Please analyze if this paper is affiliated with King Hussein Cancer Center (KHCC):
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Affiliation: {paper.affiliation}
        Abstract: {paper.abstract}
        
        Return only 'true' if it's affiliated with KHCC, or 'false' if not.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip().lower() == 'true'
    
    def get_impact_factor(self, paper: Paper) -> Dict:
        search_term = paper.issn if paper.issn else paper.journal
        result = self.factor.search(search_term)
        
        if result:
            return {
                'jcr': result[0].get('jcr'),
                'factor': result[0].get('factor')
            }
        return {'jcr': None, 'factor': None}
    
    def _parse_pubmed_record(self, record) -> Optional[Paper]:
        try:
            article = record['MedlineCitation']['Article']
            
            # Extract basic information
            title = article.get('ArticleTitle', '')
            
            # Extract authors
            author_list = article.get('AuthorList', [])
            authors = []
            for author in author_list:
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['LastName']} {author['ForeName']}")
                elif 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])
            
            # Extract journal information
            journal = article.get('Journal', {}).get('Title', '')
            issn = article.get('Journal', {}).get('ISSN', '')
            
            # Extract date
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')
            publication_date = f"{year}/{month}/{day}"
            
            # Extract abstract
            abstract = ''
            if 'Abstract' in article:
                abstract = article['Abstract'].get('AbstractText', [''])[0]
            
            # Extract affiliations
            affiliations = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'AffiliationInfo' in author:
                        for affiliation in author['AffiliationInfo']:
                            if 'Affiliation' in affiliation:
                                affiliations.append(affiliation['Affiliation'])
            
            affiliation_text = '; '.join(affiliations)
            
            # Extract DOI
            for id_obj in article.get('ELocationID', []):
                if id_obj.attributes.get('EIdType') == 'doi':
                    doi = str(id_obj)
                    break
            else:
                doi = None
            
            # Create Paper object
            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                journal=journal,
                doi=doi,
                pmid=record['MedlineCitation'].get('PMID', ''),
                issn=issn,
                citations=None,  # PubMed doesn't provide citation count
                source='pubmed',
                affiliation=affiliation_text,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{record['MedlineCitation'].get('PMID', '')}/",
                keywords=article.get('KeywordList', [[]])[0] if 'KeywordList' in article else None
            )
            
        except Exception as e:
            print(f"Error parsing PubMed record: {str(e)}")
            return None
    
    def _parse_scholar_record(self, pub) -> Optional[Paper]:
        try:
            bib = pub.bib if hasattr(pub, 'bib') else {}
            
            # Extract basic information
            title = bib.get('title', '')
            
            # Extract authors
            authors = bib.get('author', '').split(' and ') if bib.get('author') else []
            
            # Extract journal/source
            journal = bib.get('journal', '') or bib.get('venue', '')
            
            # Extract date
            year = bib.get('year', '')
            publication_date = f"{year}/01/01"  # Default to January 1st as Scholar often only provides year
            
            # Extract citations
            citations = getattr(pub, 'num_citations', None)
            
            # Create Paper object
            return Paper(
                title=title,
                authors=authors,
                abstract=bib.get('abstract', ''),
                publication_date=publication_date,
                journal=journal,
                doi=bib.get('doi', None),
                pmid=None,  # Scholar doesn't provide PMID
                issn=None,  # Scholar doesn't provide ISSN
                citations=citations,
                source='scholar',
                affiliation=bib.get('institution', ''),
                url=bib.get('url', None),
                keywords=None  # Scholar doesn't provide keywords
            )
            
        except Exception as e:
            print(f"Error parsing Google Scholar record: {str(e)}")
            return None
    
    def _parse_date(self, date_str: str) -> str:
        # Implementation to standardize date format
        pass
    
    def _generate_paper_hash(self, paper: Paper) -> str:
        # Generate unique hash for deduplication
        content = f"{paper.title}{paper.publication_date}{paper.journal}"
        return hashlib.md5(content.encode()).hexdigest()

class KHCCPaperTracker:
    def __init__(self, email: str, api_key: str):
        self.searcher = PaperSearcher(email, api_key)
        
    def run(self, start_date: str, end_date: str):
        # 1. Search both sources
        pubmed_papers = self.searcher.search_pubmed(start_date, end_date)
        scholar_papers = self.searcher.search_google_scholar(start_date, end_date)
        
        # 2. Merge and deduplicate
        all_papers = self._merge_and_deduplicate(pubmed_papers + scholar_papers)
        
        # 3. Process each paper
        with ThreadPoolExecutor() as executor:
            # Verify KHCC affiliation
            verified_papers = [p for p in all_papers 
                             if executor.submit(self.searcher.verify_khcc_affiliation, p).result()]
            
            # Get impact factors
            for paper in verified_papers:
                impact_data = executor.submit(self.searcher.get_impact_factor, paper).result()
                self._save_to_database(paper, impact_data)
    
    def _merge_and_deduplicate(self, papers: List[Paper]) -> List[Paper]:
        # Create a dictionary to store unique papers using title and date as key
        unique_papers = {}
        
        for paper in papers:
            if not paper:  # Skip None values
                continue
            
            # Generate a unique key for each paper
            paper_hash = self.searcher._generate_paper_hash(paper)
            
            # If paper already exists, merge information preferring PubMed data
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
    
    def _save_to_database(self, paper: Paper, impact_data: Dict):
        # Implementation to save to SQLite database
        pass

def main():
    import argparse
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get credentials from environment variables
    email = os.getenv('EMAIL')
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not email or not api_key:
        raise ValueError("EMAIL and OPENAI_API_KEY must be set in .env file")
    
    parser = argparse.ArgumentParser(description='Track KHCC research papers')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    
    args = parser.parse_args()
    
    tracker = KHCCPaperTracker(email, api_key)
    tracker.run(args.start_date, args.end_date)

if __name__ == "__main__":
    main()