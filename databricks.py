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

@dataclass
class Paper:
    """Data class representing a research paper from multiple sources"""
    # Core fields (common to both sources)
    title: str
    authors: List[str]
    abstract: str
    abstract_summary: Optional[str]  # GPT-4 generated summary
    publication_date: str
    journal: str
    doi: Optional[str]
    citations: Optional[int]
    source: str  # 'pubmed' or 'openalex'
    affiliation: str
    url: Optional[str]
    keywords: Optional[List[str]]
    paper_hash: Optional[str]
    
    # PubMed specific fields
    pmid: Optional[str] = None
    issn: Optional[str] = None
    impact_factor: Optional[float] = None
    quartile: Optional[str] = None
    categories: Optional[str] = None
    
    # OpenAlex specific fields
    openalex_id: Optional[str] = None
    type: Optional[str] = None
    open_access: Optional[Dict] = None
    primary_location: Optional[Dict] = None
    locations_count: Optional[int] = None
    authorships: Optional[List[Dict]] = None
    cited_by_count: Optional[int] = None
    biblio: Optional[Dict] = None
    is_retracted: bool = False
    is_paratext: bool = False
    concepts: Optional[List[Dict]] = None
    mesh: Optional[List[Dict]] = None
    referenced_works: Optional[List[str]] = None
    related_works: Optional[List[str]] = None
    counts_by_year: Optional[List[Dict]] = None
    sustainable_development_goals: Optional[List[Dict]] = None
    grants: Optional[List[Dict]] = None

    def normalize_doi(self) -> Optional[str]:
        """Normalize DOI to a standard format"""
        if not self.doi:
            return None
            
        # Remove common prefixes
        doi = self.doi.lower()
        prefixes = [
            'https://doi.org/',
            'http://doi.org/',
            'doi.org/',
            'doi:'
        ]
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break
        
        return doi.strip()

    def generate_hash(self) -> str:
        """Generate a unique hash for this paper"""
        # Always use normalized DOI if available
        normalized_doi = self.normalize_doi()
        if normalized_doi:
            return hashlib.sha256(normalized_doi.encode()).hexdigest()
        
        # Fallback to OpenAlex ID
        if self.openalex_id:
            return hashlib.sha256(self.openalex_id.encode()).hexdigest()
            
        # Use PMID as last resort for PubMed papers
        if self.pmid:
            return hashlib.sha256(f"pmid:{self.pmid}".encode()).hexdigest()
        
        # Final fallback to combination of fields
        content = {
            'title': self.title,
            'date': self.publication_date,
            'journal': self.journal,
            'authors': ','.join(sorted(self.authors)) if self.authors else ''
        }
        
        hash_content = json.dumps(content, sort_keys=True)
        return hashlib.sha256(hash_content.encode()).hexdigest()

    def merge_with(self, other: 'Paper') -> 'Paper':
        """Merge this paper with another paper, preferring OpenAlex data for overlapping fields"""
        if not isinstance(other, Paper):
            raise ValueError("Can only merge with another Paper object")
        
        # Normalize DOIs for comparison
        self_doi = self.normalize_doi()
        other_doi = other.normalize_doi()
        
        # If DOIs don't match and both exist, these are different papers
        if self_doi and other_doi and self_doi != other_doi:
            raise ValueError("Cannot merge papers with different DOIs")
            
        # Prefer OpenAlex as the primary source
        if self.source == 'openalex':
            base_paper = self
            other_paper = other
        elif other.source == 'openalex':
            base_paper = other
            other_paper = self
        else:
            # If neither is from OpenAlex, prefer the one with more data
            base_paper = self if len(self.abstract or '') > len(other.abstract or '') else other
            other_paper = other if base_paper == self else self
        
        # Create new paper with merged data
        merged_data = asdict(base_paper)
        
        # Add PubMed specific fields if they exist
        pubmed_fields = {
            'pmid': other_paper.pmid,
            'issn': other_paper.issn,
            'impact_factor': other_paper.impact_factor,
            'quartile': other_paper.quartile,
            'categories': other_paper.categories
        }
        
        for field, value in pubmed_fields.items():
            if value and not merged_data.get(field):
                merged_data[field] = value
        
        # Combine keywords and remove duplicates
        all_keywords = set(base_paper.keywords or []) | set(other_paper.keywords or [])
        merged_data['keywords'] = sorted(list(all_keywords))
        
        # Use the most recent abstract summary
        if other_paper.abstract_summary and (not base_paper.abstract_summary or 
            len(other_paper.abstract_summary) > len(base_paper.abstract_summary)):
            merged_data['abstract_summary'] = other_paper.abstract_summary
        
        # Ensure DOI is normalized
        if merged_data.get('doi'):
            merged_data['doi'] = f"https://doi.org/{self.normalize_doi()}"
        
        return Paper(**merged_data)

    def to_dict(self) -> Dict:
        """Convert paper to dictionary format"""
        data = asdict(self)
        # Convert complex objects to JSON strings
        for key, value in data.items():
            if isinstance(value, (list, dict)) and value is not None:
                data[key] = json.dumps(value)
        return data

class PaperSearcher:
    """Class to search for papers across multiple sources"""
    
    def __init__(self):
        """Initialize searcher with API keys from Databricks secrets"""
        # Get API keys from vault
        self.email = dbutils.secrets.get(scope="key-vault-secrrt", key="EMAIL")
        self.openai_api_key = dbutils.secrets.get(scope="key-vault-secrrt", key="AZURE-API-KEY")
        
        # Initialize clients
        Entrez.email = self.email
        self.openai_client = AzureOpenAI(
            api_key=self.openai_api_key,
            api_version="2024-08-01-preview",
            azure_endpoint="https://openai-aidi.openai.azure.com"
        )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.rate_limit_window = 60
        self.max_requests_per_minute = 45

    def _setup_logging(self):
        """Setup logging to DBFS"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        log_dir = "/dbfs/FileStore/khcc_paper_tracker/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, "paper_tracker.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Initialized logging to {log_path}")

    def verify_khcc_affiliation(self, paper: Paper) -> bool:
        """Verify if paper has KHCC affiliation"""
        # First check OpenAlex ID for KHCC
        if paper.source == 'openalex':
            for authorship in paper.authorships or []:
                for institution in authorship.get('institutions', []):
                    if institution.get('id') == 'I2799468983':  # KHCC's OpenAlex ID
                        return True

        # Then check affiliation text
        affiliation_text = paper.affiliation.lower()
        khcc_variants = [
            'king hussein cancer',
            'khcc',
            'king hussain cancer',  # Common misspelling
            'king husein cancer',   # Another misspelling
            'king hussein cancer center',
            'king hussein cancer centre'
        ]
        
        # Exclude false positives
        exclude_terms = [
            'king hussein medical',
            'king hussein hospital',
            'khmc'
        ]
        
        # Check for excluded terms first
        if any(term in affiliation_text for term in exclude_terms):
            self._log_non_affiliated_paper(paper, "Excluded due to medical city/hospital mention")
            return False
            
        # Check for KHCC variants
        if any(variant in affiliation_text for variant in khcc_variants):
            return True
            
        # If no match found, log and return False
        self._log_non_affiliated_paper(paper, "No KHCC affiliation found")
        return False

    def _log_non_affiliated_paper(self, paper: Paper, reason: str):
        """Log papers that were excluded due to non-KHCC affiliation"""
        log_message = f"""
        {'='*80}
        NON-AFFILIATED PAPER:
        Reason: {reason}
        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Affiliation: {paper.affiliation}
        Source: {paper.source}
        DOI: {paper.doi}
        {'='*80}
        """
        
        # Log to console
        print(log_message)
        
        # Log to DBFS
        log_path = "/dbfs/FileStore/khcc_paper_tracker/logs/non_affiliated_papers.log"
        try:
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {log_message}\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

    def search_papers(self, start_date: str, end_date: str) -> List[Paper]:
        """Search both PubMed and OpenAlex for papers"""
        # Search both sources
        pubmed_papers = self.search_pubmed(start_date, end_date)
        openalex_papers = self.search_openalex(start_date, end_date)
        
        # Filter papers by KHCC affiliation
        print("\nVerifying KHCC affiliations...")
        verified_pubmed = []
        verified_openalex = []
        
        for paper in tqdm(pubmed_papers, desc="Verifying PubMed papers"):
            if self.verify_khcc_affiliation(paper):
                verified_pubmed.append(paper)
                
        for paper in tqdm(openalex_papers, desc="Verifying OpenAlex papers"):
            if self.verify_khcc_affiliation(paper):
                verified_openalex.append(paper)
                
        print(f"\nVerified papers: {len(verified_pubmed)} from PubMed, {len(verified_openalex)} from OpenAlex")
        
        # Merge results
        merged_papers = self.merge_results(verified_pubmed, verified_openalex)
        
        # Generate summaries
        self.generate_summaries(merged_papers)
        
        return merged_papers

    def search_pubmed(self, start_date: str, end_date: str) -> List[Paper]:
        """Search PubMed for KHCC papers"""
        print("\nSearching PubMed...")
        papers = []
        
        try:
            query = f'("King Hussein Cancer Center"[Affiliation]) AND ("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            
            handle = Entrez.esearch(db="pubmed", term=query, retmax=1000, retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            
            total_papers = len(record["IdList"])
            print(f"Found {total_papers} papers in PubMed")
            
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
                        self.logger.error(f"Error processing PMID {pmid}: {str(e)}")
                        continue
            
        except Exception as e:
            self.logger.error(f"Error in PubMed search: {str(e)}")
        
        return papers

    def search_openalex(self, start_date: str, end_date: str) -> List[Paper]:
        """Search OpenAlex for KHCC papers"""
        print("\nSearching OpenAlex...")
        papers = []
        
        try:
            # Format dates for OpenAlex API
            start = datetime.strptime(start_date, '%Y/%m/%d').strftime('%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y/%m/%d').strftime('%Y-%m-%d')
            
            params = {
                'filter': f'authorships.institutions.id:I2799468983,from_publication_date:{start},to_publication_date:{end}',
                'per_page': 200,
                'cursor': '*',
                'select': ('id,doi,title,display_name,publication_date,authorships,abstract_inverted_index,'
                          'cited_by_count,concepts,primary_location,type,language,keywords,biblio,mesh,'
                          'open_access,locations_count,is_retracted,is_paratext,referenced_works,'
                          'related_works,counts_by_year,sustainable_development_goals,grants'),
                'sort': 'publication_date:desc'
            }
            
            total_processed = 0
            with tqdm(desc="Fetching OpenAlex papers") as pbar:
                while True:
                    try:
                        response = requests.get(
                            "https://api.openalex.org/works",
                            params=params,
                            headers={'User-Agent': f'KHCC Paper Tracker ({self.email})'}
                        )
                        
                        if response.status_code != 200:
                            self.logger.error(f"Error response: {response.text}")
                            break
                        
                        data = response.json()
                        results = data.get('results', [])
                        
                        if not results:
                            break
                        
                        for result in results:
                            paper = self._parse_openalex_result(result)
                            if paper:
                                papers.append(paper)
                                pbar.update(1)
                                total_processed += 1
                        
                        next_cursor = data.get('meta', {}).get('next_cursor')
                        if not next_cursor:
                            break
                            
                        params['cursor'] = next_cursor
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        self.logger.error(f"Error fetching results: {str(e)}")
                        break
            
            print(f"\nTotal papers processed: {total_processed}")
            
        except Exception as e:
            self.logger.error(f"Error in OpenAlex search: {str(e)}")
        
        return papers

    def merge_results(self, pubmed_papers: List[Paper], openalex_papers: List[Paper]) -> List[Paper]:
        """Merge papers from different sources, preferring OpenAlex data when available"""
        print("\nMerging results from PubMed and OpenAlex...")
        merged_papers = {}
        
        # First add all OpenAlex papers
        for paper in openalex_papers:
            paper_hash = paper.generate_hash()
            merged_papers[paper_hash] = paper
        
        # Then merge or add PubMed papers
        for paper in pubmed_papers:
            paper_hash = paper.generate_hash()
            if paper_hash in merged_papers:
                # Merge with existing OpenAlex paper
                merged_papers[paper_hash] = merged_papers[paper_hash].merge_with(paper)
            else:
                merged_papers[paper_hash] = paper
        
        return list(merged_papers.values())

    def generate_summaries(self, papers: List[Paper]):
        """Generate summaries for paper abstracts using GPT-4"""
        print("\nGenerating abstract summaries...")
        
        for paper in tqdm(papers, desc="Generating summaries"):
            if not paper.abstract or paper.abstract_summary:
                continue
                
            try:
                response = self.openai_client.chat.completions.create(
                    deployment_name="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a scientific paper summarizer. Create a one-sentence summary of the following abstract that captures the main research objective and key findings."},
                        {"role": "user", "content": paper.abstract}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                paper.abstract_summary = response.choices[0].message.content.strip()
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error generating summary for paper {paper.title}: {str(e)}")
                continue

    def _parse_pubmed_record(self, record) -> Optional[Paper]:
        """Parse PubMed record into Paper object"""
        try:
            article = record['MedlineCitation']['Article']
            
            # Extract DOI first
            doi = None
            for id_obj in article.get('ELocationID', []):
                if id_obj.attributes.get('EIdType') == 'doi':
                    doi = str(id_obj)
                    if not doi.startswith('https://doi.org/'):
                        doi = f"https://doi.org/{doi}"
                    break
            
            # Extract authors with affiliations
            authors = []
            affiliations = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    # Extract author name
                    if 'LastName' in author and 'ForeName' in author:
                        authors.append(f"{author['LastName']} {author['ForeName']}")
                    elif 'CollectiveName' in author:
                        authors.append(author['CollectiveName'])
                    
                    # Extract affiliations
                    if 'AffiliationInfo' in author:
                        for affiliation in author['AffiliationInfo']:
                            if 'Affiliation' in affiliation:
                                affiliations.append(affiliation['Affiliation'])
            
            # Extract and format date
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')
            
            # Handle month names
            try:
                if not month.isdigit():
                    month = datetime.strptime(month, '%b').strftime('%m')
            except ValueError:
                month = '01'
            
            publication_date = f"{year}/{month}/{day}"
            
            # Extract abstract
            abstract = ''
            abstract_text = article.get('Abstract', {}).get('AbstractText', [''])
            if isinstance(abstract_text, list):
                abstract = ' '.join(text for text in abstract_text if text)
            else:
                abstract = abstract_text
            
            # Create Paper object
            return Paper(
                title=article.get('ArticleTitle', ''),
                authors=authors,
                abstract=abstract,
                abstract_summary=None,  # Will be generated later
                publication_date=publication_date,
                journal=article.get('Journal', {}).get('Title', ''),
                doi=doi,
                citations=None,
                source='pubmed',
                affiliation='; '.join(set(affiliations)),  # Remove duplicates
                url=f"https://pubmed.ncbi.nlm.nih.gov/{record['MedlineCitation'].get('PMID', '')}/",
                keywords=article.get('KeywordList', [[]])[0] if 'KeywordList' in article else None,
                paper_hash=None,  # Will be generated when needed
                pmid=record['MedlineCitation'].get('PMID', ''),
                issn=article.get('Journal', {}).get('ISSN', '')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PubMed record: {str(e)}")
            return None

    def _parse_openalex_result(self, result: Dict) -> Optional[Paper]:
        """Parse OpenAlex API result into Paper object"""
        try:
            if not result or not isinstance(result, dict):
                return None

            # Extract and normalize DOI
            doi = result.get('doi')
            if doi and not doi.startswith('https://doi.org/'):
                doi = f"https://doi.org/{doi}"
            elif not doi and result.get('primary_location', {}).get('landing_page_url'):
                # Use landing page URL as fallback
                doi = result['primary_location']['landing_page_url']

            # Extract authors with ORCID
            authors = []
            authorships_data = []
            for authorship in result.get('authorships', []) or []:
                author_info = authorship.get('author', {}) or {}
                author_name = author_info.get('display_name')
                orcid = author_info.get('orcid')
                if author_name:
                    if orcid:
                        authors.append(f"{author_name} ({orcid})")
                    else:
                        authors.append(author_name)
                authorships_data.append(authorship)
            
            # Extract affiliations with ROR IDs
            affiliations = []
            for authorship in result.get('authorships', []) or []:
                for institution in authorship.get('institutions', []) or []:
                    name = institution.get('display_name')
                    ror_id = institution.get('ror')
                    if name:
                        if ror_id:
                            affiliations.append(f"{name} (ROR:{ror_id})")
                        else:
                            affiliations.append(name)
            
            # Extract journal info
            location = result.get('primary_location', {}) or {}
            source = location.get('source', {}) or {}
            journal_name = source.get('display_name', '')
            
            # Format publication date
            pub_date = result.get('publication_date')
            if not pub_date:
                return None
            try:
                formatted_date = datetime.strptime(pub_date, '%Y-%m-%d').strftime('%Y/%m/%d')
            except ValueError:
                formatted_date = f"{pub_date}/01/01"
            
            # Extract keywords from concepts
            keywords = []
            for concept in result.get('concepts', []) or []:
                if concept and concept.get('display_name'):
                    level = concept.get('level', 0)
                    score = concept.get('score', 0)
                    if score > 0.5:  # Only include high-confidence concepts
                        keywords.append(f"{concept['display_name']} (L{level}, S:{score:.2f})")
            
            # Reconstruct abstract from inverted index
            abstract = ""
            abstract_index = result.get('abstract_inverted_index', {}) or {}
            if abstract_index:
                word_positions = []
                for word, positions in abstract_index.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                abstract = ' '.join(word for _, word in sorted(word_positions))
            
            # Create Paper object
            return Paper(
                title=result.get('title', ''),
                authors=authors,
                abstract=abstract,
                abstract_summary=None,  # Will be generated later
                publication_date=formatted_date,
                journal=journal_name,
                doi=doi,
                citations=result.get('cited_by_count', 0),
                source='openalex',
                affiliation='; '.join(set(affiliations)),  # Remove duplicates
                url=doi or result.get('id'),  # Use DOI URL as primary URL
                keywords=keywords[:10],  # Limit to top 10 keywords
                paper_hash=None,  # Will be generated when needed
                openalex_id=result.get('id', ''),
                type=result.get('type', ''),
                open_access=result.get('open_access'),
                primary_location=location,
                locations_count=result.get('locations_count', 0),
                authorships=authorships_data,
                cited_by_count=result.get('cited_by_count', 0),
                biblio=result.get('biblio'),
                is_retracted=result.get('is_retracted', False),
                is_paratext=result.get('is_paratext', False),
                concepts=result.get('concepts', []),
                mesh=result.get('mesh', []),
                referenced_works=result.get('referenced_works', []),
                related_works=result.get('related_works', []),
                counts_by_year=result.get('counts_by_year', []),
                sustainable_development_goals=result.get('sustainable_development_goals', []),
                grants=result.get('grants', [])
            )
                
        except Exception as e:
            self.logger.error(f"Error parsing OpenAlex result: {str(e)}")
            return None

class DatabricksManager:
    """Class to manage database operations in Databricks"""
    
    def __init__(self):
        """Initialize database manager"""
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize database connection and schema"""
        try:
            # Create table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS GOLD_Bibliometry (
                paper_hash VARCHAR(64) PRIMARY KEY,
                title NVARCHAR(MAX),
                authors NVARCHAR(MAX),
                abstract NVARCHAR(MAX),
                abstract_summary NVARCHAR(MAX),
                publication_date DATE,
                journal NVARCHAR(500),
                doi VARCHAR(500),
                citations INT,
                source VARCHAR(50),
                affiliation NVARCHAR(MAX),
                url VARCHAR(500),
                keywords NVARCHAR(MAX),
                pmid VARCHAR(50),
                issn VARCHAR(50),
                impact_factor FLOAT,
                quartile VARCHAR(10),
                categories NVARCHAR(MAX),
                openalex_id VARCHAR(100),
                type VARCHAR(100),
                open_access NVARCHAR(MAX),
                primary_location NVARCHAR(MAX),
                locations_count INT,
                authorships NVARCHAR(MAX),
                cited_by_count INT,
                biblio NVARCHAR(MAX),
                is_retracted BIT,
                is_paratext BIT,
                concepts NVARCHAR(MAX),
                mesh NVARCHAR(MAX),
                referenced_works NVARCHAR(MAX),
                related_works NVARCHAR(MAX),
                counts_by_year NVARCHAR(MAX),
                sustainable_development_goals NVARCHAR(MAX),
                grants NVARCHAR(MAX),
                created_at DATETIME DEFAULT GETDATE(),
                updated_at DATETIME DEFAULT GETDATE()
            )
            """
            spark.sql(create_table_sql)
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise

    def save_papers_batch(self, papers: List[Paper]):
        """Save a batch of papers to the database"""
        try:
            # Convert papers to DataFrame
            paper_dicts = []
            for paper in papers:
                paper_dict = paper.to_dict()
                paper_dict['paper_hash'] = paper.generate_hash()
                paper_dicts.append(paper_dict)
            
            df = pd.DataFrame(paper_dicts)
            
            # Convert date columns
            df['publication_date'] = pd.to_datetime(df['publication_date'])
            df['created_at'] = datetime.now()
            df['updated_at'] = datetime.now()
            
            # Convert boolean columns
            df['is_retracted'] = df['is_retracted'].astype(int)
            df['is_paratext'] = df['is_paratext'].astype(int)
            
            # Handle duplicates by updating existing records
            existing_hashes_query = f"""
            SELECT paper_hash 
            FROM GOLD_Bibliometry 
            WHERE paper_hash IN ({','.join([f"'{h}'" for h in df['paper_hash']])})
            """
            existing_hashes = set(row.paper_hash for row in spark.sql(existing_hashes_query).collect())
            
            # Split into new and existing papers
            new_papers = df[~df['paper_hash'].isin(existing_hashes)]
            existing_papers = df[df['paper_hash'].isin(existing_hashes)]
            
            # Insert new papers
            if not new_papers.empty:
                write_to_sql(new_papers, "GOLD_Bibliometry")
                print(f"Added {len(new_papers)} new papers")
            
            # Update existing papers
            if not existing_papers.empty:
                for _, paper in existing_papers.iterrows():
                    update_sql = f"""
                    UPDATE GOLD_Bibliometry
                    SET updated_at = GETDATE(),
                        {','.join([f"{col} = '{str(val).replace("'", "''")}'" 
                                 for col, val in paper.items() 
                                 if col != 'paper_hash' and pd.notna(val)])}
                    WHERE paper_hash = '{paper['paper_hash']}'
                    """
                    spark.sql(update_sql)
                print(f"Updated {len(existing_papers)} existing papers")
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")
            raise

class DatabricksPaperTracker:
    """Main class for tracking papers in Databricks environment"""
    
    def __init__(self):
        """Initialize tracker"""
        self.searcher = PaperSearcher()
        self.db = DatabricksManager()

    def run(self, start_date: str, end_date: str):
        """Run the paper tracking process"""
        try:
            # Search and process papers
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
    import argparse
    parser = argparse.ArgumentParser(description='Track KHCC research papers')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    args = parser.parse_args()
    
    main(args.start_date, args.end_date) 

    #