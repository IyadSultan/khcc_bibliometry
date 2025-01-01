#!/usr/bin/env python3
"""
KHCC Paper Tracker using OpenAlex API
A tool to track research papers from King Hussein Cancer Center using OpenAlex.
"""
import os
import sqlite3
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import time

import requests
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from Bio import Entrez

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

class DatabaseManager:
    def __init__(self, db_path: str = "bibliometry.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database and create table if not exists"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Drop existing table to ensure schema matches
        c.execute('DROP TABLE IF EXISTS papers')
        
        # Create table with updated schema
        c.execute('''CREATE TABLE IF NOT EXISTS papers
                    (title TEXT,
                     authors TEXT,
                     abstract TEXT,
                     abstract_summary TEXT,
                     publication_date DATE,
                     journal TEXT,
                     doi TEXT,
                     citations INTEGER,
                     source TEXT,
                     affiliation TEXT,
                     url TEXT,
                     keywords TEXT,
                     paper_hash TEXT PRIMARY KEY,
                     pmid TEXT,
                     issn TEXT,
                     impact_factor REAL,
                     quartile TEXT,
                     categories TEXT,
                     openalex_id TEXT,
                     type TEXT,
                     open_access TEXT,
                     primary_location TEXT,
                     locations_count INTEGER,
                     authorships TEXT,
                     cited_by_count INTEGER,
                     biblio TEXT,
                     is_retracted BOOLEAN,
                     is_paratext BOOLEAN,
                     concepts TEXT,
                     mesh TEXT,
                     referenced_works TEXT,
                     related_works TEXT,
                     counts_by_year TEXT,
                     sustainable_development_goals TEXT,
                     grants TEXT)''')
        
        conn.commit()
        conn.close()

    def save_papers_batch(self, papers: List[Paper]):
        """Save batch of papers to database"""
        try:
            papers_data = [paper.to_dict() for paper in papers]
            df = pd.DataFrame(papers_data)
            
            if 'paper_hash' not in df.columns:
                df['paper_hash'] = [p.generate_hash() for p in papers]
            
            # Convert dates with proper format handling
            def safe_date_convert(date_str):
                try:
                    if pd.isna(date_str) or not date_str:
                        return None
                    # Handle various date formats
                    if isinstance(date_str, str):
                        if date_str.count('/') == 2:
                            return datetime.strptime(date_str, '%Y/%m/%d').date()
                        elif date_str.count('-') == 2:
                            return datetime.strptime(date_str, '%Y-%m-%d').date()
                        else:
                            # Default to first day of year if only year is provided
                            year = date_str.strip('/')
                            return datetime(int(year), 1, 1).date()
                    return date_str
                except (ValueError, TypeError):
                    return None

            df['publication_date'] = df['publication_date'].apply(safe_date_convert)
            
            # Handle numeric columns
            df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
            df['cited_by_count'] = pd.to_numeric(df['cited_by_count'], errors='coerce').fillna(0).astype(int)
            df['locations_count'] = pd.to_numeric(df['locations_count'], errors='coerce').fillna(0).astype(int)
            
            # Convert boolean columns
            df['is_retracted'] = df['is_retracted'].fillna(False).astype(bool)
            df['is_paratext'] = df['is_paratext'].fillna(False).astype(bool)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing paper hashes
            cursor.execute("SELECT paper_hash FROM papers")
            existing_hashes = {row[0] for row in cursor.fetchall()}
            
            # Split dataframe into new and existing papers
            df['exists'] = df['paper_hash'].isin(existing_hashes)
            new_papers = df[~df['exists']].drop('exists', axis=1)
            update_papers = df[df['exists']].drop('exists', axis=1)
            
            # Insert new papers
            if not new_papers.empty:
                new_papers.to_sql('papers', conn, if_exists='append', index=False)
                print(f"Added {len(new_papers)} new papers")
            
            # Update existing papers
            updated_count = 0
            for _, paper in update_papers.iterrows():
                update_cols = [col for col in paper.index if col != 'paper_hash']
                set_clause = ', '.join([f"{col} = ?" for col in update_cols])
                values = [paper[col] for col in update_cols] + [paper['paper_hash']]
                
                cursor.execute(
                    f"UPDATE papers SET {set_clause} WHERE paper_hash = ?",
                    values
                )
                if cursor.rowcount > 0:
                    updated_count += 1
            
            if updated_count > 0:
                print(f"Updated {updated_count} existing papers")
            
            conn.commit()
            conn.close()
            
            total_processed = len(new_papers) + updated_count
            print(f"\nSuccessfully processed {total_processed} papers total")
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            raise

class PaperSearcher:
    """Class to search for papers across multiple sources"""
    
    def __init__(self, email: str, openai_api_key: Optional[str] = None):
        self.email = email
        Entrez.email = email
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.db = DatabaseManager()
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        log_dir = "logs"
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
        
        # Log to file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "non_affiliated_papers.log")
        
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
        
        # Generate summaries if OpenAI client is available
        if self.openai_client:
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
                    model="gpt-4o-mini",
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
                abstract_summary=None,  # Will be generated later if OpenAI client is available
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
                abstract_summary=None,  # Will be generated later if OpenAI client is available
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

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Track KHCC research papers using PubMed and OpenAlex')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY/MM/DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY/MM/DD)')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    args = parser.parse_args()
    
    try:
        # Load environment variables
        if os.path.exists(args.env_file):
            load_dotenv(args.env_file)
        else:
            print(f"Warning: Environment file {args.env_file} not found")
        
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
                print("Error: Dates must be in YYYY/MM/DD format")
            else:
                print(f"Error: {str(e)}")
            return
        
        # Initialize searcher with OpenAI API key if available
        searcher = PaperSearcher(
            email=email,
            openai_api_key=openai_api_key
        )
        
        # Run the search
        print(f"\nSearching for papers from {args.start_date} to {args.end_date}")
        papers = searcher.search_papers(args.start_date, args.end_date)
        
        # Save papers to database
        if papers:
            print(f"\nSaving {len(papers)} papers to database...")
            searcher.db.save_papers_batch(papers)
            print("\nProcess completed successfully!")
        else:
            print("\nNo papers found in the date range.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import logging
    import json
    import hashlib
    import time
    import os
    from datetime import datetime
    from typing import Dict, List, Optional
    from dataclasses import dataclass, asdict
    import requests
    from tqdm import tqdm
    from dotenv import load_dotenv
    import openai
    from Bio import Entrez
    
    main() 