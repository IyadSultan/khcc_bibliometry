KHCC Paper Tracker
================

A comprehensive tool for tracking and analyzing research papers from King Hussein Cancer Center (KHCC) using OpenAlex and PubMed APIs. This project is designed to run in a Databricks environment and save results to an Azure SQL database.

Project Overview
---------------
The KHCC Paper Tracker automatically:
- Searches for KHCC-affiliated papers in OpenAlex and PubMed
- Verifies KHCC affiliations using institution ID and text matching
- Merges results from both sources, preferring OpenAlex data when available
- Generates paper summaries using Azure OpenAI (GPT-4o-mini)
- Calculates impact factors and journal metrics
- Saves results to a structured database
- Logs non-affiliated papers for review

Setup Requirements
----------------
1. Databricks Environment:
   - Databricks Runtime 13.3 or higher
   - Python 3.8+
   - PySpark environment

2. Required Python Packages:
   - pandas
   - numpy
   - biopython
   - openai
   - requests
   - tqdm
   - python-dotenv

3. Azure Resources:
   - Azure SQL Database
   - Azure OpenAI Service (with GPT-4o-mini model)
   - Azure Key Vault for secrets

4. API Keys and Secrets:
   Required secrets in Databricks Key Vault (scope: "key-vault-secrrt"):
   - EMAIL: Contact email for API access
   - AZURE-API-KEY: Azure OpenAI API key
   - SERPAPI-KEY: SerpAPI key (for Google Scholar)

File Structure
-------------
/
├── openAlexSearch.py    # Core implementation for OpenAlex and PubMed searches
├── databricks.py        # Databricks-specific implementation
├── aidi_functions.py    # Helper functions for SQL operations
└── /dbfs/FileStore/khcc_paper_tracker/
    ├── /logs/          # Log files directory
    └── /impact_factor/ # Journal impact factor data
        └── CopyofImpactFactor2024.xlsx

Database Schema
-------------
Table: GOLD_Bibliometry
- title (varchar(500))
- authors (varchar(1000))
- abstract (varchar(8000))
- abstract_summary (varchar(8000))
- publication_date (date)
- journal (varchar(255))
- doi (varchar(255))
- pmid (varchar(50))
- issn (varchar(20))
- citations (int)
- source (varchar(255))
- affiliation (varchar(500))
- url (varchar(1000))
- keywords (varchar(500))
- paper_hash (varchar(255), PRIMARY KEY)
- journal_name (varchar(255))
- impact_factor (varchar(255))
- quartile (varchar(50))
- categories (varchar(500))

Usage
-----
1. Setup in Databricks:
   ```python
   # Create necessary directories
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker")
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker/logs")
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker/impact_factor")

   # Upload files
   dbutils.fs.cp("openAlexSearch.py", "/FileStore/khcc_paper_tracker/openAlexSearch.py")
   dbutils.fs.cp("impact_factor/CopyofImpactFactor2024.xlsx", 
                 "/FileStore/khcc_paper_tracker/impact_factor/CopyofImpactFactor2024.xlsx")
   ```

2. Run the Script:
   ```python
   from databricks import main
   main("2024/01/01", "2024/01/31")  # Format: YYYY/MM/DD
   ```

3. Monitor Logs:
   - Console output shows real-time progress
   - Detailed logs in /dbfs/FileStore/khcc_paper_tracker/logs/
   - Non-affiliated papers logged separately for review

Error Handling
-------------
The script includes comprehensive error handling:
- API rate limiting and retry logic
- Connection error recovery
- Data validation and cleaning
- Detailed error logging
- Duplicate handling in database operations

Output Files
-----------
1. paper_tracker.log:
   - General execution logs
   - API responses
   - Processing status

2. non_affiliated_papers.log:
   - Papers excluded from processing
   - Reason for exclusion
   - Paper details for review

Maintenance
----------
1. Regular Updates:
   - Update impact factor file annually
   - Check API version compatibility
   - Monitor API rate limits

2. Database Maintenance:
   - Monitor table growth
   - Check for duplicate entries
   - Validate data integrity

Support
-------
For issues or questions:
1. Check the logs in /dbfs/FileStore/khcc_paper_tracker/logs/
2. Verify API keys and permissions
3. Ensure all dependencies are installed
4. Contact system administrator for database issues

Note: This tool is designed specifically for KHCC research paper tracking. Modifications may be needed for other institutions. 

Running in Databricks Notebook
----------------------------
1. Create a New Notebook:
   - Click "Workspace" in Databricks
   - Click "Create" -> "Notebook"
   - Name: "KHCC_Paper_Tracker"
   - Language: Python
   - Cluster: Select your cluster (ensure it meets requirements)

2. Install Required Packages:
   ```python
   # Cmd 1: Install required packages
   %pip install biopython openai requests tqdm python-dotenv

   # Cmd 2: Restart Python interpreter
   dbutils.library.restartPython()
   ```

3. Upload Files to DBFS:
   ```python
   # Cmd 3: Create directories
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker")
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker/logs")
   dbutils.fs.mkdirs("/FileStore/khcc_paper_tracker/impact_factor")

   # Cmd 4: Upload files using Databricks UI
   # Go to "Data" -> "Files" -> "/FileStore/khcc_paper_tracker/"
   # Upload openAlexSearch.py, databricks.py, and impact_factor/CopyofImpactFactor2024.xlsx
   ```

4. Set Up Environment:
   ```python
   # Cmd 5: Import required modules and setup paths
   import sys
   sys.path.append("/dbfs/FileStore/khcc_paper_tracker")
   
   # Cmd 6: Import the main function
   from databricks import main
   ```

5. Run the Script:
   ```python
   # Cmd 7: Execute the main function
   main(
       start_date="2024/01/01",  # Replace with your start date
       end_date="2024/01/31"     # Replace with your end date
   )
   ```

6. Monitor Progress:
   - Check the notebook output for real-time progress
   - View logs in DBFS:
     ```python
     # Cmd 8: View latest logs
     with open("/dbfs/FileStore/khcc_paper_tracker/logs/paper_tracker.log", "r") as f:
         print(f.read())
     ```

7. View Non-Affiliated Papers:
   ```python
   # Cmd 9: View papers that were excluded
   with open("/dbfs/FileStore/khcc_paper_tracker/logs/non_affiliated_papers.log", "r") as f:
       print(f.read())
   ```

8. Check Results in Database:
   ```python
   # Cmd 10: Query the results
   %sql
   SELECT 
     title,
     publication_date,
     journal,
     impact_factor,
     quartile
   FROM AIDI-DB.dbo.GOLD_Bibliometry
   WHERE publication_date >= '2024-01-01'
   ORDER BY publication_date DESC
   ```

Troubleshooting in Databricks
---------------------------
1. Package Installation Issues:
   - Try installing packages one by one
   - Check cluster logs for installation errors
   - Ensure cluster has internet access

2. File Access Issues:
   ```python
   # Check if files exist
   print(dbutils.fs.ls("/FileStore/khcc_paper_tracker"))
   ```

3. Permission Issues:
   - Verify secret scope access:
   ```python
   # Test secret access
   try:
       email = dbutils.secrets.get(scope="key-vault-secrrt", key="EMAIL")
       print("Secret access successful")
   except Exception as e:
       print(f"Secret access failed: {str(e)}")
   ```

4. Database Connection Issues:
   ```python
   # Test database connection
   %sql
   SELECT 1 as test
   ``` 