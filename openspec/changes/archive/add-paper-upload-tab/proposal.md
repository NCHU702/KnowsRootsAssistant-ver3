## Why
Currently, the system can query and analyze existing papers in the database, but there is no interface for adding new papers. This requires manual editing of JSON files and manual PDF uploads to the data directory, which is error-prone and time-consuming.

## What Changes
- Add new "Add new paper" tab to web interface with file upload capabilities
- **Support bilingual papers**: Accept research papers in Chinese, English, or mixed Chinese-English content
- Implement PDF validation to ensure papers contain all required academic sections (論文標題、年份、作者、摘要、研究目的、相關研究、資料集、前處理、建模、評估指標)
- Use bilingual keyword matching for field detection (e.g., "摘要" or "Abstract", "研究目的" or "Research Purpose")
- Create LLM-based extraction system using Ollama llama3.1:8b with bilingual prompts to automatically extract structured metadata from papers
- Implement automatic research lineage classification using LLM analysis (works for both languages)
- Update `data_categories.json` with extracted paper metadata (preserving original language)
- Update `relationships.json` with appropriate research area and trunk placement
- Store PDF files in data/ directory using title-based naming
- Provide detailed error messages for validation failures and processing errors

## Impact
- **Affected specs**: Creates 3 new capabilities (paper-upload, pdf-validation, database-update)
- **Affected code**: 
  - `templates/index.html` - new tab UI and upload interface
  - `agent2.py` - new `/upload_paper` endpoint
  - New modules in `system_api/`: pdf_validator.py, paper_extractor.py, database_updater.py, pdf_storage.py
  - Modified files: `json_files/data_categories.json`, `json_files/relationships.json` (updated by system)
- **User impact**: Users gain ability to add papers through UI; eliminates manual JSON editing
- **Dependencies**: Requires Ollama llama3.1:8b running; uses existing PyPDF2 library
