+# Design: Add Paper Upload Tab

## Architecture Overview

This feature adds a new end-to-end workflow for paper ingestion:

```
User Upload → Validation → LLM Extraction → Database Update → Storage
```

### Component Breakdown

#### 1. Frontend (templates/index.html)
- Add fourth tab: "Add new paper"
- File upload input with drag-and-drop support
- Progress indicators during processing
- Error display for validation failures
- Success confirmation with extracted metadata preview

#### 2. Backend API (agent2.py)
- New route: `/upload_paper` (POST) for file upload
- Orchestrates validation, extraction, and storage
- Handles error responses with detailed messages

#### 3. PDF Validation Module (system_api/pdf_validator.py)
- Extract text from PDF using PyPDF2
- Check for required sections using keyword matching and LLM assistance
- Support bilingual content (Chinese, English, or mixed)
- Required fields:
  - 論文標題 (Paper Title)
  - 年份 (Year)
  - 作者 (Author)
  - 摘要 (Abstract)
  - 研究目的 (Research Purpose)
  - 相關研究 (Related Work)
  - 所使用資料集 (Datasets Used)
  - 資料前處理方式 (Data Preprocessing)
  - 建模方式 (Modeling Approach)
  - 評估指標 (Evaluation Metrics)

#### 4. Paper Extraction Module (system_api/paper_extractor.py)
- Use Ollama llama3.1:8b to extract structured data
- Generate bilingual prompts for field extraction
- Parse LLM responses into structured format
- Handle Chinese, English, and mixed-language content
- Preserve original language in extracted fields

#### 5. Database Update Module (system_api/database_updater.py)
- Update `data_categories.json` with new paper entry
- Determine research lineage using LLM analysis
- Update `relationships.json` with appropriate placement
- Atomic updates (rollback on failure)

### Data Flow

1. **Upload Phase**
   - User selects PDF file
   - Frontend sends file to `/upload_paper`
   - Backend saves to temporary location

2. **Validation Phase**
   - Extract text from PDF
   - Check for required sections
   - If validation fails: delete temp file, return error
   - If validation passes: proceed to extraction

3. **Extraction Phase**
   - LLM extracts structured fields from PDF text
   - Generate data_categories entry
   - LLM analyzes research context to determine lineage

4. **Update Phase**
   - Backup current JSON files
   - Update data_categories.json with new entry
   - Update relationships.json with lineage placement
   - Move PDF from temp to data/ directory with title-based name
   - If any step fails: rollback changes

5. **Response Phase**
   - Return success with extracted metadata
   - Or return error with specific failure reason

### Error Handling Strategy

- **Validation Errors**: Return 400 with list of missing fields
- **Extraction Errors**: Return 500 with LLM error details
- **Database Errors**: Rollback and return 500 with conflict details
- **Storage Errors**: Clean up partial changes, return 500

### LLM Prompt Design

**Extraction Prompt Template** (Bilingual Support):
```
你是一個學術論文分析助手。請從以下論文內容中提取結構化資訊。論文可能是中文、英文或中英混合。
You are an academic paper analysis assistant. Please extract structured information from the following paper. The paper may be in Chinese, English, or mixed.

論文內容 / Paper Content:
{pdf_text}

請提取以下資訊並以JSON格式返回 / Please extract and return in JSON format:
1. 論文標題 / Paper Title (保持原語言 / keep original language)
2. 年份 / Year
3. 作者 / Authors
4. 研究目的 / Research Purpose (列表形式 / list format)
5. 所使用資料集 / Datasets Used (列表形式 / list format)
6. 資料前處理方式 / Data Preprocessing (列表形式 / list format)
7. 建模方式 / Modeling Methods (列表形式 / list format)
8. 評估指標 / Evaluation Metrics (列表形式 / list format)

請以JSON格式返回，格式如下 / Return JSON format:
{example_json}
```

**Lineage Analysis Prompt Template** (Bilingual Support):
```
請分析以下新論文應該歸類到哪個研究領域和主幹。論文可能是中文或英文。
Please analyze which research area and trunk the following new paper should be classified into. The paper may be in Chinese or English.

現有分類 / Existing Classifications:
{relationships_structure}

新論文資訊 / New Paper Information:
{paper_metadata}

請返回JSON格式 / Return JSON format:
{
  "area_name": "...",
  "trunk_name": "...",
  "reasoning": "... (可用中英文混合 / may use mixed Chinese-English)"
}
```

### File Storage Convention

- PDFs stored in `data/` directory
- Filename format: `{論文標題}.pdf`
- Sanitize filenames: remove special characters, limit length
- Handle duplicate titles by appending counter

### Atomic Update Implementation

Use try-except-finally pattern:
```python
backup_files = backup_json_files()
try:
    update_data_categories()
    update_relationships()
    move_pdf_to_data()
    cleanup_temp_files()
except Exception:
    restore_from_backup(backup_files)
    raise
finally:
    remove_backup_files()
```

## Technology Choices

- **PyPDF2**: Already in requirements.txt, handles PDF text extraction
- **Ollama llama3.1:8b**: Specified by user, handles both Chinese and English content well, supports bilingual prompts
- **Flask**: Existing framework, add new route
- **JSON**: Existing data format, maintain compatibility

## Language Support

- **Supported Languages**: Chinese (中文), English, or mixed Chinese-English content
- **Field Detection**: Bilingual keyword matching for section headers
- **LLM Prompts**: Bilingual instructions to handle both languages
- **Data Preservation**: Original language preserved in extracted fields (e.g., titles, author names)
- **Classification**: Research area classification works for both Chinese and English papers

## Security Considerations

- Validate file type (must be PDF)
- Limit file size (e.g., 50MB max)
- Sanitize filenames to prevent path traversal
- Use temporary directory for upload processing
- Validate JSON structure before writing

## Performance Considerations

- LLM extraction may take 10-30 seconds
- Show progress indicators to user
- Process uploads asynchronously if needed in future
- Consider timeout limits on LLM calls

## Future Enhancements (Out of Scope)

- Batch upload multiple papers
- Edit/update existing paper metadata
- Manual correction of LLM extraction errors
- Version history for database changes
- OCR support for scanned PDFs
