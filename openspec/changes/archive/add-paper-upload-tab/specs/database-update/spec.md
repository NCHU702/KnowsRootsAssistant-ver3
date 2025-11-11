# Database Update Specification

## ADDED Requirements

### REQ-DU-001: LLM-Based Information Extraction
The system shall use Ollama llama3.1:8b to extract structured paper information from validated PDF text.

#### Scenario: Successful extraction
**Given** a validated PDF with extracted text  
**When** the system invokes LLM with extraction prompt  
**Then** LLM returns structured JSON with all required fields:
- 論文標題 (string)
- 年份 (array of strings)
- 作者 (array of strings)
- 研究目的 (array of strings)
- 資料集 (array of strings)
- 資料前處理 (array of strings)
- 建模 (array of strings)
- 評估指標 (array of strings)
- 其他 (array of strings, optional)

**And** the extracted data matches data_categories.json schema

#### Scenario: Extraction failure
**Given** a validated PDF  
**When** LLM extraction fails or times out  
**Then** error is captured and logged  
**And** user receives error message "Failed to extract paper information. Please try again."  
**And** no database changes are made

---

### REQ-DU-002: Data Categories Update
The system shall add the extracted paper information to data_categories.json.

#### Scenario: Add new paper entry
**Given** successfully extracted paper metadata  
**When** the system updates data_categories.json  
**Then** a new entry is appended to the array  
**And** the entry follows the existing schema structure  
**And** all extracted fields are included  
**And** the JSON file is formatted with proper indentation (4 spaces)

#### Scenario: Duplicate paper detection
**Given** successfully extracted paper metadata  
**When** a paper with the same title and year already exists  
**Then** user is warned about potential duplicate  
**And** user can choose to:
- Continue anyway (adds duplicate entry)
- Cancel upload
- Update existing entry (future enhancement)

---

### REQ-DU-003: Research Lineage Classification
The system shall use LLM to determine the appropriate research area and lineage for the new paper.

#### Scenario: LLM determines research classification
**Given** extracted paper metadata  
**And** current relationships.json structure  
**When** the system invokes LLM with lineage analysis prompt  
**Then** LLM returns classification:
```json
{
  "area_name": "交通數據分析",
  "trunk_name": "人流分析與預測",
  "reasoning": "此論文專注於人流預測相關技術..."
}
```
**And** the classification is validated against existing areas/trunks

#### Scenario: New research area needed
**Given** LLM classification results  
**When** the suggested area_name does not exist in relationships.json  
**Then** the system creates a new research area  
**And** adds the paper as the first trunk in that area

#### Scenario: New trunk needed
**Given** LLM classification results  
**When** the area_name exists but trunk_name does not  
**Then** the system creates a new trunk in the specified area  
**And** adds the paper as the first lineage entry in that trunk

---

### REQ-DU-004: Relationships JSON Update
The system shall add the new paper to relationships.json in the appropriate research lineage.

#### Scenario: Add paper to existing lineage
**Given** LLM has classified the paper  
**And** the target area and trunk exist  
**When** the system updates relationships.json  
**Then** a new lineage entry is added:
```json
{
  "year": "2024",
  "student": "extracted_author",
  "title": "extracted_title",
  "children": []
}
```
**And** the entry is placed in chronological order by year  
**And** the JSON structure remains valid

#### Scenario: Determine parent-child relationships
**Given** a new paper being added to an existing trunk  
**When** the system analyzes temporal and thematic connections  
**Then** LLM determines if the paper should be:
- A root-level lineage entry (independent research line)
- A child of an existing paper (direct continuation)
**And** the placement is based on year, author, and research topic similarity

---

### REQ-DU-005: Atomic Update Transaction
The system shall ensure database updates are atomic - either all changes succeed or all are rolled back.

#### Scenario: Successful atomic update
**Given** validated and extracted paper data  
**When** the update process begins  
**Then** backup copies of both JSON files are created  
**And** data_categories.json is updated  
**And** relationships.json is updated  
**And** PDF is moved to data/ directory  
**And** all changes are committed  
**And** backup files are deleted  
**And** success is returned to user

#### Scenario: Partial failure rollback
**Given** an update transaction in progress  
**When** any step fails (e.g., JSON write error, file move error)  
**Then** all completed changes are rolled back:
- JSON files restored from backup
- Temporary PDF deleted
- No partial state persists
**And** error is logged with failure details  
**And** user receives error message explaining what failed

---

### REQ-DU-006: PDF Storage Management
The system shall store uploaded PDFs in the data/ directory with appropriate naming.

#### Scenario: Store PDF with title-based name
**Given** a successfully validated and extracted paper  
**When** the system stores the PDF file  
**Then** the filename is generated from paper title:
- Use extracted 論文標題
- Sanitize: remove special characters `/\:*?"<>|`
- Replace spaces with underscores
- Limit to 200 characters
- Add .pdf extension
**And** the file is moved to data/ directory

#### Scenario: Handle duplicate filenames
**Given** a PDF filename that already exists  
**When** the system attempts to store the file  
**Then** a counter suffix is added: `{title}_2.pdf`, `{title}_3.pdf`, etc.  
**And** the unique filename is used  
**And** no existing files are overwritten

#### Scenario: Storage failure
**Given** PDF storage is attempted  
**When** storage fails (permissions, disk space, etc.)  
**Then** the error triggers transaction rollback  
**And** user receives error "Failed to store PDF file"  
**And** no database changes persist

---

### REQ-DU-007: LLM Prompt Engineering
The system shall use well-structured prompts to ensure consistent and accurate LLM extraction.

#### Scenario: Extraction prompt structure for bilingual papers
**Given** extracted PDF text in Chinese, English, or mixed language  
**When** the extraction prompt is constructed  
**Then** it includes:
- Bilingual instructions (Chinese and English)
- Complete PDF text (or relevant excerpts if too long)
- Example JSON output format
- Field descriptions in both languages
- Instruction to return ONLY valid JSON
- Instruction to handle both Chinese and English content

Example prompt template:
```
你是一個學術論文分析助手。請從以下論文內容中提取結構化資訊。論文可能是中文、英文或中英混合。
You are an academic paper analysis assistant. Please extract structured information from the following paper content. The paper may be in Chinese, English, or mixed.

論文內容 / Paper Content:
{pdf_text}

請提取以下資訊並以JSON格式返回 / Please extract the following information and return in JSON format:
1. 論文標題 / Paper Title: 論文的完整標題（保持原語言）/ Complete title (keep original language)
2. 年份 / Year: 發表或完成年份（陣列格式）/ Publication or completion year (array format)
3. 作者 / Authors: 作者姓名（陣列格式）/ Author names (array format)
4. 研究目的 / Research Purpose: 研究的主要目標和目的（陣列格式）/ Main research objectives and purposes (array format)
5. 資料集 / Datasets: 使用的資料集名稱（陣列格式）/ Dataset names used (array format)
6. 資料前處理 / Data Preprocessing: 資料前處理方法（陣列格式）/ Data preprocessing methods (array format)
7. 建模 / Modeling: 使用的模型或方法（陣列格式）/ Models or methods used (array format)
8. 評估指標 / Evaluation Metrics: 使用的評估指標（陣列格式）/ Evaluation metrics used (array format)

請返回以下格式的JSON（只返回JSON，不要其他文字）/ Please return JSON in the following format (JSON only, no other text):
{
  "論文標題": "string",
  "年份": ["string"],
  "作者": ["string"],
  "研究目的": ["string"],
  "資料集": ["string"],
  "資料前處理": ["string"],
  "建模": ["string"],
  "評估指標": ["string"],
  "其他": []
}
```

#### Scenario: Lineage analysis prompt structure for bilingual papers
**Given** paper metadata and current relationships structure  
**When** the lineage prompt is constructed  
**Then** it includes:
- Current research areas and trunks summary
- New paper metadata (preserving original language)
- Bilingual instructions to classify and explain reasoning
- JSON output format specification
- Guidance for handling both Chinese and English paper titles/content

Example prompt template:
```
請分析以下新論文應該歸類到哪個研究領域和主幹。論文可能是中文或英文。
Please analyze which research area and trunk the following new paper should be classified into. The paper may be in Chinese or English.

現有研究領域結構 / Existing Research Area Structure:
{simplified_relationships_structure}

新論文資訊 / New Paper Information:
標題 / Title: {title}
研究目的 / Research Purpose: {purposes}
資料集 / Datasets: {datasets}
建模方式 / Modeling: {models}

請判斷此論文最適合的研究領域（area_name）和研究主幹（trunk_name），並提供理由。
Please determine the most suitable research area (area_name) and trunk (trunk_name) for this paper, and provide reasoning.

如果現有分類都不適合，建議新的area_name或trunk_name。
If existing classifications are not suitable, suggest a new area_name or trunk_name.

請返回以下格式的JSON / Please return JSON in the following format:
{
  "area_name": "研究領域名稱",
  "trunk_name": "研究主幹名稱",
  "is_new_area": false,
  "is_new_trunk": false,
  "reasoning": "分類理由說明（可用中英文混合說明）"
}
```

---

### REQ-DU-008: JSON Schema Validation
The system shall validate JSON structure before writing to ensure data integrity.

#### Scenario: Validate data_categories entry
**Given** extracted paper data  
**When** the system prepares to update data_categories.json  
**Then** the new entry is validated:
- All required fields are present
- Arrays are properly formatted
- No invalid data types
- Matches existing entry schema

#### Scenario: Validate relationships entry
**Given** classified paper data  
**When** the system prepares to update relationships.json  
**Then** the new entry is validated:
- Required fields: year, student, title, children
- Year is valid format
- children is an array (even if empty)
- Placement location exists in current structure

#### Scenario: Validation failure
**Given** invalid data structure  
**When** validation detects schema mismatch  
**Then** the update is aborted  
**And** error is logged with validation details  
**And** no file writes occur

---

### REQ-DU-009: Concurrent Update Protection
The system should handle concurrent upload attempts safely.

#### Scenario: Simultaneous uploads
**Given** multiple users uploading papers concurrently  
**When** database updates occur  
**Then** file locking or transaction queuing prevents conflicts  
**And** each update completes atomically  
**And** no data corruption occurs

#### Scenario: Lock timeout
**Given** a database lock is held too long  
**When** a new upload waits for lock  
**And** timeout (30 seconds) is reached  
**Then** the waiting upload fails with "Database busy, please try again"  
**And** user can retry the upload

---

### REQ-DU-010: Update Logging
The system shall log all database update operations for audit and debugging.

#### Scenario: Successful update logging
**Given** a successful paper upload  
**When** database updates complete  
**Then** log entry includes:
- Timestamp
- Paper title
- Extracted metadata summary
- Research classification
- File storage location

#### Scenario: Failed update logging
**Given** a failed paper upload  
**When** an error occurs  
**Then** log entry includes:
- Timestamp
- Error type and message
- Stack trace
- PDF filename
- Extraction/validation state at failure point
