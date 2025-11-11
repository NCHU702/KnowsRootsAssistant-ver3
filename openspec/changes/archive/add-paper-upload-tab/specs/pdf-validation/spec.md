# PDF Validation Specification

## ADDED Requirements

### REQ-PV-001: File Type Validation
The system shall validate that uploaded files are PDF format.

#### Scenario: Valid PDF file
**Given** a user uploads a file  
**When** the file has .pdf extension  
**And** the file has valid PDF magic bytes  
**Then** the file passes type validation  
**And** processing continues

#### Scenario: Invalid file type
**Given** a user uploads a file  
**When** the file is not a PDF  
**Then** validation fails immediately  
**And** error message "Invalid file type. Only PDF files are accepted." is returned  
**And** the file is rejected before text extraction

---

### REQ-PV-002: File Size Validation
The system shall enforce a maximum file size limit for uploaded PDFs.

#### Scenario: PDF within size limit
**Given** a user uploads a PDF file  
**When** the file size is 50MB or less  
**Then** the file passes size validation  
**And** processing continues

#### Scenario: PDF exceeds size limit
**Given** a user uploads a PDF file  
**When** the file size exceeds 50MB  
**Then** validation fails  
**And** error message "File size exceeds maximum limit of 50MB" is returned  
**And** the upload is rejected

---

### REQ-PV-003: Text Extractability Check
The system shall verify that text can be extracted from the PDF.

#### Scenario: Searchable PDF
**Given** a valid PDF file  
**When** the system attempts text extraction  
**And** extractable text is present  
**Then** the PDF passes extractability check  
**And** processing continues

#### Scenario: Scanned or image-only PDF
**Given** a valid PDF file  
**When** the system attempts text extraction  
**And** less than 100 characters are extracted  
**Then** validation fails  
**And** error message "PDF contains insufficient text. Please upload a searchable PDF, not a scanned image." is returned

---

### REQ-PV-004: Required Field Detection
The system shall validate that the PDF contains all required sections for paper metadata.

Required fields:
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

#### Scenario: PDF contains all required fields
**Given** a valid PDF with extracted text  
**When** the system checks for required sections  
**And** all 10 required fields are detected  
**Then** validation passes  
**And** the PDF proceeds to extraction phase

#### Scenario: PDF missing required fields
**Given** a valid PDF with extracted text  
**When** the system checks for required sections  
**And** one or more required fields are missing  
**Then** validation fails  
**And** error message lists all missing fields  
**And** example: "Missing required fields: 研究目的, 評估指標"  
**And** the upload is cancelled  
**And** temporary files are deleted

---

### REQ-PV-005: Content Language Support
The system shall accept research papers written in Chinese or English, or mixed Chinese-English content.

#### Scenario: Chinese language paper
**Given** a valid PDF with extracted text  
**When** the system analyzes the text content  
**And** the text is primarily in Chinese  
**Then** content validation passes  
**And** processing continues with Chinese language extraction

#### Scenario: English language paper
**Given** a valid PDF with extracted text  
**When** the system analyzes the text content  
**And** the text is primarily in English  
**Then** content validation passes  
**And** processing continues with English language extraction

#### Scenario: Mixed Chinese-English paper
**Given** a valid PDF with extracted text  
**When** the system analyzes the text content  
**And** the text contains both Chinese and English sections  
**Then** content validation passes  
**And** processing continues with bilingual extraction

#### Scenario: Non-Chinese/English language
**Given** a valid PDF with extracted text  
**When** the system analyzes the text content  
**And** the text is not primarily in Chinese or English  
**Then** validation fails with warning  
**And** error message "Paper must be in Chinese or English. Other languages are not currently supported." is returned

---

### REQ-PV-006: Field Detection Method
The system shall use a hybrid approach of keyword matching and LLM analysis to detect required fields.

#### Scenario: Keyword-based detection for bilingual content
**Given** extracted PDF text in Chinese, English, or mixed language  
**When** the validator searches for field keywords  
**Then** common section headers in both languages are matched:
- "摘要" or "Abstract" → 摘要
- "研究目的" or "Objective" or "Purpose" or "Research Objective" → 研究目的
- "相關研究" or "Related Work" or "Literature Review" or "背景" or "Background" → 相關研究
- "資料集" or "Dataset" or "Data" → 所使用資料集
- "前處理" or "Preprocessing" or "Data Preprocessing" → 資料前處理方式
- "模型" or "Model" or "方法" or "Method" or "Methodology" → 建模方式
- "評估" or "Evaluation" or "Metrics" or "實驗結果" or "Results" → 評估指標

#### Scenario: LLM-assisted field detection for bilingual papers
**Given** extracted PDF text in Chinese, English, or mixed language  
**When** keyword matching has low confidence  
**Then** LLM is invoked with bilingual prompt:
```
以下是一篇論文的文本（可能為中文、英文或中英混合）。請判斷是否包含以下部分，並返回JSON格式結果：
The following is a paper text (may be in Chinese, English, or mixed). Please determine if it contains the following sections and return JSON format results:

必需欄位 / Required fields:
{required_fields_list}

論文內容 / Paper content:
{pdf_text}

返回格式 / Return format:
{
  "論文標題": true/false,
  "年份": true/false,
  ...
}
```
**And** the LLM response determines field presence

---

### REQ-PV-007: Validation Error Reporting
The system shall provide detailed validation error messages that help users understand what is missing.

#### Scenario: Multiple validation failures
**Given** a PDF fails validation  
**When** multiple issues are detected  
**Then** all issues are reported in a structured format:
```
Validation Failed:
- Missing required field: 研究目的
- Missing required field: 評估指標  
- Insufficient text content (possible scanned image)
```
**And** the most critical error is highlighted first

#### Scenario: Validation success message
**Given** a PDF passes all validation checks  
**When** validation completes  
**Then** a success indicator is logged  
**And** the user sees progress update: "Validation successful. Extracting information..."
