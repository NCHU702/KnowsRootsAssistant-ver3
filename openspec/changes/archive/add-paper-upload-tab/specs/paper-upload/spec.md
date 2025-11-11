# Paper Upload UI Specification

## ADDED Requirements

### REQ-PU-001: Upload Tab Interface
The system shall provide a new tab labeled "Add new paper" in the main interface alongside existing tabs (General Assistant, Inheritance Analysis, Data Categories).

#### Scenario: User navigates to upload tab
**Given** the user is on the main application page  
**When** the user clicks on the "Add new paper" tab  
**Then** the upload interface is displayed  
**And** other tabs are hidden  
**And** the tab is marked as active

---

### REQ-PU-002: File Upload Control
The system shall provide a file upload input that accepts PDF files only.

#### Scenario: User selects valid PDF file
**Given** the user is on the "Add new paper" tab  
**When** the user clicks the file upload button  
**And** selects a PDF file  
**Then** the file is prepared for upload  
**And** the filename is displayed to the user

#### Scenario: User attempts to upload non-PDF file
**Given** the user is on the "Add new paper" tab  
**When** the user attempts to select a non-PDF file  
**Then** the file selection is rejected  
**And** an error message states "Only PDF files are accepted"

---

### REQ-PU-003: Upload Progress Indication
The system shall display upload and processing status to the user during paper submission.

#### Scenario: Upload in progress
**Given** the user has submitted a valid PDF  
**When** the system is processing the upload  
**Then** a progress indicator is displayed  
**And** processing status messages are shown (e.g., "Validating PDF...", "Extracting information...", "Updating database...")  
**And** the upload button is disabled during processing

#### Scenario: Upload completes successfully
**Given** the system has successfully processed the paper  
**When** all database updates are complete  
**Then** a success message is displayed  
**And** extracted metadata is shown to the user  
**And** the form is reset for new uploads

---

### REQ-PU-004: Error Display
The system shall display clear error messages when upload or processing fails.

#### Scenario: Validation failure
**Given** the uploaded PDF fails validation  
**When** the system detects missing required fields  
**Then** an error message is displayed  
**And** the message lists all missing fields  
**And** the upload is cancelled  
**And** the temporary file is deleted

#### Scenario: Processing failure
**Given** an error occurs during LLM extraction or database update  
**When** the system cannot complete the operation  
**Then** an error message explaining the failure is displayed  
**And** no database changes are persisted  
**And** the user can retry the upload

---

### REQ-PU-005: Drag-and-Drop Support
The system should support drag-and-drop file upload for improved user experience.

#### Scenario: User drags PDF file to upload area
**Given** the user is on the "Add new paper" tab  
**When** the user drags a PDF file over the upload area  
**Then** the upload area highlights to indicate drop target  
**When** the user drops the file  
**Then** the upload process begins

---

### REQ-PU-006: Metadata Preview
The system shall display extracted paper metadata before finalizing the upload.

#### Scenario: Successful extraction preview
**Given** the system has extracted paper information  
**When** LLM processing completes  
**Then** extracted fields are displayed for user review:
- 論文標題 (Title)
- 年份 (Year)  
- 作者 (Author)
- 研究目的 (Research Purpose)
- 資料集 (Datasets)
- 資料前處理 (Preprocessing)
- 建模方式 (Modeling)
- 評估指標 (Metrics)
- 研究領域分類 (Research Area Classification)

**And** the user sees a confirmation button to finalize the upload  
**And** the user can cancel to abort the operation

---

### REQ-PU-007: Upload Confirmation
The system shall require explicit user confirmation before persisting database changes.

#### Scenario: User confirms upload
**Given** extracted metadata is displayed  
**When** the user clicks "Confirm" button  
**Then** database files are updated  
**And** PDF is stored in data directory  
**And** success message is displayed

#### Scenario: User cancels upload
**Given** extracted metadata is displayed  
**When** the user clicks "Cancel" button  
**Then** no database changes are made  
**And** temporary files are deleted  
**And** the form is reset
