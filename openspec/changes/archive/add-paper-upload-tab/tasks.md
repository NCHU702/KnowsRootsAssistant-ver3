# Implementation Tasks

## Phase 1: Backend Foundation

### Task 1.1: Create PDF Validator Module
- [x] Create `system_api/pdf_validator.py`
- [x] Implement file type validation (check .pdf extension and magic bytes)
- [x] Implement file size validation (50MB max)
- [x] Implement text extraction using PyPDF2
- [x] Implement extractability check (minimum character count)
- [x] Implement keyword-based field detection for required sections
- [x] Implement LLM-assisted field detection for low-confidence cases
- [x] Create detailed error reporting for validation failures
- [ ] Add unit tests for validator functions

**Validation**: Run tests with valid/invalid PDFs and verify error messages.

---

### Task 1.2: Create Paper Extraction Module
- [x] Create `system_api/paper_extractor.py`
- [x] Initialize Ollama LLM connection (llama3.1:8b)
- [x] Design and implement extraction prompt template
- [x] Implement LLM invocation for paper metadata extraction
- [x] Parse and validate LLM JSON responses
- [x] Handle extraction timeouts and errors
- [x] Add retry logic for transient LLM failures
- [ ] Add unit tests with mock LLM responses

**Validation**: Test extraction with sample PDFs and verify JSON output matches schema.

---

### Task 1.3: Create Database Updater Module
- [x] Create `system_api/database_updater.py`
- [x] Implement JSON file backup mechanism
- [x] Implement data_categories.json update logic
- [x] Implement duplicate paper detection
- [x] Design and implement lineage analysis prompt template
- [x] Implement LLM invocation for research classification
- [x] Implement relationships.json update logic (add to existing trunk)
- [x] Implement new area/trunk creation when needed
- [x] Implement parent-child relationship determination
- [x] Implement atomic transaction with rollback on failure
- [x] Implement JSON schema validation before writing
- [x] Add file locking for concurrent update protection
- [x] Add comprehensive logging for all operations
- [ ] Add unit tests for update operations

**Validation**: Test updates with sample data and verify atomicity and rollback.

---

### Task 1.4: Create PDF Storage Module
- [x] Create `system_api/pdf_storage.py`
- [x] Implement filename sanitization function
- [x] Implement duplicate filename handling (counter suffix)
- [x] Implement file move from temp to data/ directory
- [x] Add error handling for storage failures
- [ ] Add unit tests for filename generation and storage

**Validation**: Test with various paper titles including special characters.

---

### Task 1.5: Create Flask Upload Endpoint
- [x] Add `/upload_paper` POST route in `agent2.py`
- [x] Implement file upload handling (save to temp directory)
- [x] Call pdf_validator for validation
- [x] Call paper_extractor for metadata extraction
- [x] Call database_updater for classification and updates
- [x] Call pdf_storage for final file storage
- [x] Implement comprehensive error handling at each step
- [x] Return detailed success/error responses with appropriate HTTP codes
- [x] Add request logging

**Validation**: Test endpoint with curl/Postman before frontend integration.

---

## Phase 2: Frontend Implementation

### Task 2.1: Add Upload Tab UI
- [x] Update `templates/index.html` with new tab structure
- [x] Add "Add new paper" tab button in tabs section
- [x] Add corresponding tab content div
- [x] Update `switchTab()` JavaScript function to handle new tab
- [x] Update CSS for consistent tab styling
- [x] Add sidebar function item for quick access

**Validation**: Verify tab switching works and styling is consistent.

---

### Task 2.2: Create File Upload Interface
- [x] Add file input element with PDF accept filter
- [x] Add drag-and-drop zone with visual feedback
- [x] Implement file selection handling in JavaScript
- [x] Add file type validation on client side
- [x] Add file size validation on client side (50MB)
- [x] Display selected filename to user
- [x] Add upload button (initially disabled until file selected)

**Validation**: Test file selection and drag-drop on different browsers.

---

### Task 2.3: Implement Progress Indicators
- [x] Create progress indicator component (spinner + status text)
- [x] Add function to show progress with custom messages
- [x] Add function to update progress message
- [x] Add function to hide progress indicator
- [x] Implement progress stages:
  - "Uploading file..."
  - "Validating PDF..."
  - "Extracting information..."
  - "Analyzing research classification..."
  - "Updating database..."
  - "Finalizing..."
- [x] Disable upload button during processing

**Validation**: Verify progress indicators display correctly during upload.

---

### Task 2.4: Create Metadata Preview Component
- [x] Design metadata preview layout (card-based or form-based)
- [x] Display all extracted fields with labels:
  - 論文標題, 年份, 作者, 研究目的, 資料集, 資料前處理, 建模, 評估指標
- [x] Display research classification (area, trunk)
- [x] Add "Confirm" button to finalize upload
- [x] Add "Cancel" button to abort upload
- [x] Implement confirmation handler (call final storage)
- [x] Implement cancellation handler (cleanup temp files)

**Validation**: Test preview display with sample extracted data.

---

### Task 2.5: Implement Error Display
- [x] Create error message component with styling
- [x] Implement `showUploadError()` function
- [x] Format validation errors (list missing fields)
- [x] Format processing errors (show error message)
- [x] Add error auto-dismiss after 10 seconds (or manual close)
- [x] Style errors for visibility (red background, icon)

**Validation**: Test with various error scenarios from backend.

---

### Task 2.6: Implement Upload Submission Flow
- [x] Create `submitPaperUpload()` JavaScript function
- [x] Gather FormData with selected PDF
- [x] Make POST request to `/upload_paper`
- [x] Handle response streaming for progress updates (if implemented)
- [x] Handle success response:
  - Show metadata preview
  - Enable confirmation
- [x] Handle error response:
  - Show error message
  - Reset form state
- [x] Implement form reset after successful upload

**Validation**: End-to-end test of upload flow with valid and invalid PDFs.

---

## Phase 3: Integration and Testing

### Task 3.1: Integration Testing
- [ ] Test complete upload flow with valid research papers
- [ ] Test validation rejection with invalid PDFs
- [ ] Test extraction with Chinese and English papers
- [ ] Test database updates and verify JSON structure
- [ ] Test duplicate paper handling
- [ ] Test new area/trunk creation
- [ ] Test atomic rollback on failures
- [ ] Test concurrent uploads (multiple users)
- [ ] Test file storage with special characters in titles

**Validation**: Create test checklist and verify all scenarios pass.

---

### Task 3.2: Error Recovery Testing
- [ ] Test behavior when LLM is unavailable
- [ ] Test behavior when disk is full
- [ ] Test behavior when JSON files are corrupted
- [ ] Test behavior when file permissions are wrong
- [ ] Test timeout handling for slow LLM responses
- [ ] Verify all errors result in proper rollback

**Validation**: Confirm system remains stable after errors.

---

### Task 3.3: Performance Testing
- [ ] Measure average upload processing time
- [ ] Test with large PDFs (near 50MB limit)
- [ ] Test with PDFs containing many pages (100+)
- [ ] Monitor LLM response times
- [ ] Identify bottlenecks and optimize if needed

**Validation**: Ensure reasonable response times (< 60 seconds total).

---

### Task 3.4: UI/UX Refinement
- [ ] Review UI with actual users
- [ ] Improve error messages based on feedback
- [ ] Enhance progress messages for clarity
- [ ] Adjust styling for better visual hierarchy
- [ ] Add helpful tooltips or instructions
- [ ] Ensure responsive design on different screen sizes

**Validation**: User acceptance testing.

---

## Phase 4: Documentation and Deployment

### Task 4.1: Documentation
- [ ] Update README.md with new upload tab feature
- [ ] Document upload requirements and limitations
- [ ] Add example workflow with screenshots
- [ ] Document LLM prompt templates for future tuning
- [ ] Document JSON schema expectations
- [ ] Create troubleshooting guide for common issues

**Validation**: Review documentation for completeness.

---

### Task 4.2: Configuration and Setup
- [ ] Verify Ollama llama3.1:8b is available
- [ ] Update requirements.txt if new dependencies added
- [ ] Create data/ directory if it doesn't exist
- [ ] Set appropriate file permissions for data/ directory
- [ ] Create backup directory for JSON backups (optional)

**Validation**: Fresh installation test on clean environment.

---

### Task 4.3: Deployment
- [ ] Test on production environment
- [ ] Create backup of current database files
- [ ] Deploy updated code
- [ ] Verify all functionality works in production
- [ ] Monitor logs for any issues
- [ ] Create rollback plan if issues arise

**Validation**: Production smoke test with sample upload.

---

## Dependencies and Parallelization

**Can be done in parallel:**
- Tasks 1.1, 1.2, 1.3, 1.4 (backend modules are independent)
- Tasks 2.1, 2.2 (frontend structure and file input)

**Sequential dependencies:**
- Task 1.5 depends on Tasks 1.1, 1.2, 1.3, 1.4
- Task 2.6 depends on Tasks 2.1, 2.2, 2.3, 2.4, 2.5
- Phase 3 depends on completion of Phases 1 and 2
- Phase 4 depends on completion of Phase 3

**Critical path:**
Phase 1 (Backend) → Phase 2 (Frontend) → Phase 3 (Testing) → Phase 4 (Deployment)

Estimated total time: 3-5 days for experienced developer
