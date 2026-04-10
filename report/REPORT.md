# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Minh Khang
**Nhóm:** Nhóm 06 - E402
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai văn bản có cosine similarity cao nghĩa là vector embedding của chúng trỏ về cùng một hướng trong không gian nhiều chiều, tức là chúng mang ý nghĩa ngữ nghĩa tương đồng nhau, bất kể độ dài hay từ ngữ cụ thể.

**Ví dụ HIGH similarity:**
- Sentence A: "Machine learning uses data to train models."
- Sentence B: "Deep learning is a subset of machine learning."
- Tại sao tương đồng: Cả hai đều nói về machine learning — cùng domain kỹ thuật, dùng nhiều từ khoá chung.

**Ví dụ LOW similarity:**
- Sentence A: "How to bake a chocolate cake."
- Sentence B: "The history of the Roman Empire."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (ẩm thực vs lịch sử), không có từ khoá hay khái niệm chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector, không bị ảnh hưởng bởi độ dài (magnitude) của vector — điều này quan trọng vì một đoạn văn dài và một câu ngắn cùng chủ đề sẽ có vector có magnitude khác nhau nhưng hướng gần nhau. Euclidean distance sẽ phạt sự khác biệt về độ dài, dẫn đến kết quả không phản ánh đúng sự tương đồng ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> Phép tính: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25 chunks` — tăng thêm 2 chunks. Overlap nhiều hơn giúp đảm bảo thông tin nằm ở ranh giới giữa hai chunk không bị mất, cải thiện khả năng retrieval cho các câu hỏi liên quan đến nội dung chuyển tiếp giữa các đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tài liệu kỹ thuật về RAG, vector store, và lập trình Python (Internal Knowledge Assistant)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì nó phản ánh đúng bài toán thực tế của lab — xây dựng knowledge assistant cho nội bộ tổ chức. Tài liệu có cấu trúc đa dạng (markdown có headers, plain text, tiếng Anh lẫn tiếng Việt) giúp thử nghiệm nhiều chiến lược chunking khác nhau. Domain cũng có ranh giới chủ đề rõ ràng, dễ thiết kế metadata filter và benchmark queries có gold answers cụ thể.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `rag_system_design.md` | data/ | 2,391 | category=architecture, lang=en |
| 2 | `vector_store_notes.md` | data/ | 2,123 | category=database, lang=en |
| 3 | `vi_retrieval_notes.md` | data/ | 2,177 | category=retrieval, lang=vi |
| 4 | `customer_support_playbook.txt` | data/ | 1,692 | category=support, lang=en |
| 5 | `python_intro.txt` | data/ | 1,944 | category=programming, lang=en |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | str | "architecture", "database", "support" | Filter theo chủ đề — tránh trả về chunk từ tài liệu không liên quan |
| lang | str | "en", "vi" | Filter theo ngôn ngữ — đặc biệt quan trọng khi query tiếng Việt |
| doc_id | str | "rag_design", "python_intro" | Trace nguồn chunk, dùng để delete hoặc re-index tài liệu cụ thể |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu `vector_store_notes.md` (2,123 chars, chunk_size=200):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| vector_store_notes.md | FixedSizeChunker (`fixed_size`) | 8 | 183.0 | Trung bình — đôi khi cắt giữa câu |
| vector_store_notes.md | SentenceChunker (`by_sentences`) | 5 | 221.6 | Tốt — giữ nguyên câu hoàn chỉnh |
| vector_store_notes.md | RecursiveChunker (`recursive`) | 15 | 72.4 | Tốt — tôn trọng cấu trúc đoạn, nhưng chunk ngắn |

### Strategy Của Tôi

**Loại:** SentenceChunker (max_sentences_per_chunk=3)

**Mô tả cách hoạt động:**
> SentenceChunker dùng regex `(?<=[.!?])\s+|(?<=\.)\n` để tách văn bản thành các câu riêng lẻ dựa trên dấu kết thúc câu (`.`, `!`, `?`) theo sau bởi khoảng trắng hoặc newline. Sau đó gom nhóm theo `max_sentences_per_chunk=3` câu mỗi chunk bằng `range(0, len, step)`. Mỗi chunk là một đơn vị ngữ nghĩa hoàn chỉnh vì không bao giờ cắt giữa câu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu kỹ thuật (RAG design, vector store notes) thường được viết theo dạng định nghĩa và giải thích — mỗi câu mang một khái niệm độc lập. Cắt theo câu giúp mỗi chunk chứa đủ ngữ cảnh để trả lời câu hỏi mà không bị mất thông tin ở ranh giới. Với max_sentences_per_chunk=3, mỗi chunk đủ dài để có ngữ cảnh nhưng không quá dài gây nhiễu.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality (Q1 score) |
|-----------|----------|-------------|------------|------------------------------|
| 5 docs | RecursiveChunker(300) — baseline tốt nhất | 74 | ~72 | 0.2937 |
| 5 docs | **SentenceChunker(3) — của tôi** | 27 | ~180 | 0.2722 |
| 5 docs | FixedSize(300, overlap=50) | 41 | ~250 | 0.3104 |
| 5 docs | FixedSize(500, overlap=100) | 25 | ~420 | 0.1447 |

### So Sánh Với Thành Viên Khác

Kết quả benchmark 5 queries trên cùng bộ tài liệu (dùng `search_with_filter`, mock embedder):

| Thành viên | Strategy | Total Chunks | Q1 | Q2 | Q3 | Q4 | Q5 | Tổng /10 |
|-----------|----------|-------------|-----|-----|-----|-----|-----|---------|
| Khang (tôi) | SentenceChunker(3) | 27 | 1đ | 1đ | 1đ | 1đ | 1đ | 5/10 |
| [TV2] | FixedSize(300, overlap=50) | 41 | 2đ | 2đ | 2đ | 2đ | 2đ | 10/10 |
| [TV3] | RecursiveChunker(300) | 74 | 2đ | 1đ | 2đ | 2đ | 2đ | 9/10 |
| [TV4] | FixedSize(500, overlap=100) | 25 | 1đ | 1đ | 1đ | 1đ | 1đ | 5/10 |

> Lưu ý: Scores thấp do dùng MockEmbedder (hash-based). Với real embedder, tất cả strategies sẽ cho scores cao hơn và ranking sẽ chính xác hơn.

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `FixedSize(300, overlap=50)` cho kết quả tốt nhất trên bộ tài liệu này vì chunk_size=300 vừa đủ để chứa một ý hoàn chỉnh trong tài liệu kỹ thuật, trong khi overlap=50 giúp không mất thông tin ở ranh giới. RecursiveChunker(300) cũng tốt nhưng tạo ra quá nhiều chunk nhỏ (74 chunks), làm tăng noise. SentenceChunker tạo chunk có ngữ nghĩa tốt nhưng chunk dài hơn có thể chứa nhiều ý không liên quan đến query.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)` để tách câu dựa trên dấu kết thúc câu (`.`, `!`, `?`) theo sau bởi khoảng trắng hoặc newline. Sau khi tách, lọc bỏ chuỗi rỗng và strip whitespace, rồi gom nhóm theo `max_sentences_per_chunk` dùng `range(0, len, step)`. Edge case: `max_sentences_per_chunk` được clamp về tối thiểu 1 trong `__init__`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy: nếu text đã ngắn hơn `chunk_size` thì trả về ngay (base case). Nếu không, thử separator đầu tiên trong danh sách — split text theo separator đó, rồi với mỗi phần: nếu đủ ngắn thì giữ nguyên, nếu vẫn dài thì đệ quy với danh sách separator còn lại. Separator rỗng `""` là fallback cuối cùng, cắt theo ký tự. Edge case: nếu `separators=[]` thì trả về `[text]` ngay.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed bằng `_embedding_fn` và lưu dưới dạng dict `{id, content, embedding, metadata}` vào `self._store`. Khi search, embed query rồi tính cosine similarity với từng record qua `compute_similarity`, sort descending, trả top_k kết quả kèm score. Dùng `compute_similarity` từ `chunking.py` thay vì tự tính lại để tránh duplicate code.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter trước bằng list comprehension — chỉ giữ records có metadata khớp tất cả key-value trong `metadata_filter`, sau đó gọi `_search_records` trên tập đã lọc. `delete_document` so sánh `metadata['doc_id']` với `doc_id` cần xóa, rebuild `self._store` bằng list comprehension loại bỏ matches, trả `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top_k chunks từ store bằng `self.store.search(question, top_k)`, ghép nội dung các chunk thành `context` bằng `"\n".join`. Build prompt theo format `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"` — format này rõ ràng phân tách context và question, giúp LLM hiểu đúng vai trò của từng phần. Kết quả là output của `self.llm_fn(prompt)`.

### Test Results

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.08s ==============================
```

**Số tests pass: 42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Dùng `_mock_embed` (MockEmbedder, dim=64) — đây là hash-based embedder nên scores không phản ánh ngữ nghĩa thực sự như real embedder.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "The cat sat on the mat." | "A feline rested on the rug." | high | -0.0506 | Không |
| 2 | "Python is a programming language." | "I love eating pizza." | low | -0.0166 | Gần đúng |
| 3 | "Machine learning uses data to train models." | "Deep learning is a subset of machine learning." | high | -0.0035 | Không |
| 4 | "The stock market crashed today." | "Investors lost billions in the financial collapse." | high | 0.0516 | Gần đúng |
| 5 | "How to bake a chocolate cake." | "The history of the Roman Empire." | low | 0.0085 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là pair 1 — "cat/mat" và "feline/rug" có score âm dù về ngữ nghĩa rất tương đồng. Điều này cho thấy `MockEmbedder` dùng MD5 hash để tạo vector ngẫu nhiên, không nắm bắt được ngữ nghĩa thực sự. Với real embedder (như `all-MiniLM-L6-v2`), pair 1 và 3 sẽ có score cao vì model được train để hiểu quan hệ ngữ nghĩa giữa các từ đồng nghĩa và khái niệm liên quan.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries với `SentenceChunker(max_sentences_per_chunk=3)` trên 5 tài liệu trong `data/`, dùng `search_with_filter` với metadata filter.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is a vector store and what is it used for? | Stores embeddings, retrieves most similar items; used for semantic search, RAG, recommendation |
| 2 | How does RAG combine retrieval with generation? | Retrieves relevant chunks, injects as context into prompt, calls LLM to generate grounded answer |
| 3 | What are common use cases for Python in production? | APIs, data pipelines, internal tools, model-serving; FastAPI, Django, Flask |
| 4 | What should a customer support playbook include? | Issue descriptions, step-by-step resolutions, escalation criteria, references to owning team |
| 5 | Retrieval đóng vai trò gì trong trợ lý tri thức nội bộ? | Tìm đoạn tài liệu phù hợp nhất trước khi LLM tạo câu trả lời, đảm bảo bám sát nguồn dữ liệu |

### Kết Quả Của Tôi (SentenceChunker, 27 chunks)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Filter dùng |
|---|-------|--------------------------------|-------|-----------|------------|
| 1 | What is a vector store...? | "That is why teams should test retrieval quality with realistic queries..." | 0.2722 | Gần đúng — đề cập retrieval quality, không phải định nghĩa trực tiếp | `{"category": "database"}` |
| 2 | How does RAG combine...? | "The assistant should reduce hallucinations by grounding its responses..." | 0.0829 | Có — đề cập grounding, nhưng không giải thích cơ chế RAG đầy đủ | `{"category": "architecture"}` |
| 3 | Python in production? | "Python is a high-level programming language widely used for automation..." | 0.0125 | Có — đúng tài liệu, score thấp do mock embedder | `{"category": "programming"}` |
| 4 | Customer support playbook? | "A high-quality support assistant should also recognize when retrieval is insufficient..." | 0.0826 | Gần đúng — về support assistant, không phải nội dung playbook cụ thể | `{"category": "support"}` |
| 5 | Retrieval đóng vai trò gì...? | "Ví dụ, một công ty có thể gắn nhãn tài liệu theo phòng ban, ngôn ngữ..." | 0.0421 | Gần đúng — cùng tài liệu tiếng Việt nhưng không phải đoạn định nghĩa vai trò | `{"lang": "vi"}` |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

**Nhận xét:** SentenceChunker tạo chunk dài (~180 chars trung bình), mỗi chunk chứa nhiều ý — điều này làm embedding bị "pha loãng" và score thấp hơn so với FixedSize(300) hay Recursive(300). Với mock embedder, vấn đề càng rõ hơn vì embedding không nắm ngữ nghĩa thực sự.

---

## 7. What I Learned (5 điểm — Demo)

### Failure Analysis (Exercise 3.5)

**Query thất bại:** Query 5 — *"Retrieval đóng vai trò gì trong trợ lý tri thức nội bộ?"*

**Kết quả thực tế:** Top-1 trả về chunk *"Ví dụ, một công ty có thể gắn nhãn tài liệu theo phòng ban..."* (score 0.0421) — đây là ví dụ về metadata, không phải định nghĩa vai trò của retrieval.

**Nguyên nhân:**
- **Chunk coherence kém:** SentenceChunker gom 3 câu/chunk, câu định nghĩa vai trò retrieval bị gộp chung với các câu ví dụ không liên quan → embedding bị pha loãng, không đại diện đúng cho ý chính.
- **Mock embedder không hiểu ngữ nghĩa:** Hash-based embedding không nhận ra "retrieval đóng vai trò" và "retrieval tìm ra những đoạn tài liệu phù hợp" là cùng ý.
- **Filter `{"lang": "vi"}` quá rộng:** Lọc được đúng tài liệu nhưng không phân biệt được đoạn định nghĩa vs đoạn ví dụ trong cùng tài liệu.

**Đề xuất cải thiện:**
- Dùng `SentenceChunker(max_sentences_per_chunk=1)` hoặc `RecursiveChunker` để tách câu định nghĩa ra riêng, tránh gộp với ví dụ.
- Thêm metadata `section` (ví dụ: "definition", "example", "conclusion") để filter chính xác hơn.
- Dùng real embedder (all-MiniLM-L6-v2) — model này được train để hiểu paraphrase, sẽ match đúng câu hỏi với đoạn định nghĩa.

---

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Một thành viên dùng FixedSizeChunker với overlap lớn (100 ký tự) và cho thấy overlap giúp cải thiện retrieval cho các câu hỏi liên quan đến thông tin nằm ở ranh giới chunk. Điều này cho thấy tham số overlap quan trọng không kém gì chunk_size, và cần được tune theo đặc điểm của domain.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác dùng metadata filtering theo `date` để ưu tiên tài liệu mới nhất khi trả lời câu hỏi về thông tin cập nhật. Đây là cách dùng metadata rất thực tế — không chỉ filter theo category mà còn theo thời gian, giúp tránh trả lời dựa trên thông tin lỗi thời.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ dùng real embedder (all-MiniLM-L6-v2) thay vì MockEmbedder để có kết quả retrieval phản ánh đúng ngữ nghĩa. Ngoài ra, tôi sẽ thêm metadata `section` để phân biệt các phần trong cùng một tài liệu (ví dụ: "definition", "example", "summary"), giúp filter chính xác hơn khi query yêu cầu loại thông tin cụ thể.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
