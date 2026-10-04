# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Tiến Đạt

**Mã số học viên (MSSV):** 2A202602606

**Khóa:** K4 - Track 3A

**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu giữa các khái niệm lý thuyết cốt lõi trong bài giảng và việc hiện thực hóa trong mã nguồn:

| Lecture Concept                              | Module | Hàm cụ thể                                  | Observation & Phân tích                                                                                                                                                                                     |
| -------------------------------------------- | ------ | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Semantic Chunking**                        | M1     | `chunk_semantic()`                          | Dùng cosine similarity giữa các câu liên tiếp qua `all-MiniLM-L6-v2`. Giúp nhóm thông tin chung chủ đề, giảm đứt gãy thông tin hơn so với Basic Chunking.                                                   |
| **Hierarchical Chunking**                    | M1     | `chunk_hierarchical()`                      | Phân lớp Parent/Child. Index child chunk để tăng precision, nhưng trả về parent chunk để đưa ngữ cảnh trọn vẹn cho LLM. Đây là nguyên nhân chính giúp Recall tăng +91.4% so với baseline.                   |
| **Structure-Aware Chunking**                 | M1     | `chunk_structure_aware()`                   | Parse heading Markdown (`#`, `##`) giữ vẹn toàn bảng biểu. Rất hiệu quả cho các file chính sách và bảng lương.                                                                                              |
| **Vietnamese Word Tokenization**             | M2     | `segment_vietnamese()`                      | `underthesea` kết hợp `.replace("_", " ")`. Bắt buộc phải có để BM25 hoạt động tốt với truy vấn tiếng Việt.                                                                                                 |
| **BM25 + Dense Fusion (RRF)**                | M2     | `reciprocal_rank_fusion()`                  | Kết hợp keyword-based và semantic-based giúp khắc phục lỗi trượt từ khóa đặc thù (như tên dự án, mức tiền).                                                                                                 |
| **Cross-Encoder Reranking**                  | M3     | `CrossEncoderReranker.rerank()`             | Kéo Context Precision lên 0.9250 nhờ lọc được các chunk nhiễu. Cross-Encoder tính toán cross-attention toàn diện giữa `(query, document)` rất hiệu quả.                                                     |
| **RAGAS 4 Metrics**                          | M4     | `evaluate_ragas()`                          | Đo lường hiệu quả cụ thể: Production cải thiện trung bình +36% -> +91% so với Naive Baseline nhờ các module M1-M5, đặc biệt là recall và precision.                                                         |
| **Contextual Prepend & Combined Enrichment** | M5     | `enrich_chunks()` / `_enrich_single_call()` | Tạo meta-context (summary, questions) để index giúp Dense Search tốt hơn. Bài học lớn: chỉ dùng enriched text cho tìm kiếm, còn đưa text gốc cho LLM để tránh ảo giác (giúp Faithfulness duy trì ở 0.7988). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

1. **Lỗi không tương thích phiên bản PyTorch và Transformers trên macOS x86\_64:**
   - *Lỗi kỹ thuật (Exact error message):*
   - *Nguyên nhân & Cách debug:* Trên kiến trúc Intel Mac (x86\_64), PyTorch bản mới nhất hiện có trên PyPI là 2.2.2, trong khi `transformers 5.18.0` yêu cầu `torch >= 2.5`. Khi PyTorch bị disable, transformers không import được `torch.nn`.
   - *Cách giải quyết:* Hạ phiên bản `transformers` về `<4.46` (`transformers==4.45.2`) và cài đặt `sentence-transformers<4.0` (`3.4.1`) để đồng bộ hoàn toàn với `torch 2.2.2`.
2. **Lỗi tokenize tiếng Việt với BM25:**
   - *Hiện tượng:* BM25 search không tìm thấy văn bản chứa "nghỉ phép" dù trong tài liệu có từ này.
   - *Nguyên nhân:* `underthesea` mặc định nối từ ghép bằng dấu gạch dưới (`nghỉ_phép`), trong khi BM25Okapi tách từ theo khoảng trắng `split(" ")`. Khi câu hỏi của người dùng nhập "nghỉ phép" (2 token), BM25 không thể khớp với token `nghỉ_phép` (1 token duy nhất).
   - *Cách giải quyết:* Sau khi phân tách bằng `word_tokenize`, thực hiện thay thế `.replace("_", " ")` để đưa toàn bộ token về dạng chuẩn đồng nhất.
3. **Thời gian tải trọng số mô hình lớn qua mạng:**
   - *Hiện tượng:* Các mô hình embedding và reranking đa ngôn ngữ (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) có dung lượng từ 1.1GB đến 2.2GB, dễ bị timeout khi tải qua mạng thông thường hoặc khi hf-xet bị nghẽn.
   - *Cách giải quyết:* Cài đặt `hf-transfer` để tăng tốc độ tải đa luồng, đồng thời thiết kế cơ chế graceful fallback trong Module 3 (fallback sang mô hình semantic gọn nhẹ hoặc Flashrank khi weights chưa sẵn sàng) để đảm bảo pipeline và test case luôn chạy thông suốt.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý AI Tra cứu Quy chế Doanh nghiệp & Tài liệu Kỹ thuật

#### 1. Hiện trạng

- **Pipeline hiện tại:** Sử dụng Basic RAG đơn giản: chia đoạn theo độ dài cố định 500 ký tự (Fixed-length chunking) + lưu trữ trên FAISS + truy vấn Dense-only qua OpenAI `text-embedding-3-small`.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context fragmentation:* Các bảng biểu quy định chính sách (phụ cấp, hạn mức công tác, thang bảng lương) bị cắt vụn giữa các chunk khiến LLM trả lời sai số liệu.
  - *Low precision:* Truy vấn dense-only dễ bị nhiễu bởi các đoạn có ngữ nghĩa tương tự nhưng không chứa đúng điều khoản cần tìm (thiếu từ khóa số hiệu điều, khoản).
  - *Lack of evaluation:* Chưa có bộ metric đánh giá định lượng, chỉ kiểm tra thủ công bằng mắt.

#### 2. Kế hoạch cải tiến từ kiến thức Lab 18

1. **Chunking strategy:** Áp dụng **Hierarchical Chunking** (Parent 2048 chars, Child 256 chars) kết hợp **Structure-aware Chunking** đối với tài liệu dạng Markdown/PDF hành chính để giữ nguyên vẹn cấu trúc các Điều, Khoản và Bảng biểu.
2. **Search retrieval:** Xây dựng **Hybrid Search** kết hợp BM25 (đã qua xử lý `underthesea`) để bắt trúng các mã quy định, số hiệu văn bản và Dense Search (Qdrant) để hiểu ngữ nghĩa. Hợp nhất bằng thuật toán **RRF (Reciprocal Rank Fusion)** với $k=60$.
3. **Reranking:** Tích hợp mô hình Cross-Encoder (`BAAI/bge-reranker-v2-m3`) lọc top-20 candidate xuống top-3 chunk có độ tương đồng cao nhất trước khi đưa vào prompt của LLM.
4. **Enrichment:** Sử dụng kỹ thuật **Contextual Prepend** theo phong cách Anthropic (bổ sung 1 câu bối cảnh vị trí chunk trong văn bản) thông qua Combined single-call mode để tiết kiệm chi phí gọi API.
5. **Evaluation:** Thiết lập bộ 50 câu hỏi benchmark đa dạng và tự động đo lường định kỳ với **RAGAS 4 metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) trong CI/CD pipeline.

#### 3. Timeline triển khai

- **Tuần 1:** Tái cấu trúc cơ chế Chunking sang Hierarchical và Structure-Aware; triển khai vector database Qdrant trên môi trường staging.
- **Tuần 2:** Xây dựng module Hybrid Search (BM25 + Dense Qdrant + RRF) và kiểm thử module Reranking với Cross-Encoder.
- **Tuần 3:** Tích hợp Contextual Prepend vào luồng tiền xử lý (indexing pipeline); thiết lập bộ test benchmark và báo cáo đánh giá tự động với RAGAS.
