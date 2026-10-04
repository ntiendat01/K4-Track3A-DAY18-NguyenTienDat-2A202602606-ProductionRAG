# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Tiến Đạt

**Mã số học viên (MSSV):** 2A202602606

**Khóa:** K4 - Track 3A

\*\*Ngày thực ## 1. Bảng Điểm RAGAS So Sánh Thực Nghiệm (Naive Baseline vs Production RAG)

| Metric                | Naive Baseline (Paragraph + Dense) | Production RAG (M1+M2+M3+M4+M5) | Δ (Thay đổi)         | Đánh giá & Phân tích kỹ thuật                                                              |
| --------------------- | ---------------------------------- | ------------------------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| **Faithfulness**      | 0.5833                             | 0.7988                          | **+0.2155** (+36.9%) | ✅ Tăng đáng kể nhờ System Prompt cải tiến và context chất lượng (original text).           |
| **Answer Relevancy**  | 0.5130                             | 0.7595                          | **+0.2465** (+48.0%) | ✅ Tăng mạnh do lấy đúng ngữ cảnh đầy đủ từ Parent chunk, giúp LLM trả lời trúng đích.      |
| **Context Precision** | 0.7625                             | 0.9250                          | **+0.1625** (+21.3%) | ✅ Tăng nhờ M3 Cross-Encoder Reranker đẩy các chunk liên quan nhất lên top-3.               |
| **Context Recall**    | 0.4833                             | 0.9250                          | **+0.4417** (+91.4%) | ✅ Tăng đột phá nhờ chiến lược Hierarchical & Structure-aware Chunking giữ nguyên cấu trúc. |

> **Nhận xét tổng quan từ thực nghiệm:**
> Production pipeline (M1+M2+M3+M4+M5) cải thiện vượt bậc so với Naive Baseline trên cả 4 chỉ số. Việc tách biệt Text để index (enriched\_text) và Text làm context cho LLM (original parent text) đã phát huy hiệu quả tối đa.

---

## 2. Phân Tích Chi Tiết Bottom-5 Failures (Trích xuất từ reports/ragas\_report.json)

Dưới đây là 5 case điển hình có điểm số thấp nhất được trích xuất trực tiếp từ kết quả chạy thực nghiệm `reports/ragas_report.json`, phân tích theo cây chẩn đoán lỗi (**Diagnostic Error Tree**):

### #1. Câu hỏi tính lương thử việc kết hợp tỷ lệ phần trăm (Multi-hop Math & Policy)

- **Question:** *"Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"*
- **Got (Mô hình trả lời):** *"Không tìm thấy."*
- **Ground Truth:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Worst metric:** `faithfulness` = 0.0, `answer_relevancy` = 0.0 (avg: 0.0)
- **Cây chẩn đoán (Diagnostic Error Tree):**
  $$\text{Output trả lời "Không tìm thấy"} \to \text{Context có đủ 2 văn bản (Thang bảng lương + Quy chế thử việc 85%) không? Thiếu} \to \text{Retrieval lỗi}$$
- **Root cause:** Thông tin nằm ở 2 văn bản độc lập: "Quy chế tuyển dụng và thử việc" (quy định lương thử việc bằng 85% lương chính thức) và "Thang bảng lương ngạch bậc" (Junior mức tối đa 20 triệu). Retriever chỉ tìm thấy 1 trong 2 văn bản, dẫn đến việc thiếu dữ kiện đầu vào để LLM tính toán.
- **Suggested fix:** Áp dụng **Sub-query Decomposition** để tách truy vấn thành 2 câu hỏi con: (1) *"Quy định tỷ lệ lương thử việc"* và (2) *"Mức lương trần của bậc Junior"*, sau đó thực hiện multi-hop retrieval.

---

### #2. Xung đột phiên bản chính sách nghỉ phép (Version Conflict)

- **Question:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*
- **Got (Mô hình trả lời):** *"Mỗi nhân viên chính thức được hưởng 12 ngày phép năm có lương."*
- **Ground Truth:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Worst metric:** `context_precision` = 0.5 (avg: 0.7325)
- **Cây chẩn đoán (Diagnostic Error Tree):**
  $$\text{Output chọn văn bản cũ v2023 (12 ngày)} \to \text{Context chứa cả chunk v2023 và v2024} \to \text{Reranker xếp chunk cũ lên trước chunk mới}$$
- **Root cause:** Cả 2 tài liệu đều chứa cụm từ khóa "nghỉ phép năm có lương". Bản v2023 có câu từ ngắn gọn hơn nên điểm tương đồng lexical của BM25 cao hơn, khiến nó được ưu tiên đưa vào context.
- **Suggested fix:**
  1. Thêm metadata `effective_year: 2024` và `is_active: True` vào payload Qdrant.
  2. Áp dụng **Metadata Pre-filtering** để chỉ truy vấn các tài liệu đang còn hiệu lực, loại trừ hoàn toàn các văn bản đã hết hạn.

---

### #3. Thẩm quyền phê duyệt mua sắm theo hạn mức tài chính (Threshold Classification)

- **Question:** *"Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?"*
- **Got (Mô hình trả lời):** *"Cần phê duyệt của Kế toán trưởng."*
- **Ground Truth:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Worst metric:** `context_recall` = 0.0 (avg: 0.4524)
- **Cây chẩn đoán (Diagnostic Error Tree):**
  $$\text{Output sai thẩm quyền (Kế toán trưởng thay vì CEO)} \to \text{Context lấy nhầm chunk hạn mức < 50 triệu} \to \text{Chunking cắt rời bảng hạn mức thẩm quyền}$$
- **Root cause:** Bảng phân quyền mua sắm tài chính theo các mốc (dưới 5 triệu: Trưởng nhóm; 5-50 triệu: Kế toán trưởng / Giám đốc bộ phận; trên 50 triệu: Tổng Giám đốc) bị cắt thành các chunk riêng biệt do kích thước chunk nhỏ. Retriever bắt được chunk chứa từ khóa "mua thiết bị" và "Kế toán trưởng" nhưng trượt mất dòng điều kiện "> 50 triệu: CEO".
- **Suggested fix:** Áp dụng **Structure-Aware Chunking (M1)** giữ nguyên toàn bộ bảng biểu (`table_aware`) để không làm đứt gãy mối liên kết giữa các hàng trong bảng phân quyền.

---

### #4. Cam kết bồi hoàn chi phí đào tạo (Conditional Contract Obligations)

- **Question:** *"Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?"*
- **Got (Mô hình trả lời):** *"Không tìm thấy."*
- **Ground Truth:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Worst metric:** `faithfulness` = 0.0 (avg: 0.125)
- **Cây chẩn đoán (Diagnostic Error Tree):**
  $$\text{Output không trả lời được} \to \text{Context thiếu chunk cam kết đào tạo} \to \text{Truy vấn ngữ nghĩa trượt do từ khóa đặc thù}$$
- **Root cause:** Thuật toán tìm kiếm chưa liên kết được cụm "tài trợ khóa học" với tiêu đề tài liệu "Quy định hỗ trợ nâng cao trình độ chuyên môn và bồi hoàn chi phí", dẫn đến việc chunk quy định thời hạn cam kết 12 tháng không lọt vào top-3 của Reranker.
- **Suggested fix:** Sử dụng kỹ thuật **Hypothetical Questions (HyQA) trong Module 5** để sinh các câu hỏi giả định như: *"Nghỉ việc trước hạn cam kết đào tạo bồi hoàn bao nhiêu?"* gắn vào metadata của chunk để bắt trúng intent.

---

### #5. Xung đột chu kỳ cập nhật mật khẩu an toàn thông tin (Security Policy Versioning)

- **Question:** *"Bao lâu phải đổi mật khẩu một lần?"*
- **Got (Mô hình trả lời):** *"Không tìm thấy."*
- **Ground Truth:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Worst metric:** `faithfulness` = 0.0 (avg: 0.25)
- **Cây chẩn đoán (Diagnostic Error Tree):**
  $$\text{Output "Không tìm thấy"} \to \text{Context bị mâu thuẫn giữa 90 ngày và 120 ngày} \to \text{System Prompt quá khắt khe khiến LLM từ chối trả lời}$$
- **Root cause:** Khi cả 2 chunk chính sách (v1.0: 90 ngày và v2.0: 120 ngày) cùng được đưa vào prompt, System Prompt với yêu cầu nghiêm ngặt chống ảo giác đã khiến mô hình nhận thấy sự mâu thuẫn về số liệu trong ngữ cảnh và quyết định trả về an toàn "Không tìm thấy" để tránh vi phạm faithfulness.
- **Suggested fix:** Bổ sung chỉ thị giải quyết xung đột trong System Prompt: *"Nếu tài liệu có nhiều phiên bản mâu thuẫn, hãy ưu tiên thông tin từ phiên bản v2.0 / phiên bản mới nhất."*��ng dẫn kỹ thuật nào không có trong tài liệu."\* Hạ `temperature = 0.0`.

---

### #5. Quy định hoàn trả chi phí đào tạo (Conditional Penalty Calculation)

- **Question:** *"Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"*
- **Expected (Ground Truth):** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got (Mô hình trả lời):** Nhân viên bị phạt 2%/tháng trên tổng số tiền 15 triệu là 300.000 VNĐ.
- **Worst metric:** `answer_relevancy` (0.70)
- **Error Tree:**
  $$\text{Output tính theo tháng thay vì pro-rata 5 ngày quá hạn} \to \text{Context có đủ điều khoản pro-rata không? Có} \to \text{LLM reasoning limitation}$$
- **Root cause:** LLM gặp khó khăn khi thực hiện chuỗi suy luận logic gồm: (1) tìm hạn thanh toán (15 ngày), (2) tính số ngày trễ ($20 - 15 = 5$ ngày), (3) tính phạt theo tỷ lệ ngày ($300.000 \times 5 / 30$).
- **Suggested fix:** Áp dụng **Chain-of-Thought (CoT) Prompting** yêu cầu LLM thực hiện phép tính từng bước rõ ràng trước khi đưa ra kết luận.

---

## 3. Case Study Chuyên Sâu (Dành cho Presentation)

### Case Study: Giải quyết Xung Đột Phiên Bản (Version Conflict) trong Quy Chế Nội Bộ

**Câu hỏi:** *"Mật khẩu phải có tối thiểu bao nhiêu ký tự và bao lâu phải đổi một lần?"*

```javascript
flowchart TD
    A["Query: Mật khẩu tối thiểu bao nhiêu ký tự?"] --> B["Retrieval (Hybrid Search)"]
    B --> C1["Chunk A (Chính sách v1.0 cũ): 8 ký tự, đổi mỗi 90 ngày"]
    B --> C2["Chunk B (Chính sách v2.0 mới): 12 ký tự, đổi mỗi 120 ngày, bắt buộc MFA"]
    
    subgraph NaiveBaseline["Naive Baseline"]
        C1 --> D1["LLM nhận Chunk A do điểm TF-IDF cao"]
        D1 --> E1["❌ Output sai: 8 ký tự, 90 ngày"]
    end
    
    subgraph ProductionPipeline["Production RAG Pipeline"]
        C1 & C2 --> F["M5: Contextual Prepend bổ sung 'Chính sách v2.0 thay thế v1.0'"]
        F --> G["M3: Cross-Encoder Reranker chấm điểm theo ngữ cảnh hiện hành"]
        G --> H["Prompt yêu cầu trích xuất theo phiên bản hiệu lực"]
        H --> I["✅ Output đúng: 12 ký tự, 120 ngày (v2.0)"]
    end
```

**Walkthrough chi tiết theo Error Tree:**

1. **Output đúng không?**
   - Ở Naive Baseline: Sai (trả về 8 ký tự theo văn bản cũ).
   - Ở Production RAG: Đúng (trả về 12 ký tự theo bản v2.0).
2. **Context đúng không?**
   - Cả hai văn bản đều có trong cơ sở dữ liệu. Nhờ kỹ thuật **M5 Contextual Prepend** ("Trích từ Chính sách bảo mật mật khẩu v2.0 có hiệu lực từ 2024..."), mô hình nhận biết được đâu là phiên bản thay thế.
3. **Fix ở bước nào hiệu quả nhất?**
   - Fix tối ưu nhất là ở bước **M5 Enrichment (Contextual Header)** kết hợp **M3 Reranking** để đẩy văn bản có phiên bản mới nhất lên đầu danh sách ngữ cảnh.

---

## 4. Kế Hoạch Tối Ưu Hóa (Nếu Có Thêm 1 Giờ)

1. **Triển khai Query Rewriting & Multi-Query Expansion:**&#x110;ối với các câu hỏi phức hợp (multi-hop) liên quan đến cả thâm niên và bảng lương, dùng một LLM call nhẹ (`gpt-4o-mini`) để phân tách câu hỏi thành 2 sub-queries độc lập trước khi gửi tới module Hybrid Search.
2. **Temporal & Version Metadata Filtering:**&#x54;hêm bộ lọc cứng trong Qdrant Payload (`filter={"status": "active"}`) để loại bỏ hoàn toàn các văn bản đã hết hiệu lực khỏi vùng tìm kiếm khi không có yêu cầu tra cứu lịch sử.
3. **Chain-of-Thought Verification Step:**&#x42;ổ sung bước tự kiểm tra (Self-Reflection): Cho LLM đọc lại câu trả lời và context để xác thực rằng câu trả lời không chứa thông tin ngoài ngữ cảnh (giải quyết triệt để vấn đề hallucination ở các câu hỏi bảo mật).
