# HotFlip Gold Context trước Contriever retrieval

Pipeline này được viết lại từ hai notebook trong thư mục và giữ nguyên:

- retriever: `facebook/contriever`;
- generator: `Qwen/Qwen2.5-7B-Instruct`;
- dữ liệu: `hotpot_qa`, cấu hình `distractor`, split `validation`.

## Chạy baseline trên Google Colab trước

Mở [colab_baseline.ipynb](colab_baseline.ipynb) bằng Google Colab, chọn
`Runtime > Change runtime type > T4 GPU`, rồi chạy lần lượt các cell.

Baseline chưa dùng HotFlip:

```text
question
  -> facebook/contriever xếp hạng 10 HotpotQA documents
  -> lấy top-2 passages
  -> Qwen/Qwen2.5-7B-Instruct (4-bit trên Colab)
  -> câu trả lời và báo cáo xác suất
```

Có thể chạy tương đương bằng lệnh:

```powershell
python -m hotflip_rag.baseline `
  --split validation `
  --num-examples 100 `
  --top-k 2 `
  --load-in-4bit `
  --output-dir outputs/colab_baseline
```

Mỗi mẫu in câu hỏi, đáp án đúng, đáp án LLM, từng context Contriever lấy,
cosine score, EM/F1, xác suất cả chuỗi đáp án chuẩn và geometric-mean token
probability. Báo cáo cuối in retrieval recall, Exact Match Accuracy và Average
F1. Kết quả cũng được lưu thành CSV, JSONL và aggregate JSON.

Khác với yêu cầu “fixed retrieved context” trong bản mô tả đính kèm, pipeline
này làm đúng thứ tự được yêu cầu sau cùng:

```text
HotpotQA question
  -> tạo Gold Context và distractors
  -> HotFlip chỉ các token trong Gold Context
  -> Contriever truy xuất trên attacked Gold Context + distractors
  -> Qwen sinh câu trả lời
```

Query và distractors không bị sửa. HotFlip hiện chỉ thực hiện **token
replacement**, không chèn hoặc xóa token.

## Mục tiêu

Tất cả chiến lược tìm kiếm đều tối đa hóa một objective.

- Untargeted:

  ```text
  J = -cos(Contriever(question), Contriever(gold_context))
  ```

  Tối đa hóa `J` làm Gold Context rời xa câu hỏi và có khả năng mất vị trí
  retrieval.

- Targeted:

  ```text
  J = cos(question, context) + λ cos(target_answer, context)
  ```

  Thành phần đầu giữ context bị sửa còn liên quan tới câu hỏi; thành phần sau
  kéo biểu diễn của nó về phía đáp án đích. Targeted success chỉ được tính khi
  câu trả lời của generator khớp hoặc chứa đáp án đích đã chuẩn hóa.

Ở vị trí `i`, thay token `a` bằng `b` được xấp xỉ theo HotFlip:

```text
ΔJ ≈ ∇e_i J · (E_b - E_a)
```

Gradient được lấy theo input embeddings vì ID token là số nguyên và không thể
đạo hàm. Các candidate tốt nhất theo xấp xỉ được đánh giá lại bằng forward pass
thật khi bật `--exact-rerank`.

## Chạy

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

Untargeted, greedy, thay tối đa 3 token:

```powershell
python -m hotflip_rag.attack `
  --num-examples 100 `
  --attack-mode untargeted `
  --search-strategy greedy `
  --max-token-changes 3 `
  --hotflip-top-k 20 `
  --exact-rerank `
  --output-dir outputs/hotflip_untargeted
```

Targeted dùng danh sách đáp án giả hiện có:

```powershell
python -m hotflip_rag.attack `
  --num-examples 100 `
  --attack-mode targeted `
  --target-answer-file hotpotqa_answers_300.pkl `
  --target-weight 1.0 `
  --search-strategy beam `
  --beam-width 3 `
  --max-token-changes 3 `
  --hotflip-top-k 20 `
  --exact-rerank `
  --output-dir outputs/hotflip_targeted
```

Để chạy nhanh một mẫu mà không lọc theo câu trả lời baseline:

```powershell
python -m hotflip_rag.attack --num-examples 1 --no-only-clean-correct
```

## Kết quả

Mỗi thư mục kết quả có:

- `results.jsonl`: đầy đủ context trước/sau, retrieval trước/sau, token flips,
  approximate score, exact objective và câu trả lời;
- `summary.csv`: bảng dễ đọc;
- `aggregate_metrics.json`: ASR, retrieval rate và số token sửa trung bình;
- `config.json`: cấu hình, model, thiết bị và phiên bản môi trường;
- `failures.json`: mẫu bị bỏ qua hoặc lỗi.

## Kiểm thử

Các test dùng mô hình nhỏ cục bộ, không tải model:

```powershell
python -m unittest discover -s tests -v
```

## Giới hạn

- Đây là tấn công white-box vào **retrieval representation** của Gold Context,
  không phải objective NLL của generator trong mô tả “fixed context”.
- Targeted retrieval objective là một phép thích nghi từ HotFlip; nó không đảm
  bảo câu văn tự nhiên hoặc bảo toàn ngữ nghĩa.
- Candidate vocabulary mặc định giới hạn ở 5.000 token ID đầu để giảm chi phí;
  đây không phải thống kê tần suất corpus.
- Chưa căn chỉnh riêng supporting-fact span ở cấp token; toàn bộ Gold Context
  tạo từ các tài liệu chứa supporting facts đều có thể bị sửa.
- Chạy Qwen 7B và exact reranking cần GPU có VRAM phù hợp.

Đây là pipeline nghiên cứu robustness trong môi trường white-box có kiểm soát.
