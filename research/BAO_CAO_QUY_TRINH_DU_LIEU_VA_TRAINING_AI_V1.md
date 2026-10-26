# BÁO CÁO QUY TRÌNH SINH DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH AI PHỎNG VẤN V1

## 1. Thông tin chung

- Tên phiên bản dữ liệu: `weak-supervision-v1`.
- Số bộ dữ liệu hoàn chỉnh: 1.000 bộ.
- Số vòng sinh dữ liệu: 250 vòng, mỗi vòng tạo 4 bộ hoàn chỉnh.
- Số ngữ cảnh CV/JD độc lập: 240 context group.
- Số mô hình đã huấn luyện: 2 mô hình baseline.
- Nguồn nhãn: điểm đồng thuận do nhiều AI chấm chéo.
- Người thực hiện review thủ công: chủ đề tài, thực hiện ở giai đoạn tiếp theo.
- Chuyên gia tham gia gán nhãn: không có.

## 2. Bối cảnh và hướng đi hiện tại

Hướng cũ của dự án tập trung vào so sánh CV với Job Description (JD), tính mức độ phù
hợp, tìm kỹ năng còn thiếu và đề xuất lộ trình học tập. Hướng hiện tại chuyển trọng tâm
sang hệ thống phỏng vấn AI thích ứng theo thời gian thực.

Luồng nghiệp vụ mục tiêu:

```text
CV + JD
  -> trích xuất thông tin có cấu trúc
  -> nhiều AI sinh các câu hỏi ứng viên
  -> mô hình chọn câu hỏi phù hợp nhất
  -> ứng viên trả lời
  -> Speech-to-Text nếu trả lời bằng giọng nói
  -> mô hình chấm câu trả lời theo 7 tiêu chí
  -> cập nhật trạng thái phỏng vấn
  -> sinh câu hỏi tiếp theo
  -> Text-to-Speech nếu cần đọc câu hỏi bằng giọng nói
```

Trong phiên bản V1, dự án tập trung xây dựng phần dữ liệu và hai mô hình nền tảng:

1. Mô hình chọn câu hỏi tốt nhất trong các câu hỏi do nhiều AI sinh ra.
2. Mô hình chấm chất lượng câu trả lời của ứng viên theo 7 tiêu chí.

## 3. Phạm vi không sử dụng chuyên gia

Phiên bản hiện tại đã bỏ yêu cầu thu thập nhãn từ chuyên gia. Toàn bộ nhãn dùng để
training được sinh theo cơ chế AI chấm chéo.

Quy trình hiện tại không sử dụng:

- Nhận xét của chuyên gia tuyển dụng.
- Điểm đồng thuận chuyên gia, hay `Expert Consensus Reference Standard`.
- Mô hình hiệu chỉnh điểm AI theo điểm chuyên gia.
- Dữ liệu biểu cảm khuôn mặt để kết luận năng lực ứng viên.
- Điểm chuyên gia làm ground truth.

Chủ đề tài sẽ tự review dữ liệu sau. Phần review này được gọi là `owner manual review`,
tức kiểm tra thủ công bởi người thực hiện đề tài, không được mô tả là đánh giá chuyên gia.

Một số bảng, API hoặc script liên quan đến chuyên gia vẫn còn trong repository dưới dạng
mã legacy để tránh làm hỏng các chức năng cũ. Chúng không tham gia vào quá trình export,
chia dữ liệu, training hoặc đánh giá hai mô hình V1.

## 4. Kiến trúc tổng thể của quy trình dữ liệu

```text
Ngữ cảnh CV + JD + trạng thái phỏng vấn
  -> chọn 4 nhà cung cấp AI độc lập
  -> mỗi AI sinh 1 câu hỏi
  -> 3 AI còn lại chấm câu hỏi
  -> tổng hợp nhãn câu hỏi
  -> mỗi AI sinh 1 câu trả lời tổng hợp theo mức chất lượng 1-5
  -> 3 AI còn lại chấm câu trả lời
  -> tổng hợp nhãn câu trả lời
  -> lưu toàn bộ dữ liệu vào Supabase PostgreSQL
  -> export CSV
  -> kiểm tra chất lượng dữ liệu
  -> chia train/validation/test theo context
  -> huấn luyện 2 mô hình baseline
  -> đánh giá trên test set
  -> chủ đề tài review các mẫu bị cảnh báo
```

## 5. Các AI tham gia sinh và chấm dữ liệu

Các nhà cung cấp từng tham gia vào dữ liệu V1 gồm:

| Nhà cung cấp/model | Vai trò | Số mẫu đã sinh |
| --- | --- | ---: |
| Cloudflare Qwen3 30B | Sinh và chấm chéo | 231 |
| OpenTyphoon 2.5 30B | Sinh và chấm chéo | 225 |
| Ollama Cloud Gemma 4 31B | Sinh và chấm chéo | 222 |
| Inception Mercury 2.5 | Sinh và chấm chéo | 195 |
| Groq GPT-OSS 120B | Sinh và chấm chéo | 109 |
| Cohere Command A | Dữ liệu lịch sử hợp lệ | 9 |
| Hugging Face DeepSeek V3 | Dữ liệu lịch sử hợp lệ | 9 |

Cohere và Hugging Face đã bị loại khỏi pool hoạt động hiện tại, nhưng 18 mẫu đã hoàn
thành trước đó vẫn đáp ứng đủ quy tắc chấm chéo nên được giữ lại trong dataset.

### 5.1. Cơ chế nhiều API key cho cùng một AI

Một nhà cung cấp có thể có nhiều API key từ các tài khoản khác nhau. Các key này vẫn chỉ
được tính là một AI, không phải nhiều AI độc lập.

Ví dụ:

```text
Groq key 1 hết quota
  -> chuyển sang Groq key 2
  -> nếu key 2 hết thì dùng key 3
  -> khi toàn bộ key Groq không dùng được mới loại Groq khỏi pool hiện tại
```

Mục đích của nhiều key dự phòng là giúp quy trình không bị gián đoạn vì quota. Nó không
làm tăng số lượng model độc lập dùng để chấm một mẫu.

### 5.2. Cơ chế pool và failover

`Provider pool` là danh sách các nhà cung cấp AI đang có thể hoạt động. Mỗi vòng sinh dữ
liệu yêu cầu 4 provider khác nhau.

`Failover` nghĩa là khi một provider gặp lỗi quota, xác thực, mạng hoặc JSON không hợp lệ,
hệ thống tự loại provider đó khỏi vị trí đang chạy và đưa provider còn hoạt động khác vào.

Quy tắc dừng:

- Có từ 4 provider hoạt động trở lên: tiếp tục sinh dữ liệu.
- Chỉ còn 3 provider hoạt động: tạm dừng vì không đủ 4 AI độc lập.
- Dữ liệu của vòng đang lỗi được rollback để tránh lưu một vòng không cân bằng.

`Rollback` nghĩa là hủy toàn bộ thay đổi chưa hoàn chỉnh của một vòng. Nhờ đó không xảy
ra trường hợp chỉ lưu 1 hoặc 2 mẫu trong một vòng đáng lẽ phải có 4 mẫu.

## 6. Một bộ dữ liệu hoàn chỉnh gồm những gì?

Một `complete set`, hay một bộ hoàn chỉnh, gồm:

1. Một ngữ cảnh CV/JD và trạng thái phỏng vấn.
2. Một câu hỏi do một AI sinh.
3. Ba lượt chấm câu hỏi từ ba AI khác.
4. Một câu trả lời tổng hợp do AI sinh.
5. Ba lượt chấm câu trả lời từ ba AI khác.
6. Metadata về model, độ trễ, prompt version và trạng thái.
7. Nhãn đồng thuận cho câu hỏi và câu trả lời.

Một vòng tạo 4 bộ vì có 4 AI tham gia. Tổng cộng 250 vòng tạo ra:

```text
250 vòng x 4 bộ mỗi vòng = 1.000 bộ hoàn chỉnh
```

## 7. Ngữ cảnh đầu vào của quá trình sinh dữ liệu

Ngữ cảnh không chỉ có văn bản CV và JD thô. Hệ thống sử dụng dữ liệu đã được trích xuất:

- Ngôn ngữ phỏng vấn.
- Giai đoạn phỏng vấn hiện tại.
- Tiêu chí năng lực cần kiểm tra.
- Hồ sơ năng lực từ CV.
- Yêu cầu công việc từ JD.
- Lịch sử các câu hỏi và câu trả lời trước đó.

`Context` nghĩa là toàn bộ thông tin nền mà AI hoặc mô hình cần để hiểu ứng viên đang
ứng tuyển vị trí nào, đã được hỏi gì và câu hỏi tiếp theo cần kiểm tra năng lực nào.

## 8. Quy trình sinh câu hỏi

Mỗi AI nhận cùng một context trong một vòng và sinh một câu hỏi riêng. Câu hỏi phải:

- Bám sát CV và JD.
- Đúng giai đoạn phỏng vấn.
- Đúng tiêu chí đang cần đánh giá.
- Không lặp lại nội dung đã hỏi.
- Có khả năng làm ứng viên đưa ra bằng chứng cụ thể.
- Không chứa nội dung thiên vị hoặc không công bằng.

AI sinh câu hỏi không được tự chấm câu hỏi đó. Quy tắc này làm giảm nguy cơ model tự ưu
tiên phong cách câu hỏi của chính nó.

## 9. Rubric chấm câu hỏi

`Rubric` là bộ tiêu chí và hướng dẫn chấm điểm thống nhất. Mỗi tiêu chí có thang điểm từ
1 đến 5.

| Tiêu chí | Ý nghĩa tiếng Việt |
| --- | --- |
| `context_relevance` | Câu hỏi có phù hợp CV, JD và trạng thái phỏng vấn không |
| `technical_depth` | Câu hỏi có đủ chiều sâu chuyên môn không |
| `clarity` | Câu hỏi có rõ ràng, dễ hiểu và không mơ hồ không |
| `evidence_elicitation` | Câu hỏi có khuyến khích ứng viên đưa bằng chứng cụ thể không |
| `adaptivity` | Câu hỏi có thích ứng với thông tin và câu trả lời trước đó không |
| `non_redundancy` | Câu hỏi có tránh lặp lại nội dung đã hỏi không |
| `fairness` | Câu hỏi có công bằng và tránh thiên vị không |

Ba AI chấm độc lập. Với mỗi tiêu chí, hệ thống lấy trung vị của ba điểm làm pseudo-label.

Ví dụ:

```text
technical_depth do 3 AI chấm: 3, 4, 5
Trung vị: 4
Pseudo-label technical_depth: 4
```

Sau đó, `mean_quality_score` được tính bằng trung bình của 7 pseudo-label. Điểm này là
mục tiêu chính của mô hình chọn câu hỏi.

## 10. Quy trình sinh câu trả lời tổng hợp

Sau khi có câu hỏi, AI tạo một câu trả lời giả lập ứng viên. Mỗi mẫu được gán trước một
mức chất lượng mục tiêu từ 1 đến 5 để dataset có cả câu trả lời yếu, trung bình và tốt.

| Mức | Mô tả |
| ---: | --- |
| 1 | Sai, không liên quan hoặc không có bằng chứng |
| 2 | Có liên quan nhưng mơ hồ, nông hoặc có lỗi đáng kể |
| 3 | Cơ bản đúng nhưng thiếu chiều sâu hoặc thiếu bằng chứng |
| 4 | Đúng, có cấu trúc và có ví dụ hoặc lập luận tốt |
| 5 | Rất tốt, sâu, chính xác, có bằng chứng và kết quả đo lường |

Mỗi mức có đúng 200 mẫu, tạo ra phân bố cân bằng:

```text
Mức 1: 200
Mức 2: 200
Mức 3: 200
Mức 4: 200
Mức 5: 200
```

Mức chất lượng mục tiêu chỉ là metadata phục vụ phân tích. Nó không được đưa vào feature
training, vì nếu đưa vào thì model sẽ biết trước đáp án và gây `data leakage`.

## 11. Rubric chấm câu trả lời

| Tiêu chí | Ý nghĩa tiếng Việt |
| --- | --- |
| `answer_relevance` | Câu trả lời có đúng trọng tâm câu hỏi không |
| `technical_correctness` | Kiến thức và nội dung chuyên môn có chính xác không |
| `reasoning_depth` | Lập luận có chiều sâu, logic và giải thích được lựa chọn không |
| `evidence_specificity` | Có ví dụ, số liệu, hành động hoặc kết quả cụ thể không |
| `clarity` | Câu trả lời có rõ ràng và có cấu trúc không |
| `consistency` | Nội dung có nhất quán với CV, context và chính câu trả lời không |
| `criterion_score` | Mức độ thể hiện đúng năng lực đang được phỏng vấn |

Mỗi câu trả lời được ba AI khác chấm. Trung vị của từng tiêu chí được dùng làm 7 nhãn
đầu ra của Candidate Answer Scorer.

## 12. Giải thích Weak Supervision và pseudo-label

`Weak Supervision`, hay giám sát yếu, là cách tạo nhãn huấn luyện từ các nguồn không phải
ground truth tuyệt đối. Trong dự án này, các nguồn nhãn là nhiều AI độc lập.

`Pseudo-label`, hay nhãn giả lập, là nhãn được tạo tự động thay vì do chuyên gia xác nhận.
Pseudo-label vẫn dùng để training được, nhưng phải hiểu rằng model học cách mô phỏng sự
đồng thuận của nhóm AI, không phải mô phỏng một chuẩn chuyên gia khách quan.

Ưu điểm:

- Sinh được dữ liệu nhanh hơn.
- Không phụ thuộc vào việc tìm nhiều chuyên gia.
- Có thể mở rộng lên hàng nghìn mẫu.
- Giữ được cùng một rubric cho toàn bộ dataset.

Hạn chế:

- Các AI có thể cùng mắc một loại sai lệch.
- Điểm cao không bảo đảm câu hỏi thực sự tốt ngoài thực tế.
- Câu trả lời hiện tại là câu trả lời tổng hợp, chưa phản ánh đầy đủ cách người thật nói.
- Model được train trước hết để bắt chước AI jury.

`AI jury` nghĩa là nhóm các AI đóng vai trò hội đồng chấm điểm.

## 13. Cách tổng hợp nhãn đồng thuận

Hệ thống dùng `median`, tức trung vị, thay vì trung bình trực tiếp của ba AI cho từng tiêu
chí. Trung vị ít bị ảnh hưởng hơn nếu một AI chấm quá cao hoặc quá thấp.

Ví dụ:

```text
Điểm: 2, 4, 5
Trung bình: 3,67
Trung vị: 4
```

Sau khi có trung vị cho từng tiêu chí, hệ thống mới tính điểm chất lượng tổng hợp.

Một mẫu được đánh dấu `needs_human_review=true` khi:

- Khoảng cách giữa điểm thấp nhất và cao nhất của các AI từ 2 điểm trở lên; hoặc
- Quyết định consensus không phải `accept`.

Kết quả:

- 636/1.000 câu hỏi được đánh dấu cần review.
- 712/1.000 câu trả lời được đánh dấu cần review.
- Số mẫu đã được chủ đề tài review tại thời điểm training V1: 0.

Các mẫu bị đánh dấu vẫn được giữ trong baseline V1. Cờ này tạo danh sách ưu tiên để chủ
đề tài kiểm tra và tạo phiên bản dữ liệu V1.1 sau này.

## 14. Lưu trữ dữ liệu và cơ chế checkpoint

Dữ liệu được lưu trong PostgreSQL trên Supabase. Các nhóm bảng chính gồm:

- Benchmark case chứa snapshot context.
- Benchmark run chứa một lần chạy sinh dữ liệu.
- Question output chứa câu hỏi do AI sinh.
- Question AI judgment chứa điểm chấm câu hỏi.
- Synthetic answer sample chứa câu trả lời tổng hợp.
- Answer AI judgment chứa điểm chấm câu trả lời.

`Checkpoint` là file lưu tiến độ sinh dữ liệu. Nó ghi lại:

- Số bộ đã hoàn thành.
- Số vòng đã hoàn thành.
- Provider đang hoạt động.
- Provider đã lỗi.
- Token đã dùng.
- Lần failover gần nhất.
- Lỗi gần nhất.

Nhờ checkpoint, khi máy tắt, mạng lỗi hoặc API hết quota, quy trình có thể tiếp tục từ
vòng hoàn chỉnh gần nhất thay vì sinh lại từ đầu.

## 15. Các lỗi đã gặp và cách xử lý

### 15.1. API hết quota

Groq, Gemini và một số provider từng hết quota. Hệ thống chuyển key dự phòng hoặc
provider khác. Gemini sau đó được loại khỏi pool sinh dữ liệu hiện tại.

### 15.2. Lỗi DNS và mạng

Inception và Supabase từng gặp lỗi `getaddrinfo failed`, `network request failed` hoặc
`SSL SYSCALL EOF`. Hệ thống được bổ sung retry, failover và export theo batch.

### 15.3. Lỗi JSON không hợp lệ

Một số AI trả nội dung không phải JSON chuẩn. Inception được yêu cầu trả structured JSON
và provider có thể bị failover nếu liên tục trả JSON không hợp lệ.

### 15.4. Lỗi ghi checkpoint trên Windows

Windows từng báo `PermissionError` khi thay file tạm. Cơ chế lưu checkpoint được bổ sung
retry và phương án ghi trực tiếp dự phòng.

### 15.5. Export dữ liệu quá nặng

Ban đầu export tải quá nhiều trường như evidence, prompt và CV/JD text không cần thiết.
Truy vấn được tối ưu bằng cách:

- Chỉ tải các cột cần cho training.
- Chia thành batch 100 bản ghi.
- Retry riêng batch gặp lỗi.
- Không phải tải lại toàn bộ từ đầu.

## 16. Kiểm tra chất lượng dữ liệu trước training

| Kiểm tra | Câu hỏi | Câu trả lời |
| --- | ---: | ---: |
| Tổng số mẫu | 1.000 | 1.000 |
| ID duy nhất | 1.000 | 1.000 |
| Văn bản rỗng | 0 | 0 |
| Số AI judge tối thiểu | 3 | 3 |
| Miền điểm | 1-5 | 1-5 |
| Context group | 240 | 240 |

Tập group của câu hỏi và câu trả lời giống nhau hoàn toàn. Điều này cho phép dùng cùng
một phép chia dữ liệu cho cả hai mô hình.

## 17. Export dataset

Hai file dữ liệu đầy đủ:

- `data/research/weak_supervision_v1/all/question_ranking_all.csv`
- `data/research/weak_supervision_v1/all/answer_scoring_all.csv`

Mỗi dòng câu hỏi chứa:

- ID context và output.
- Stage và criterion.
- Context đã trích xuất.
- Nội dung câu hỏi.
- Model sinh câu hỏi.
- Số AI judge.
- Điểm consensus.
- Các pseudo-label theo rubric.
- Cờ cần manual review.

Mỗi dòng câu trả lời chứa thêm:

- Nội dung câu trả lời.
- Mức chất lượng mục tiêu dưới dạng metadata.
- 7 pseudo-label câu trả lời.

## 18. Chia train, validation và test

Dataset được chia bằng random seed `42` theo hai quy tắc:

1. `Grouped split`: toàn bộ mẫu của cùng một context chỉ nằm trong một tập.
2. `Stratified split`: cố gắng giữ tỷ lệ stage và criterion tương đối cân bằng.

Kết quả:

| Tập | Số mẫu cho mỗi task | Context group | Mục đích |
| --- | ---: | ---: | --- |
| Train | 701 | 168 | Dùng để model học tham số |
| Validation | 147 | 35 | Theo dõi khả năng tổng quát trong lúc phát triển |
| Test | 152 | 37 | Đánh giá cuối trên dữ liệu model chưa được học |

Số context bị trùng giữa các tập là 0.

### 18.1. Vì sao không chia ngẫu nhiên từng dòng?

Nếu các câu hỏi của cùng một CV/JD xuất hiện ở cả train và test, model có thể nhớ context
thay vì học quy luật tổng quát. Đây là `data leakage`, tức rò rỉ thông tin từ dữ liệu đánh
giá vào dữ liệu huấn luyện.

### 18.2. Manifest và checksum

File `manifest.json` lưu:

- Số dòng của từng file.
- Số context group.
- Chiến lược chia.
- Random seed.
- SHA-256 checksum.

`Checksum` là mã băm dùng để xác nhận file không bị thay đổi. Nếu một CSV thay đổi, mã
SHA-256 cũng thay đổi.

## 19. Mô hình 1: Question Quality Ranker

### 19.1. Mục tiêu

Khi nhiều AI sinh nhiều câu hỏi realtime, model nhận các câu hỏi ứng viên và xếp hạng để
chọn câu phù hợp nhất.

### 19.2. Đầu vào

- Context CV/JD đã trích xuất.
- Interview stage.
- Target criterion.
- Nội dung câu hỏi.

### 19.3. Đầu ra

- Điểm chất lượng dự đoán từ 1 đến 5.
- Danh sách câu hỏi sắp xếp từ tốt nhất đến thấp nhất.

### 19.4. Kiến trúc baseline

```text
Văn bản đầu vào
  -> TF-IDF unigram + bigram
  -> tối đa 30.000 đặc trưng
  -> Ridge Regression
  -> điểm chất lượng 1-5
```

`TF-IDF` là phương pháp chuyển văn bản thành vector số dựa trên mức độ quan trọng của từ
và cụm từ trong toàn bộ tập dữ liệu.

`Unigram` là một từ đơn. `Bigram` là hai từ liên tiếp. Kết hợp hai loại giúp model học cả
từ khóa và cụm từ thường thể hiện chất lượng câu hỏi.

`Ridge Regression` là hồi quy tuyến tính có regularization L2. Regularization giúp hạn
chế model phụ thuộc quá mạnh vào một số feature và giảm overfitting.

`Pointwise ranking` nghĩa là model chấm từng câu hỏi thành một điểm riêng rồi sắp xếp các
điểm. Đây là baseline đơn giản, chưa phải learning-to-rank pairwise hoặc listwise.

### 19.5. Quy trình training

1. Fit model tạm trên 701 mẫu train.
2. Tính metric trên 147 mẫu validation.
3. Fit lại artifact cuối trên train + validation, tổng 848 mẫu.
4. Đánh giá một lần trên 152 mẫu test.

### 19.6. Kết quả test

| Metric | Giá trị |
| --- | ---: |
| MAE | 0,180899 |
| RMSE | 0,241709 |
| Spearman | 0,403913 |
| NDCG | 0,992435 |
| Top-1 accuracy | 0,351351 |

Model dự đoán điểm trung bình khá gần nhãn AI, nhưng khả năng chọn đúng câu tốt nhất mới
đạt khoảng 35,1%. Vì mục tiêu cuối là chọn câu hỏi tốt nhất, Top-1 accuracy là điểm cần
ưu tiên cải thiện.

NDCG rất cao một phần vì điểm câu hỏi tập trung nhiều ở vùng cao. Vì vậy không nên dùng
NDCG một mình để kết luận model đã chọn câu hỏi tốt.

## 20. Mô hình 2: Candidate Answer Scorer

### 20.1. Mục tiêu

Model nhận context, câu hỏi và câu trả lời rồi dự đoán 7 điểm chất lượng. Trong realtime,
model có thể dùng sau khi Speech-to-Text chuyển giọng nói ứng viên thành văn bản.

### 20.2. Đầu vào

- Context CV/JD.
- Interview stage.
- Target criterion.
- Nội dung câu hỏi.
- Nội dung câu trả lời.

### 20.3. Đầu ra

7 điểm tương ứng 7 tiêu chí câu trả lời, mỗi điểm từ 1 đến 5.

### 20.4. Kiến trúc baseline

```text
Context + câu hỏi + câu trả lời
  -> TF-IDF unigram + bigram
  -> tối đa 30.000 đặc trưng
  -> Ridge Multi-output Regression
  -> 7 điểm đầu ra
```

`Multi-output regression` là mô hình hồi quy dự đoán nhiều giá trị cùng lúc. Ở đây một
lần dự đoán trả về 7 điểm.

### 20.5. Kết quả test tổng thể

| Metric | Giá trị |
| --- | ---: |
| MAE | 0,727102 |
| RMSE | 0,901325 |
| Spearman | 0,801857 |

### 20.6. Kết quả theo tiêu chí

| Tiêu chí | MAE | RMSE | Spearman |
| --- | ---: | ---: | ---: |
| answer_relevance | 0,811207 | 1,012916 | 0,793598 |
| technical_correctness | 0,750406 | 0,937625 | 0,790377 |
| reasoning_depth | 0,596660 | 0,725825 | 0,865705 |
| evidence_specificity | 0,784639 | 0,914318 | 0,830218 |
| clarity | 0,509671 | 0,646426 | 0,744740 |
| consistency | 0,948275 | 1,127552 | 0,552971 |
| criterion_score | 0,688860 | 0,854125 | 0,807719 |

`Consistency` là tiêu chí yếu nhất. Model gặp khó khi phải xác định câu trả lời có nhất
quán với toàn bộ CV/JD và các thông tin trước đó hay không.

## 21. Giải thích các metric đánh giá

### 21.1. MAE

`Mean Absolute Error`, hay sai số tuyệt đối trung bình, đo trung bình model lệch bao nhiêu
điểm so với nhãn.

Ví dụ MAE 0,73 trên thang 1-5 nghĩa là dự đoán lệch trung bình khoảng 0,73 điểm. MAE càng
thấp càng tốt.

### 21.2. RMSE

`Root Mean Squared Error`, hay căn bậc hai của sai số bình phương trung bình, phạt các sai
số lớn mạnh hơn MAE. RMSE càng thấp càng tốt.

Nếu RMSE lớn hơn MAE đáng kể, model đang có một số dự đoán sai khá xa.

### 21.3. Spearman correlation

`Spearman rank correlation`, hay tương quan thứ hạng Spearman, đo model có xếp mẫu theo
đúng thứ tự cao-thấp giống nhãn hay không.

- Gần 1: thứ hạng rất giống nhau.
- Gần 0: gần như không có quan hệ thứ hạng.
- Gần -1: thứ hạng ngược nhau.

### 21.4. NDCG

`Normalized Discounted Cumulative Gain` là metric xếp hạng. Nó thưởng khi các câu có nhãn
cao được đưa lên đầu danh sách. NDCG gần 1 là tốt, nhưng có thể cao giả tạo nếu các câu
trong cùng nhóm đều có điểm gần nhau.

### 21.5. Top-1 accuracy

Top-1 accuracy đo tỷ lệ model chọn đúng câu có nhãn cao nhất trong mỗi context group.
Đây là metric trực tiếp nhất cho bài toán chọn một câu hỏi để đưa ra cho ứng viên.

## 22. Artifact và file kết quả

### 22.1. Dataset đầy đủ

- `data/research/weak_supervision_v1/all/question_ranking_all.csv`
- `data/research/weak_supervision_v1/all/answer_scoring_all.csv`

### 22.2. Dataset đã chia

- `data/research/weak_supervision_v1/splits/question_train.csv`
- `data/research/weak_supervision_v1/splits/question_validation.csv`
- `data/research/weak_supervision_v1/splits/question_test.csv`
- `data/research/weak_supervision_v1/splits/answer_train.csv`
- `data/research/weak_supervision_v1/splits/answer_validation.csv`
- `data/research/weak_supervision_v1/splits/answer_test.csv`

### 22.3. Model và metric

- `artifacts/weak_supervision/v1/question_quality_ranker.joblib`
- `artifacts/weak_supervision/v1/question_quality_ranker_metrics.json`
- `artifacts/weak_supervision/v1/candidate_answer_scorer.joblib`
- `artifacts/weak_supervision/v1/candidate_answer_scorer_metrics.json`

`Artifact` là file chứa model đã train cùng cấu hình cần thiết để load và dự đoán mà
không cần training lại.

## 23. Vì sao cần thêm dữ liệu?

1.000 bộ đủ để tạo baseline và kiểm tra pipeline, nhưng chưa đủ để khẳng định model tổng
quát tốt trong nhiều ngành nghề và cách trả lời khác nhau.

### 23.1. Thêm context CV/JD

Mục đích:

- Bao phủ nhiều vị trí công việc hơn.
- Bao phủ intern, junior, middle, senior và lead.
- Tránh model chỉ quen với backend, Python hoặc một nhóm kỹ năng phổ biến.
- Giúp model xử lý CV ngắn, dài, thiếu thông tin hoặc có kinh nghiệm trái ngành.

### 23.2. Thêm câu hỏi khó phân biệt

Hiện phần lớn câu hỏi được AI chấm khá cao. Cần thêm `hard negatives`, tức các câu nhìn
có vẻ hợp lý nhưng thực tế bị lặp, mơ hồ, quá chung hoặc không bám CV.

Mục đích:

- Tăng khoảng cách giữa câu tốt và câu chưa tốt.
- Cải thiện Top-1 accuracy.
- Giảm tình trạng model chỉ dự đoán mọi câu ở vùng điểm cao.

### 23.3. Thêm nhiều câu trả lời cho cùng một câu hỏi

Hiện mỗi output chủ yếu có một câu trả lời tổng hợp. Có thể sinh thêm nhiều phong cách:

- Câu trả lời ngắn nhưng chính xác.
- Câu trả lời dài nhưng lan man.
- Câu trả lời đúng lý thuyết nhưng thiếu bằng chứng.
- Câu trả lời có ví dụ nhưng mâu thuẫn CV.
- Câu trả lời có lỗi nhỏ về kỹ thuật.

Mục đích là giúp model phân biệt chất lượng dựa trên nội dung, không dựa vào độ dài hoặc
một mẫu câu cố định.

### 23.4. Thêm câu trả lời thật từ người dùng

Câu trả lời tổng hợp thường sạch và có cấu trúc hơn câu nói thật. Dữ liệu thật có thể có:

- Từ đệm.
- Câu nói chưa hoàn chỉnh.
- Lỗi Speech-to-Text.
- Trả lời vòng vo.
- Trộn tiếng Việt và tiếng Anh.
- Cách diễn đạt không chuẩn nhưng ý đúng.

Mục đích là giảm khoảng cách giữa dữ liệu tổng hợp và môi trường realtime.

### 23.5. Thêm dữ liệu owner review

Bạn review các mẫu AI bất đồng để:

- Phát hiện lỗi consensus.
- Phát hiện câu hỏi không tự nhiên.
- Sửa các nhãn quá cao hoặc quá thấp.
- Tạo một tập kiểm tra thủ công riêng.
- Đánh giá model theo tiêu chuẩn nhất quán của chính đề tài.

Owner review không được gọi là expert ground truth, nhưng vẫn là lớp kiểm soát chất lượng
quan trọng.

## 24. Quy trình owner manual review đề xuất

1. Lấy mẫu theo stage, criterion, model sinh và mức độ bất đồng.
2. Ưu tiên các mẫu có `needs_human_review=true`.
3. Ẩn tên model sinh và điểm AI trong lúc review để giảm thiên kiến.
4. Chọn `approve`, `correct` hoặc `reject`.
5. Nếu chọn `correct`, nhập lại điểm theo đúng rubric.
6. Ghi lý do ngắn gọn cho mỗi thay đổi.
7. Không ghi đè raw AI judgments.
8. Tạo dataset mới `weak-supervision-v1.1`.
9. Train lại trên dữ liệu cũ cộng dữ liệu đã sửa.
10. Giữ một test set khóa riêng, không dùng để điều chỉnh model.

## 25. Cần bao nhiêu dữ liệu tiếp theo?

| Mốc | Mục đích |
| ---: | --- |
| 1.000 bộ | Baseline, kiểm tra pipeline và phát hiện lỗi chính |
| 3.000-5.000 bộ | Cải thiện độ đa dạng và bắt đầu so sánh model nghiêm túc |
| Trên 10.000 bộ | Phù hợp hơn khi cần bao phủ nhiều ngành, level và cách trả lời |

Không nên chỉ tăng số lượng. Dữ liệu mới phải tăng tính đa dạng và bổ sung các trường hợp
model hiện đang làm sai.

## 26. Các cải thiện ưu tiên tiếp theo

### Ưu tiên 1: Cải thiện mô hình chọn câu hỏi

Lý do: Top-1 accuracy hiện chỉ khoảng 35,1%.

Hướng cải thiện:

- Sinh thêm hard negatives.
- Chuyển từ pointwise regression sang pairwise learning-to-rank.
- Thử LambdaMART hoặc XGBoost Ranker.
- Dùng multilingual sentence embedding thay TF-IDF.
- Thử cross-encoder đọc trực tiếp context và câu hỏi.
- Tối ưu metric Top-1 hoặc NDCG theo group thay vì chỉ tối ưu sai số điểm.

`Pairwise ranking` là cho model học câu A tốt hơn câu B. `Listwise ranking` là cho model
học thứ tự của toàn bộ danh sách câu hỏi cùng lúc.

### Ưu tiên 2: Cải thiện tiêu chí consistency của câu trả lời

Hướng cải thiện:

- Tạo thêm câu trả lời cố ý mâu thuẫn với CV.
- Tạo cặp câu trả lời gần giống nhưng khác một chi tiết quan trọng.
- Tách feature so sánh answer với CV/JD.
- Dùng embedding hoặc cross-encoder để đo quan hệ giữa context và answer.

### Ưu tiên 3: Thử mô hình mạnh hơn baseline

Baseline hiện tại nhẹ, nhanh và dễ giải thích nhưng TF-IDF không hiểu ngữ nghĩa sâu.

Các lựa chọn tiếp theo:

- Sentence Transformer đa ngôn ngữ để tạo embedding.
- Gradient Boosting trên embedding và feature thống kê.
- Neural multi-task model dự đoán đồng thời 7 tiêu chí.
- Cross-encoder cho bài toán chọn câu hỏi.

`Embedding` là vector số biểu diễn ý nghĩa ngữ nghĩa của văn bản. Hai văn bản gần nghĩa
sẽ có vector gần nhau dù không dùng đúng cùng từ khóa.

### Ưu tiên 4: Phân tích lỗi có hệ thống

Cần xuất các nhóm:

- Câu model xếp hạng sai nhiều nhất.
- Câu có AI disagreement cao.
- Câu có dự đoán chênh nhãn trên 1 điểm.
- Lỗi theo stage.
- Lỗi theo criterion.
- Lỗi theo provider sinh dữ liệu.

Mục đích là quyết định cần thêm loại dữ liệu nào thay vì sinh thêm ngẫu nhiên.

### Ưu tiên 5: Tích hợp vào phỏng vấn realtime

Luồng dự kiến:

```text
3-4 AI sinh câu hỏi ứng viên
  -> Question Quality Ranker chấm và chọn câu tốt nhất
  -> TTS đọc câu hỏi
  -> ứng viên trả lời
  -> STT tạo transcript
  -> Candidate Answer Scorer chấm 7 tiêu chí
  -> Interview State cập nhật năng lực đã kiểm tra
  -> sinh vòng câu hỏi tiếp theo
```

Cần đo thêm latency P50/P95 để bảo đảm việc gọi nhiều AI và chạy ranker không làm người
dùng phải chờ quá lâu.

## 27. Rủi ro và giới hạn cần ghi trong báo cáo đề tài

- Nhãn là pseudo-label AI, không phải expert ground truth.
- Các AI khác nhau vẫn có thể cùng chia sẻ thiên kiến từ dữ liệu huấn luyện chung.
- Dữ liệu câu trả lời hiện chủ yếu là dữ liệu tổng hợp.
- Điểm câu hỏi tập trung nhiều ở vùng cao.
- Question Ranker chưa đạt Top-1 accuracy cao.
- Answer Scorer yếu nhất ở tiêu chí consistency.
- Chưa có kiểm thử với câu trả lời giọng nói thật và lỗi STT.
- Chưa tích hợp hai model vào luồng realtime cuối cùng.
- Chưa có owner manual review trong dataset V1.

## 28. Các thuật ngữ chuyên ngành chính

| Thuật ngữ | Giải thích tiếng Việt |
| --- | --- |
| Dataset | Tập dữ liệu dùng để huấn luyện hoặc đánh giá model |
| Training | Quá trình model học quy luật từ dữ liệu |
| Inference | Quá trình dùng model đã train để dự đoán dữ liệu mới |
| Label | Giá trị đúng hoặc mục tiêu model cần học |
| Pseudo-label | Nhãn tự động do AI hoặc quy tắc tạo ra |
| Weak supervision | Huấn luyện bằng nhãn tự động hoặc nhãn chưa phải ground truth tuyệt đối |
| Ground truth | Nhãn tham chiếu được coi là chuẩn |
| Rubric | Bộ tiêu chí và hướng dẫn chấm điểm |
| Consensus | Kết quả đồng thuận sau khi tổng hợp nhiều người/model chấm |
| Median | Trung vị, giá trị ở giữa sau khi sắp xếp |
| Provider | Nhà cung cấp dịch vụ AI/API |
| Failover | Tự chuyển sang provider hoặc key dự phòng khi có lỗi |
| Quota | Hạn mức request hoặc token của API |
| Token | Đơn vị văn bản mà mô hình ngôn ngữ xử lý |
| Context | Thông tin nền gồm CV, JD, lịch sử và trạng thái phỏng vấn |
| Feature | Đặc trưng đầu vào dạng số mà model sử dụng |
| TF-IDF | Cách biểu diễn mức quan trọng của từ/cụm từ thành vector |
| Ridge Regression | Hồi quy tuyến tính có regularization L2 |
| Multi-output | Một model dự đoán nhiều giá trị đầu ra cùng lúc |
| Baseline | Mô hình nền tảng đầu tiên dùng làm mốc so sánh |
| Overfitting | Model nhớ dữ liệu train nhưng dự đoán kém dữ liệu mới |
| Data leakage | Thông tin từ test hoặc đáp án vô tình lọt vào training |
| Train set | Dữ liệu dùng để model học |
| Validation set | Dữ liệu dùng để kiểm tra và lựa chọn trong quá trình phát triển |
| Test set | Dữ liệu khóa riêng để đánh giá cuối |
| Grouped split | Chia theo nhóm để một context không xuất hiện ở nhiều tập |
| Stratified split | Chia sao cho phân bố các nhóm quan trọng tương đối cân bằng |
| Artifact | File model đã train có thể load để dự đoán |
| Checkpoint | File lưu tiến độ để tiếp tục sau khi dừng |
| Rollback | Hủy thay đổi chưa hoàn chỉnh trong database |
| Latency | Thời gian từ lúc gửi yêu cầu đến lúc có kết quả |
| STT | Speech-to-Text, chuyển giọng nói thành văn bản |
| TTS | Text-to-Speech, chuyển văn bản thành giọng nói |
| Embedding | Vector số biểu diễn ý nghĩa ngữ nghĩa của văn bản |
| Learning-to-rank | Mô hình học cách sắp xếp các ứng viên theo chất lượng |

## 29. Lệnh tái tạo quy trình

Export dataset từ Supabase:

```powershell
python -B scripts\export_weak_supervision_datasets.py
```

Chia train, validation và test:

```powershell
python -B scripts\split_weak_supervision_datasets.py
```

Training hai baseline:

```powershell
python -B scripts\train_weak_supervision_models.py
```

Chạy kiểm thử:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Kết quả kiểm thử hiện tại: 58/58 test thành công.

## 30. Trạng thái hiện tại và bước tiếp theo

### Đã hoàn thành

- Kết nối database Supabase.
- Xây dựng cơ chế nhiều provider và nhiều key dự phòng.
- Xây dựng failover và checkpoint.
- Sinh đủ 1.000 bộ hoàn chỉnh.
- Export 1.000 dòng câu hỏi và 1.000 dòng câu trả lời.
- Kiểm tra tính hợp lệ của dataset.
- Chia grouped-stratified train/validation/test.
- Train Question Quality Ranker baseline.
- Train Candidate Answer Scorer baseline.
- Lưu artifact và metric.
- Smoke test load model và inference.
- Chạy thành công 58 unit test.

### Chưa thực hiện

- Owner manual review.
- Dataset V1.1 sau review.
- Mô hình learning-to-rank mạnh hơn.
- Mô hình embedding/cross-encoder.
- Tích hợp model vào API phỏng vấn realtime.
- STT/TTS hoàn chỉnh.
- Đo WER, latency P50/P95 và load test.
- Đánh giá trên câu trả lời thật của người dùng.

### Bước nên làm ngay

1. Tạo file hoặc giao diện owner-review cho các mẫu `needs_human_review=true`.
2. Review trước một tập stratified khoảng 100-200 câu hỏi và 100-200 câu trả lời.
3. Phân tích lỗi Top-1 của Question Ranker.
4. Sinh thêm hard negatives cho câu hỏi.
5. Tạo dataset V1.1 và retrain.
6. Sau khi model ổn định hơn, tích hợp vào luồng phỏng vấn realtime.

## 31. Kết luận

Dự án đã hoàn thành pipeline dữ liệu và training baseline theo hướng nhiều AI tự sinh,
tự chấm chéo và tạo pseudo-label. Hai model hiện tại chứng minh quy trình từ dữ liệu đến
artifact có thể chạy hoàn chỉnh và tái tạo được.

Candidate Answer Scorer đã học được thứ hạng nhãn AI tương đối tốt, nhưng vẫn cần cải
thiện consistency và kiểm thử trên câu trả lời thật. Question Quality Ranker dự đoán điểm
khá gần nhãn nhưng Top-1 accuracy còn thấp, vì vậy cần ưu tiên bổ sung hard negatives và
chuyển sang mô hình learning-to-rank mạnh hơn.

Kết quả V1 là baseline kỹ thuật dựa trên AI consensus. Nó không phải kết quả đối chiếu
chuyên gia. Lớp kiểm soát chất lượng tiếp theo là owner manual review do chính người thực
hiện đề tài tiến hành.
