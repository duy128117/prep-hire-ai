# Bản nháp cũ không dấu

> Tài liệu này đã được thay thế bởi bản báo cáo tiếng Việt đầy đủ:
> `BAO_CAO_QUY_TRINH_DU_LIEU_VA_TRAINING_AI_V1.md`.

# Bao cao training Weak Supervision V1

## 1. Tom tat

Phien ban V1 da train hai mo hinh baseline tren 1.000 bo du lieu hoan chinh:

1. `Question Quality Ranker`: cham diem va xep hang cac cau hoi phong van do AI sinh.
2. `Candidate Answer Scorer`: du doan bay diem chat luong cho cau tra loi phong van.

Toan bo nhan training la `multi-LLM consensus pseudo-label`. Khong co chuyen gia nao
tham gia gan nhan, tao reference standard hay hieu chinh diem trong phien ban nay.
Nguoi thuc hien de tai se tu manual review cac mau can kiem tra o giai doan sau.

## 2. Pham vi hien tai

Pipeline dang duoc su dung:

```text
CV + JD + interview state
  -> 4 AI doc lap sinh cau hoi/cau tra loi
  -> moi mau duoc 3 AI con lai cham cheo
  -> median theo tung tieu chi
  -> pseudo-label consensus
  -> grouped-stratified train/validation/test
  -> train Question Quality Ranker
  -> train Candidate Answer Scorer
  -> owner manual review sau training
```

Pipeline V1 khong su dung:

- Expert annotation.
- Expert consensus reference standard.
- Candidate evaluation calibration theo diem chuyen gia.
- Bieu cam khuon mat de danh gia nang luc.
- Ten provider/model sinh mau lam feature training.
- Muc chat luong muc tieu 1-5 cua cau tra loi lam feature training.

Nhung bang, endpoint hoac script expert-calibration cu con ton tai trong repository chi la
ma legacy. Chung khong duoc goi trong pipeline V1 va khong anh huong hai artifact da train.

## 3. Cach sinh va gan nhan du lieu

Moi generation round chon bon provider AI doc lap. Moi provider sinh mot cau hoi va mot
cau tra loi tu cung context CV/JD. Provider sinh mau khong duoc cham chinh mau cua no.
Ba provider con lai cham doc lap theo rubric 1-5.

### 3.1. Nhan cau hoi

Bay tieu chi cham cau hoi:

- `context_relevance`
- `technical_depth`
- `clarity`
- `evidence_elicitation`
- `adaptivity`
- `non_redundancy`
- `fairness`

Voi moi tieu chi, nhan consensus la trung vi cua ba diem AI. `mean_quality_score` la
trung binh cua bay trung vi nay. Day la muc tieu hoi quy cua Question Quality Ranker.

### 3.2. Nhan cau tra loi

Bay tieu chi cham cau tra loi:

- `answer_relevance`
- `technical_correctness`
- `reasoning_depth`
- `evidence_specificity`
- `clarity`
- `consistency`
- `criterion_score`

Voi moi tieu chi, nhan consensus cung la trung vi cua ba diem AI. Candidate Answer
Scorer hoc truc tiep bay muc tieu nay bang multi-output regression.

### 3.3. Co che danh dau manual review

Mot mau duoc dat `needs_human_review=true` neu it nhat mot tieu chi co khoang cach diem
giua cac AI tu 2 diem tro len, hoac quyet dinh consensus khong phai `accept`.

- Cau hoi can manual review: 636/1.000.
- Cau tra loi can manual review: 712/1.000.
- Manual review da hoan thanh tai thoi diem train V1: 0.

Con so tren khong lam mau bi loai khoi V1. No tao hang doi de nguoi thuc hien de tai tu
kiem tra, sua nhan neu can va tao dataset V1.1 sau nay.

## 4. Kiem tra dataset truoc training

| Kiem tra | Cau hoi | Cau tra loi |
| --- | ---: | ---: |
| Tong so mau | 1.000 | 1.000 |
| ID duy nhat | 1.000 | 1.000 |
| Van ban rong | 0 | 0 |
| So AI judge toi thieu | 3 | 3 |
| Mien diem | 1-5 | 1-5 |

Hai dataset co cung 240 context group. Cac quality level tong hop cua cau tra loi duoc
can bang: moi muc 1, 2, 3, 4 va 5 co dung 200 mau.

Phan bo model sinh mau:

| Model | So mau |
| --- | ---: |
| Cloudflare Qwen3 30B | 231 |
| OpenTyphoon 2.5 30B | 225 |
| Ollama Cloud Gemma 4 31B | 222 |
| Inception Mercury 2.5 | 195 |
| Groq GPT-OSS 120B | 109 |
| Cohere Command A | 9 |
| Hugging Face DeepSeek V3 | 9 |

Cohere va Hugging Face la du lieu lich su hop le duoc sinh truoc khi hai provider nay bi
loai khoi pool hien tai.

## 5. Chia train, validation va test

Dataset duoc chia bang random seed `42`, grouped theo `case_id` va stratified theo
`stage + target_criterion`. Cau hoi va cau tra loi dung cung mot anh xa group-to-split.

| Tap | So mau moi task | Context group | Ty le gan dung |
| --- | ---: | ---: | ---: |
| Train | 701 | 168 | 70% |
| Validation | 147 | 35 | 15% |
| Test | 152 | 37 | 15% |

So context trung nhau giua train/validation/test la 0. Vi vay cac cau hoi hoac cau tra
loi tu cung mot CV/JD context khong the xuat hien o hai tap khac nhau.

Manifest luu row count, split strategy va SHA-256 cua tung file tai
`data/research/weak_supervision_v1/manifest.json`.

## 6. Mo hinh 1: Question Quality Ranker

### Dau vao

- Context CV/JD dang JSON da trich xuat.
- Interview stage.
- Target criterion.
- Candidate question.

### Xu ly va mo hinh

- Noi cac truong thanh mot chuoi co marker `CONTEXT`, `STAGE`, `TARGET`, `QUESTION`.
- TF-IDF word unigram + bigram.
- Toi da 30.000 feature.
- Ridge regression, `alpha=1.0`, solver `lsqr`.
- Diem du doan duoc gioi han trong khoang 1-5.
- Cac cau hoi duoc sap xep giam dan theo diem du doan.

### Quy trinh fit

1. Fit model tam tren 701 mau train.
2. Do metric tren 147 mau validation.
3. Fit lai artifact cuoi tren train + validation, tong 848 mau.
4. Danh gia mot lan tren 152 mau test giu rieng.

### Ket qua test

| Metric | Gia tri |
| --- | ---: |
| MAE | 0.180899 |
| RMSE | 0.241709 |
| Spearman | 0.403913 |
| NDCG | 0.992435 |
| Top-1 accuracy | 0.351351 |

MAE thap cho thay model uoc luong diem trung binh kha sat pseudo-label. Tuy nhien Top-1
35,1% cho thay kha nang chon dung cau hoi tot nhat trong group van chi o muc baseline.
NDCG rat cao mot phan vi phan bo diem cau hoi tap trung o vung cao, nen khong duoc dung
NDCG mot minh de ket luan model chon cau hoi tot.

## 7. Mo hinh 2: Candidate Answer Scorer

### Dau vao

- Context CV/JD dang JSON da trich xuat.
- Interview stage.
- Target criterion.
- Question.
- Candidate answer.

### Xu ly va mo hinh

- Noi cac truong thanh chuoi co marker ro rang.
- TF-IDF word unigram + bigram, toi da 30.000 feature.
- Ridge multi-output regression, `alpha=1.0`, solver `lsqr`.
- Dau ra la bay diem 1-5 tuong ung bay tieu chi cau tra loi.

Quy trinh fit cung la 701 train, 147 validation, sau do fit artifact tren 848 mau va
danh gia tren 152 mau test.

### Ket qua test tong the

| Metric | Gia tri |
| --- | ---: |
| MAE | 0.727102 |
| RMSE | 0.901325 |
| Spearman | 0.801857 |

### Ket qua theo tieu chi

| Tieu chi | MAE | RMSE | Spearman |
| --- | ---: | ---: | ---: |
| answer_relevance | 0.811207 | 1.012916 | 0.793598 |
| technical_correctness | 0.750406 | 0.937625 | 0.790377 |
| reasoning_depth | 0.596660 | 0.725825 | 0.865705 |
| evidence_specificity | 0.784639 | 0.914318 | 0.830218 |
| clarity | 0.509671 | 0.646426 | 0.744740 |
| consistency | 0.948275 | 1.127552 | 0.552971 |
| criterion_score | 0.688860 | 0.854125 | 0.807719 |

`consistency` la diem yeu nhat va can duoc uu tien khi manual review hoac khi thay doi
feature/model o phien ban tiep theo.

## 8. File dau ra

Dataset day du:

- `data/research/weak_supervision_v1/all/question_ranking_all.csv`
- `data/research/weak_supervision_v1/all/answer_scoring_all.csv`

Dataset da chia:

- `data/research/weak_supervision_v1/splits/question_train.csv`
- `data/research/weak_supervision_v1/splits/question_validation.csv`
- `data/research/weak_supervision_v1/splits/question_test.csv`
- `data/research/weak_supervision_v1/splits/answer_train.csv`
- `data/research/weak_supervision_v1/splits/answer_validation.csv`
- `data/research/weak_supervision_v1/splits/answer_test.csv`

Artifact va metric:

- `artifacts/weak_supervision/v1/question_quality_ranker.joblib`
- `artifacts/weak_supervision/v1/question_quality_ranker_metrics.json`
- `artifacts/weak_supervision/v1/candidate_answer_scorer.joblib`
- `artifacts/weak_supervision/v1/candidate_answer_scorer_metrics.json`

CSV va model binary khong duoc commit Git vi co the tai tao. README, manifest va metric
JSON duoc giu lai de theo doi thuc nghiem.

## 9. Trang thai hoan thanh

- Export dung 1.000 mau cho moi task: hoan thanh.
- Data validation: hoan thanh.
- Grouped-stratified split: hoan thanh.
- Train hai baseline: hoan thanh.
- Smoke test load artifact va inference: hoan thanh.
- Unit test: 58/58 dat.
- Owner manual review: chua thuc hien.
- Tich hop model vao realtime interview: chua thuc hien.

## 10. Cach manual review theo huong khong dung chuyen gia

Nguoi thuc hien de tai co the review theo cac buoc sau:

1. Lay mau stratified theo stage, criterion, generator va `needs_human_review`.
2. An ten model sinh va diem consensus trong luc review de giam thien kien.
3. Danh dau `approve`, `correct` hoac `reject` va ghi ly do ngan gon.
4. Neu sua diem, luu diem owner-review rieng, khong ghi de raw AI judgments.
5. Tao dataset V1.1 gom V1 va cac sua doi da review.
6. Train lai tren toan bo train/validation V1.1 va giu mot test set khoa rieng.

Ket qua V1 hien tai chi do kha nang model nho hoc lai consensus cua nhom AI. No khong
duoc mo ta la do chinh xac so voi chuyen gia, vi de tai hien tai khong dung chuyen gia.
