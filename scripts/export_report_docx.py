import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import shutil
from pathlib import Path

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    # Page setup: Standard A4, margins 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Set normal style font
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_sub = p_title.add_run("DỰ ÁN NEXTSTEP — PBL5 AI RESEARCH\n")
    run_sub.font.size = Pt(11)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)

    run_title = p_title.add_run("BÁO CÁO TỔNG HỢP: QUY TRÌNH SINH DỮ LIỆU, QUY CHUẨN ĐÁNH GIÁ, CƠ CHẾ TÍNH ĐIỂM VÀ LƯU TRỮ SUPABASE")
    run_title.font.size = Pt(17)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x11, 0x18, 0x27)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(20)
    run_meta = p_meta.add_run("Hệ thống phỏng vấn AI thích ứng (Stateful Adaptive AI Interview) — Multi-LLM Weak Supervision")
    run_meta.font.italic = True
    run_meta.font.size = Pt(10.5)
    run_meta.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    # Helper for Headings
    def add_heading1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A)
        return h

    def add_heading2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)
        return h

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_bold = p.add_run(bold_prefix + ": ")
        r_bold.font.bold = True
        p.add_run(text)
        return p

    def add_p(text, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.2
        r = p.add_run(text)
        r.font.italic = italic
        return p

    def style_table(table, col_widths, headers, rows_data):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_cells = table.rows[0].cells
        for idx, title in enumerate(headers):
            hdr_cells[idx].text = title
            set_cell_background(hdr_cells[idx], "1E40AF")
            set_cell_margins(hdr_cells[idx], top=120, bottom=120, left=150, right=150)
            p = hdr_cells[idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for row_idx, r_data in enumerate(rows_data):
            row_cells = table.add_row().cells
            bg_color = "F9FAFB" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(r_data):
                row_cells[col_idx].text = str(val)
                set_cell_background(row_cells[col_idx], bg_color)
                set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
                p = row_cells[col_idx].paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                if col_idx == 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    if len(p.runs) > 0:
                        p.runs[0].font.size = Pt(9.5)
                else:
                    if len(p.runs) > 0:
                        p.runs[0].font.size = Pt(9.5)

        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    # 1. TỔNG QUAN
    add_heading1("1. TỔNG QUAN HƯỚNG TIẾP CẬN VÀ BỐI CẢNH")
    add_p("Hệ thống AI phỏng vấn của NextStep được xây dựng trên nền tảng Weak Supervision (Giám sát yếu) kết hợp Multi-LLM Jury (Hội đồng LLM chấm chéo). Mục tiêu trọng tâm là giải quyết bài toán cốt lõi: Làm thế nào để tạo ra các mô hình Machine Learning nhỏ gọn, suy luận theo thời gian thực (real-time) với chi phí gần như bằng 0 mà vẫn đạt độ chính xác cao tương đương các mô hình ngôn ngữ lớn (LLM), hoàn toàn không phụ thuộc vào chuyên gia con người tốn kém?")
    add_p("Đặc điểm cốt lõi của giải pháp:", italic=True)
    add_bullet("Không sử dụng chuyên gia", "Loại bỏ hoàn toàn quy trình dán nhãn thủ công hoặc hiệu chỉnh điểm chuyên gia (expert-calibration).")
    add_bullet("Tự động hóa toàn phần", "Thu thập nhãn đồng thuận (consensus pseudo-labels) từ nhiều mô hình AI độc lập để huấn luyện mô hình.")
    add_bullet("Tối ưu hóa thời gian thực", "Mô hình sau khi huấn luyện có dung lượng siêu nhẹ, suy luận dưới 30ms, phục vụ trực tiếp trên giao diện web.")

    # 2. QUY TRÌNH SINH DỮ LIỆU
    add_heading1("2. QUY TRÌNH CÁC AI THAY NHAU SINH DỮ LIỆU HUẤN LUYỆN")
    add_heading2("2.1. Hội đồng AI (LLM Pool)")
    add_p("Quy trình sử dụng một tập hợp các mô hình ngôn ngữ lớn nguồn mở và thương mại độc lập nhằm tạo sự phong phú về phong cách ngôn ngữ và triệt tiêu thiên kiến chủ quan (bias):")
    add_bullet("Cloudflare Qwen3 30B", "Đảm nhiệm vai trò sinh câu hỏi và chấm chéo đa tiêu chí.")
    add_bullet("OpenTyphoon 2.5 30B", "Mô hình tối ưu ngữ cảnh Đông Nam Á, sinh câu trả lời và đánh giá kỹ thuật.")
    add_bullet("Ollama Cloud Gemma 4 31B", "Tham gia hội đồng chấm chéo và sinh câu trả lời đối sánh.")
    add_bullet("Inception Mercury 2.5", "Chấm chéo độ liên quan và tính nhất quán ngữ cảnh.")
    add_bullet("Groq GPT-OSS 120B", "Mô hình tham chiếu kích thước lớn, tích hợp multi-key và failover tự động.")

    add_heading2("2.2. Cơ chế xoay vòng: 1 Generator – 3 Jury")
    add_p("Mỗi vòng sinh dữ liệu (Generation Round) tuân thủ quy tắc khép kín và khách quan:")
    add_bullet("Bước 1 - Ngữ cảnh thực tế", "Hệ thống bốc ngẫu nhiên thông tin trích xuất có cấu trúc từ cặp CV và JD (trong tổng số 240 context groups độc lập).")
    add_bullet("Bước 2 - Chọn 4 AI độc lập", "Trong mỗi vòng, 4 nhà cung cấp AI được chỉ định từ pool.")
    add_bullet("Bước 3 - Sinh câu hỏi và câu trả lời", "1 AI đóng vai Generator sinh ra 1 câu hỏi phỏng vấn tương thích theo giai đoạn (stage) và tiêu chí mục tiêu. Tiếp đó, AI này sinh 1 câu trả lời nhân tạo (Synthetic Answer) theo mức chất lượng chỉ định từ 1 đến 5.")
    add_bullet("Bước 4 - Hội đồng 3 AI chấm chéo", "3 AI còn lại nhận mẫu và chấm điểm độc lập theo bảng Rubric định lượng (thang 1 - 5). AI sinh mẫu tuyệt đối KHÔNG được tự chấm mẫu của mình (No Self-Grading).")
    add_bullet("Quy mô dữ liệu qua 2 đợt", "Hệ thống đã hoàn thành 2 đợt sinh lớn (balanced-weak-v1 và balanced-weak-v1.1), tạo ra 2.000 câu hỏi và 2.000 câu trả lời cân bằng đều trên 5 mức chất lượng (mỗi mức 1-5 có đúng 400 mẫu).")

    # 3. CƠ SỞ ĐÁNH GIÁ VÀ CHI TIẾT CÁCH CHẤM ĐIỂM 7 TIÊU CHÍ
    add_heading1("3. CƠ SỞ ĐÁNH GIÁ VÀ CHI TIẾT CÁCH CHẤM ĐIỂM 7 TIÊU CHÍ")
    add_heading2("3.1. Câu trả lời được chấm dựa vào đâu? (Không chấm cô lập)")
    add_p("Một sai lầm phổ biến là đánh giá câu trả lời một cách cô lập. Trong NextStep, câu trả lời của ứng viên được neo chặt vào một Ngữ cảnh Đánh giá Toàn diện (Evaluation Context) bao gồm:")
    add_bullet("cv_profile", "Hồ sơ CV ứng viên (kỹ năng, kinh nghiệm, dự án thực tế đã khai).")
    add_bullet("job_profile", "Yêu cầu kỹ thuật, trách nhiệm và vai trò thực tế trong JD.")
    add_bullet("interview_history", "Lịch sử 4 lượt hỏi-đáp gần nhất để đảm bảo tính tiếp nối và không tự mâu thuẫn.")
    add_bullet("question & metadata", "Câu hỏi vừa hỏi, độ khó (difficulty), dạng câu hỏi (question_type).")
    add_bullet("required_signals", "Danh sách các tín hiệu năng lực cốt lõi bắt buộc phải xuất hiện trong câu trả lời.")
    add_bullet("bonus_signals", "Các tín hiệu mở rộng/nâng cao để cộng thêm điểm.")
    add_bullet("contradiction_rules", "Các quy tắc phát hiện mâu thuẫn với CV hoặc lời nói trước đó.")

    add_heading2("3.2. Bảng quy chuẩn và căn cứ chấm điểm 7 tiêu chí câu trả lời (Turn Evidence Rubric)")
    table_a = doc.add_table(rows=1, cols=4)
    a_data = [
        ["1", "answer_relevance (Độ liên quan)", "So sánh nội dung câu trả lời với câu hỏi (question).", "Trả lời trực diện trọng tâm. Trừ điểm nặng nếu vòng vo, chèn văn mẫu rào đón, lạc đề."],
        ["2", "technical_correctness (Độ chính xác kỹ thuật)", "Đối chiếu chuẩn kỹ thuật chuyên ngành và yêu cầu trong job_profile.", "Thuật ngữ, kiến trúc, luồng xử lý chính xác. Trừ điểm nặng (mức 1-2) nếu phát biểu sai bản chất kỹ thuật."],
        ["3", "expected_signal_coverage (Độ phủ tín hiệu)", "Đối soát định lượng với required_signals và bonus_signals.", "Bao phủ bao nhiêu % tín hiệu bắt buộc. Chốt chặn: Dù câu dài bao nhiêu, nếu thiếu tín hiệu này điểm vẫn thấp."],
        ["4", "reasoning_quality (Chiều sâu tư duy)", "Đánh giá cấu trúc giải thích và cách tiếp cận vấn đề.", "Giải thích được lý do Tại sao (Why), phân tích đánh đổi (trade-offs). Trừ điểm nếu chỉ nói bề mặt."],
        ["5", "evidence_quality (Chất lượng dẫn chứng)", "Đối chiếu với các dự án đã khai trong cv_profile.", "Dẫn chứng theo phương pháp STAR (Tình huống, Hành động, Số liệu kết quả). Trừ điểm nếu nói lý thuyết suông."],
        ["6", "clarity_conciseness (Rõ ràng & Súc tích)", "Đo lường mật độ thông tin hữu ích (Information Density).", "Gãy gọn, đủ ý trong dung lượng hợp lý. Trừ điểm mạnh nếu dài dòng, lặp từ, chém gió lan man."],
        ["7", "consistency (Tính nhất quán)", "Đối chiếu chéo với cv_profile và interview_history.", "Khớp với CV và không phản bội lại lời nói ở các lượt trước. Trừ điểm nếu tự mâu thuẫn hoặc bịa đặt kinh nghiệm."]
    ]
    style_table(table_a, [0.5, 1.8, 1.9, 2.3], ["STT", "Tiêu chí", "Căn cứ đánh giá", "Quy tắc cộng / trừ điểm"], a_data)

    add_p("Khung điểm chuẩn neo (Score Anchors):", italic=True)
    add_bullet("Mức 1 (Fails)", "Không đạt, sai kiến thức kỹ thuật cơ bản, hoàn toàn lạc đề hoặc mâu thuẫn nghiêm trọng.")
    add_bullet("Mức 2 (Limited)", "Minh chứng rất hạn chế, còn nhiều lỗ hổng lớn về kỹ năng hoặc trả lời hời hợt.")
    add_bullet("Mức 3 (Basic)", "Đạt mức cơ bản, chấp nhận được, nhưng còn thiếu sót những tín hiệu trọng yếu.")
    add_bullet("Mức 4 (Strong)", "Vững vàng, chính xác, có dẫn chứng dự án rõ ràng và lập luận kỹ thuật tốt.")
    add_bullet("Mức 5 (Excellent)", "Xuất sắc, chính xác tuyệt đối, lập luận sắc bén, súc tích và bao quát toàn diện.")

    add_heading2("3.3. Cơ chế triệt tiêu thiên vị câu trả lời dài (Length / Verbosity Bias)")
    add_p("Hiện tượng Thiên vị độ dài (Verbosity Bias) trong NLP khiến các mô hình có xu hướng ngầm chấm điểm cao cho câu trả lời dài. Hệ thống NextStep áp dụng 4 tầng bảo vệ:")
    add_bullet("Lớp 1 - Chỉ thị Chấm mù", "Ẩn hoàn toàn độ dài yêu cầu. Kèm chỉ thị nghiêm ngặt: 'Never reward verbosity by itself. A concise answer can score 5 when it fully satisfies a concise question.' (Tuyệt đối không thưởng điểm cho sự dài dòng).")
    add_bullet("Lớp 2 - Tiêu chuẩn tín hiệu thực chất", "Đo độ phủ required_signals thay vì đếm từ. Câu trả lời dài mà rỗng ý sẽ bị điểm thấp ở cả tiêu chí expected_signal_coverage và clarity_conciseness.")
    add_bullet("Lớp 3 - Chuẩn hóa vector L2 Norm", "TfidfVectorizer chuẩn hóa v_norm = v / ||v||_2, đưa độ dài hình học của vector biểu diễn về bằng 1, triệt tiêu hoàn toàn ưu thế số lượng từ khi đưa vào Ridge Regression.")
    add_bullet("Lớp 4 - 7 mô hình độc lập", "Tách riêng 7 mô hình Ridge Regression giúp mô hình học đúng trọng số ngữ nghĩa đặc thù của từng tiêu chí.")

    # 4. CÔNG THỨC TOÁN HỌC TÍNH ĐIỂM
    add_heading1("4. CÔNG THỨC TOÁN HỌC TÍNH ĐIỂM TỔNG HỢP NĂNG LỰC VÀ OVERALL READINESS")
    add_p("Khi ứng viên kết thúc buổi phỏng vấn tương tác thực tế, module InterviewEvidenceSummaryService tự động tổng hợp bằng chứng từ các lượt đối đáp (turns) thành báo cáo năng lực cuối phiên.")

    add_heading2("4.1. Bước 1: Tính điểm năng lực từng lượt (Turn Competency Score)")
    add_p("Mỗi nhóm năng lực C được cấu thành từ các tiêu chí thành phần trong Rubric với trọng số chuyên biệt w_{C, k}:")
    add_p("Score_turn(C) = [Sum_{k in Applicable} (s_k * w_{C, k})] / [Sum_{k in Applicable} w_{C, k}]", italic=True)
    add_p("Ví dụ trọng số tiêu biểu:")
    add_bullet("Kiến thức kỹ thuật", "technical_correctness (0.40) + expected_signal_coverage (0.25) + reasoning_quality (0.20) + answer_relevance (0.10) + consistency (0.05).")
    add_bullet("Giao tiếp", "clarity_conciseness (0.45) + answer_relevance (0.30) + consistency (0.15) + expected_signal_coverage (0.10).")

    add_heading2("4.2. Bước 2: Tổng hợp điểm nhóm năng lực qua các lượt phỏng vấn")
    add_p("Một nhóm năng lực được tổng hợp từ các lượt hỏi liên quan trực tiếp (trọng số 1.0) hoặc gián tiếp xuyên suốt phiên (trọng số 0.6):")
    add_p("Score(C) = [Sum_{i=1}^{N_C} (Score_{turn, i}(C) * W_{evidence, i})] / [Sum_{i=1}^{N_C} W_{evidence, i}]", italic=True)
    add_p("Công thức tính Độ tin cậy (Confidence):")
    add_p("Confidence(C) = 0.5 * Coverage + 0.3 * Agreement + 0.2 * Quality", italic=True)
    add_bullet("Coverage", "min(1.0, Total_Weight / 2.0) — Mức độ bao phủ số lượng bằng chứng.")
    add_bullet("Agreement", "max(0.0, 1.0 - Standard_Deviation / 2.0) — Độ đồng thuận giữa các câu trả lời.")
    add_bullet("Quality", "Điểm chất lượng dẫn chứng trung bình chuẩn hóa.")
    add_p("Ngưỡng kết luận: Nếu Confidence < 0.6 hoặc chưa có câu hỏi trực tiếp, hệ thống đánh dấu trạng thái 'insufficient_evidence' (Chưa đủ), không phỏng đoán tùy tiện.")

    add_heading2("4.3. Bước 3: Công thức tính Điểm sẵn sàng tổng thể (Overall Readiness Score)")
    add_p("Hệ thống quy định bảng trọng số chuẩn (OVERALL_WEIGHTS) cho 6 nhóm năng lực:")
    table_w = doc.add_table(rows=1, cols=2)
    w_data = [
        ["1. Kiến thức kỹ thuật (technical_knowledge)", "0.20"],
        ["2. Giải quyết vấn đề (problem_solving)", "0.20"],
        ["3. Bằng chứng kinh nghiệm (experience_evidence)", "0.15"],
        ["4. Giao tiếp (communication)", "0.15"],
        ["5. Phù hợp vị trí (role_relevance)", "0.15"],
        ["6. Học hỏi & thích nghi (learning_adaptability)", "0.15"]
    ]
    style_table(table_w, [4.5, 1.8], ["Nhóm năng lực", "Trọng số chuẩn (W_C)"], w_data)

    add_p("Cơ chế Tái chuẩn hóa Trọng số (Re-weighting):", italic=True)
    add_p("Nếu một số năng lực chưa đủ câu hỏi khảo sát, hệ thống KHÔNG tính 0 điểm (không phạt ứng viên), mà chỉ tính trung bình trên tập các năng lực đã có bằng chứng hợp lệ (Available):")
    add_p("Overall Readiness Score = [Sum_{C in Available} (Score(C) * W_C)] / [Sum_{C in Available} W_C]", italic=True)

    add_p("Minh họa phân tích trường hợp thực tế (Đạt điểm 3.7):")
    add_bullet("Kiến thức kỹ thuật", "3.71 (Trọng số 0.20)")
    add_bullet("Bằng chứng kinh nghiệm", "3.40 (Trọng số 0.15)")
    add_bullet("Giao tiếp", "4.03 (Trọng số 0.15)")
    add_bullet("Phù hợp vị trí", "3.62 (Trọng số 0.15)")
    add_bullet("Giải quyết vấn đề & Học hỏi", "Chưa đủ bằng chứng (loại khỏi mẫu số)")
    add_p("1. Tổng trọng số hợp lệ: W_available = 0.20 + 0.15 + 0.15 + 0.15 = 0.65")
    add_p("2. Tổng điểm có trọng số: Tu_so = (3.71 * 0.20) + (3.40 * 0.15) + (4.03 * 0.15) + (3.62 * 0.15) = 0.742 + 0.510 + 0.6045 + 0.543 = 2.3995")
    add_p("3. Điểm tổng thể: Overall Score = 2.3995 / 0.65 = 3.6915...")
    add_p("4. Hàm toFixed(1) làm tròn 1 chữ số thập phân hiển thị ra vòng tròn cam: 3.7.")

    # 5. KIẾN TRÚC LƯU TRỮ SUPABASE
    add_heading1("5. KIẾN TRÚC LƯU TRỮ VÀ THỐNG KÊ DỮ LIỆU TRÊN SUPABASE")
    add_p("Hệ thống NextStep sử dụng chung một cơ sở dữ liệu PostgreSQL trên nền tảng Supabase giữa NestJS Backend và FastAPI AI Server. Dữ liệu được tổ chức chặt chẽ để phục vụ cả huấn luyện mô hình offline lẫn vận hành phỏng vấn trực tiếp online.")

    table_s = doc.add_table(rows=1, cols=3)
    s_data = [
        ["question_benchmark_outputs", "2.000 dòng", "Lưu trữ toàn bộ câu hỏi do các LLM độc lập đề xuất qua 2 đợt V1 và V1.1."],
        ["synthetic_answer_samples", "2.000 dòng", "Lưu trữ câu trả lời nhân tạo (Synthetic Answers) cân bằng đều trên 5 mức chất lượng."],
        ["question_ai_judgments", "6.000 dòng", "Lưu các lượt phán quyết chấm điểm câu hỏi của Hội đồng AI (3 giám khảo / câu)."],
        ["answer_ai_judgments", "6.000 dòng", "Lưu các lượt phán quyết chấm điểm câu trả lời của Hội đồng AI (3 giám khảo / câu)."],
        ["question_benchmark_runs", "2.000 dòng", "Quản lý phiên chạy dữ liệu (balanced-weak-v1: 1.000 runs, balanced-weak-v1.1: 1.000 runs)."],
        ["question_benchmark_cases", "240 dòng", "240 nhóm ngữ cảnh phỏng vấn (CV/JD Context) độc lập làm nguồn phát sinh."],
        ["interview_sessions", "11 dòng", "Lưu trữ các phiên phỏng vấn tương tác thử nghiệm trực tiếp trên web demo."],
        ["interview_turns", "35 dòng", "Lưu chi tiết từng lượt hỏi - đáp thực tế kèm điểm chấm real-time (answer_scores_json)."],
        ["interview_evidence_summaries", "2 dòng", "Lưu kết quả tổng hợp bằng chứng năng lực và điểm Overall Readiness cuối phiên."],
        ["cv_skills", "1.212 dòng", "Dữ liệu kỹ năng trích xuất từ các hồ sơ CV ứng viên."],
        ["skill_gaps", "913 dòng", "Lỗ hổng kỹ năng phân tích giữa ứng viên và yêu cầu tuyển dụng."],
        ["skills", "164 dòng", "Danh mục kỹ năng công nghệ chuẩn hóa trong toàn hệ thống."],
        ["jobs / companies / cvs", "48 / 42 / 44", "Tin tuyển dụng crawl từ TopDev, thông tin công ty và file CV người dùng tải lên."]
    ]
    style_table(table_s, [2.3, 1.2, 3.0], ["Bảng dữ liệu Supabase", "Số dòng thực tế", "Chức năng và Vai trò"], s_data)

    add_p("Ý nghĩa con số '5.000 dữ liệu':", italic=True)
    add_p("Trong mỗi đợt sinh hoàn chỉnh (batch): 1.000 câu hỏi + 1.000 câu trả lời + 3.000 lượt AI chấm chéo = đúng 5.000 bản ghi dữ liệu được tạo ra một cách đồng bộ. Qua 2 đợt hoàn tất (V1 và V1.1), tổng số bản ghi phục vụ nghiên cứu và đánh giá AI trên Supabase hiện nay đã đạt tới hơn 18.000 bản ghi.")

    # 6. HUẤN LUYỆN VÀ HIỆU NĂNG MÔ HÌNH
    add_heading1("6. HUẤN LUYỆN VÀ HIỆU NĂNG HAI MÔ HÌNH AI")
    add_heading2("6.1. Mô hình 1: Question Quality Ranker V2 (Xếp hạng câu hỏi)")
    add_bullet("Thuật toán", "Pairwise Ranking sử dụng Logistic Regression trên hiệu vector đặc trưng TF-IDF (XA - XB).")
    add_bullet("Hiệu năng trên tập Test độc lập", "Top-1 Accuracy đạt 54.05% (vượt trội hơn bản V1 Pointwise +10.81%), MRR đạt 0.731, Pairwise Accuracy đạt 58.20%, và NDCG đạt 0.993.")

    add_heading2("6.2. Mô hình 2: Candidate Answer Scorer (Chấm điểm câu trả lời)")
    add_bullet("Thuật toán", "Hệ thống 7 mô hình hồi quy Ridge Regression (alpha=1.0) độc lập cho 7 tiêu chí trong Rubric, kết hợp chuẩn hóa vector L2 Norm.")
    add_bullet("Hiệu năng trên tập Test độc lập", "MAE tổng thể ~0.80 trên thang 5 điểm; hệ số tương quan Spearman đạt từ 0.57 đến 0.79. Tốc độ suy luận dưới 30 mili-giây trên CPU thông thường.")

    # 7. TỔNG KẾT
    add_heading1("7. TỔNG KẾT VÀ GIÁ TRỊ THỰC TIỄN")
    add_p("1. Tính độc lập và khả thi cao: Đề tài đã xây dựng thành công pipeline Weak Supervision tự sinh và tự gán nhãn dữ liệu mà không cần phụ thuộc vào chuyên gia bên ngoài.")
    add_p("2. Khoa học và Minh bạch: Điểm số đánh giá năng lực của ứng viên (như điểm 3.7 trong phiên demo) có cơ sở toán học rõ ràng, được suy diễn từ bằng chứng cụ thể qua từng câu hỏi thay vì cho điểm cảm tính.")
    add_p("3. Tiết kiệm tài nguyên: Thay thế hoàn toàn việc phải gọi 4 LLM đắt đỏ và chậm chạp ở mỗi lượt hỏi-đáp bằng 2 mô hình máy học nội bộ siêu nhanh và hoàn toàn miễn phí.")
    add_p("4. Lưu trữ toàn vẹn: Cơ sở dữ liệu Supabase được tổ chức đồng bộ, lưu vết toàn bộ quá trình từ sinh mẫu, chấm chéo đến phiên đối đáp thực tế của người dùng.")

    # Save to disk
    output_workspace_path = Path(r"d:\CODE\PBL5\BAO_CAO_QUY_TRINH_SINH_DU_LIEU_VA_CHAM_DIEM_AI.docx")
    output_v2_path = Path(r"d:\CODE\PBL5\BAO_CAO_QUY_TRINH_SINH_DU_LIEU_VA_CHAM_DIEM_AI_V2.docx")
    output_be_path = Path(r"d:\CODE\PBL5\ai_job_server\BAO_CAO_QUY_TRINH_SINH_DU_LIEU_VA_CHAM_DIEM_AI.docx")
    output_artifact_path = Path(r"C:\Users\ASUS-PRO\.gemini\antigravity-ide\brain\d4c1c83c-0572-47e0-8449-181d94cd1017\BAO_CAO_QUY_TRINH_SINH_DU_LIEU_VA_CHAM_DIEM_AI.docx")
    
    saved_path = output_workspace_path
    try:
        doc.save(str(output_workspace_path))
    except PermissionError:
        doc.save(str(output_v2_path))
        saved_path = output_v2_path

    try:
        shutil.copy(str(saved_path), str(output_be_path))
        shutil.copy(str(saved_path), str(output_artifact_path))
    except Exception:
        pass
    print(f"Report exported successfully to: {saved_path}")

if __name__ == "__main__":
    create_report()
