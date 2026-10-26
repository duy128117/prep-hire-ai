const API = "/api/v1/interview";

const elements = {
  setupForm: document.querySelector("#setupForm"),
  cvSelect: document.querySelector("#cvSelect"),
  jobSelect: document.querySelector("#jobSelect"),
  questionCount: document.querySelector("#questionCount"),
  languageSelect: document.querySelector("#languageSelect"),
  externalAi: document.querySelector("#externalAi"),
  startButton: document.querySelector("#startButton"),
  modelHealth: document.querySelector("#modelHealth"),
  sessionTitle: document.querySelector("#sessionTitle"),
  progressLabel: document.querySelector("#progressLabel"),
  progressBar: document.querySelector("#progressBar"),
  notice: document.querySelector("#notice"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#emptyState"),
  answerForm: document.querySelector("#answerForm"),
  answerInput: document.querySelector("#answerInput"),
  sendButton: document.querySelector("#sendButton"),
  summaryPanel: document.querySelector("#summaryPanel"),
  overallScore: document.querySelector("#overallScore"),
  competencyGrid: document.querySelector("#competencyGrid"),
};

const state = {
  session: null,
  options: null,
  busy: false,
};

const competencyLabels = {
  technical_knowledge: "Kiến thức kỹ thuật",
  problem_solving: "Giải quyết vấn đề",
  experience_evidence: "Bằng chứng kinh nghiệm",
  communication: "Giao tiếp",
  role_relevance: "Phù hợp vị trí",
  learning_adaptability: "Học hỏi & thích nghi",
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let body = null;
  try { body = await response.json(); } catch { body = null; }
  if (!response.ok) {
    throw new Error(body?.detail || `Yêu cầu thất bại (${response.status})`);
  }
  return body;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function requestWithRetry(path, attempts = 4) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await request(path);
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await wait(attempt * 900);
    }
  }
  throw lastError;
}

function showNotice(message, type = "info") {
  elements.notice.textContent = message;
  elements.notice.className = `notice ${type === "error" ? "error" : ""}`;
  elements.notice.hidden = !message;
}

function setBusy(value, label) {
  state.busy = value;
  elements.startButton.disabled = value || !elements.cvSelect.value || !elements.jobSelect.value;
  elements.sendButton.disabled = value;
  if (label) showNotice(label);
}

function populateSelect(select, items, valueKey, labelBuilder, placeholder) {
  select.replaceChildren();
  const first = document.createElement("option");
  first.value = "";
  first.textContent = placeholder;
  select.append(first);
  for (const item of items) {
    const option = document.createElement("option");
    option.value = String(item[valueKey]);
    option.textContent = labelBuilder(item);
    select.append(option);
  }
  select.disabled = items.length === 0;
}

async function loadSetup() {
  elements.modelHealth.classList.remove("ready", "error");
  elements.modelHealth.querySelector("span:last-child").textContent = "\u0110ang k\u1ebft n\u1ed1i h\u1ec7 th\u1ed1ng...";
  try {
    const [options, health] = await Promise.all([
      requestWithRetry(`${API}/demo/options`),
      requestWithRetry(`${API}/realtime-models/status`),
    ]);
    state.options = options;
    populateSelect(
      elements.cvSelect,
      options.cvs,
      "cv_id",
      (item) => `${item.file_name}${item.fallback_ready ? " · sẵn sàng" : ""}`,
      `Chọn một CV (${options.cvs.length} mẫu)`,
    );
    populateSelect(
      elements.jobSelect,
      options.jobs,
      "job_id",
      (item) => [item.title, item.company_name, item.level].filter(Boolean).join(" · "),
      `Chọn một Job (${options.jobs.length} mẫu)`,
    );
    const ready = health.question_ranker.ready && health.answer_scorer.ready;
    elements.modelHealth.classList.toggle("ready", ready);
    elements.modelHealth.classList.toggle("error", !ready);
    elements.modelHealth.querySelector("span:last-child").textContent = ready
      ? `${health.configured_providers.length} AI · 2 model sẵn sàng`
      : "Model chưa sẵn sàng";
  } catch (error) {
    elements.modelHealth.classList.add("error");
    elements.modelHealth.querySelector("span:last-child").textContent = "Không tải được hệ thống";
    showNotice(error.message, "error");
  }
}

function addMessage(kind, text, meta = [], customLabel = null) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${kind}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = customLabel || (kind === "ai" ? "AI Interviewer" : "Bạn");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.append(label, bubble);
  if (meta.length) {
    const metaRow = document.createElement("div");
    metaRow.className = "question-meta";
    for (const value of meta) {
      const chip = document.createElement("span");
      chip.textContent = value;
      metaRow.append(chip);
    }
    wrapper.append(metaRow);
  }
  elements.conversation.append(wrapper);
}

function addCandidateReview(turn) {
  const candidates = turn.question_candidates || [];
  if (candidates.length <= 1) return;

  const review = document.createElement("details");
  review.className = "candidate-review";
  review.open = !turn.answer;

  const summary = document.createElement("summary");
  const summaryTitle = document.createElement("span");
  summaryTitle.textContent = `${candidates.length} c\u00e2u h\u1ecfi do c\u00e1c AI \u0111\u1ec1 xu\u1ea5t`;
  const ranker = document.createElement("small");
  ranker.textContent = turn.question_ranker_version || "Question Ranker";
  summary.append(summaryTitle, ranker);

  const grid = document.createElement("div");
  grid.className = "candidate-grid";
  for (const [index, candidate] of candidates.entries()) {
    const card = document.createElement("article");
    card.className = `candidate-card${candidate.selected ? " selected" : ""}`;

    const head = document.createElement("div");
    head.className = "candidate-head";
    const provider = document.createElement("strong");
    provider.textContent = `${index + 1}. ${candidate.provider_name || "AI"}`;
    const badge = document.createElement("span");
    badge.textContent = candidate.selected
      ? "RANKER CH\u1eccN"
      : "\u0110\u1ec0 XU\u1ea4T";
    head.append(provider, badge);

    const question = document.createElement("p");
    question.textContent = candidate.question;

    const meta = document.createElement("div");
    meta.className = "candidate-card-meta";
    const utility = typeof candidate.pairwise_utility === "number"
      ? candidate.pairwise_utility.toFixed(3)
      : "N/A";
    meta.textContent = `utility ${utility} \u00b7 ${candidate.generation_latency_ms || 0} ms`;
    card.append(head, question, meta);
    grid.append(card);
  }

  const selected = candidates.find((candidate) => candidate.selected);
  const decision = document.createElement("div");
  decision.className = "ranker-decision";
  decision.textContent = selected
    ? `Ranker ch\u1ecdn: ${selected.provider_name || "AI"} \u00b7 utility ${Number(selected.pairwise_utility || 0).toFixed(3)}`
    : "Ranker kh\u00f4ng tr\u1ea3 v\u1ec1 quy\u1ebft \u0111\u1ecbnh.";

  review.append(summary, grid, decision);
  elements.conversation.append(review);
}

function questionMeta(turn) {
  const values = [];
  values.push(turn.stage.replaceAll("_", " "));
  if (turn.question_candidate_count > 1) values.push(`${turn.question_candidate_count} câu cạnh tranh`);
  if (turn.provider_name) values.push(turn.provider_name);
  if (turn.selected_question_score !== null) values.push(`utility ${turn.selected_question_score.toFixed(3)}`);
  values.push(`${turn.question_selection_latency_ms} ms`);
  return values;
}

function selectedLabel() {
  const cv = state.options?.cvs.find((item) => item.cv_id === Number(elements.cvSelect.value));
  const job = state.options?.jobs.find((item) => item.job_id === Number(elements.jobSelect.value));
  return { cv, job };
}

function renderSession(session) {
  state.session = session;
  elements.emptyState.hidden = true;
  elements.conversation.replaceChildren();
  for (const turn of session.turns) {
    addCandidateReview(turn);
    addMessage(
      "ai",
      turn.question,
      questionMeta(turn),
      turn.question_candidates?.length > 1
        ? "C\u00e2u h\u1ecfi \u0111\u01b0\u1ee3c Ranker ch\u1ecdn"
        : "AI Interviewer",
    );
    if (turn.answer) addMessage("user", turn.answer);
  }
  const answered = session.turns.filter((turn) => Boolean(turn.answer)).length;
  const visibleTurn = session.status === "completed" ? session.max_questions : Math.min(answered + 1, session.max_questions);
  elements.progressLabel.textContent = `${visibleTurn} / ${session.max_questions}`;
  elements.progressBar.style.width = `${Math.round((answered / session.max_questions) * 100)}%`;
  const labels = selectedLabel();
  elements.sessionTitle.textContent = labels.job?.title || "Phỏng vấn đang diễn ra";
  elements.answerForm.hidden = session.status !== "active";
  if (session.status === "active") {
    elements.answerInput.value = "";
    elements.answerInput.focus();
  } else {
    elements.progressBar.style.width = "100%";
    showNotice("Phiên phỏng vấn đã hoàn tất. Đang tổng hợp bằng chứng...");
    loadSummary(session.session_id);
  }
  requestAnimationFrame(() => {
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
  });
}

async function startInterview(event) {
  event.preventDefault();
  if (state.busy) return;
  const useMultipleAi = elements.externalAi.checked;
  setBusy(
    true,
    useMultipleAi
      ? "Đang đọc CV/JD và chờ các AI tạo câu hỏi cạnh tranh..."
      : "Đang đọc CV/JD và tạo câu hỏi fallback...",
  );
  elements.summaryPanel.hidden = true;
  try {
    const session = await request(`${API}/demo/sessions`, {
      method: "POST",
      body: JSON.stringify({
        cv_id: Number(elements.cvSelect.value),
        job_id: Number(elements.jobSelect.value),
        language: elements.languageSelect.value,
        max_questions: Number(elements.questionCount.value),
        allow_external_ai: useMultipleAi,
      }),
    });
    showNotice("");
    renderSession(session);
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function submitAnswer(event) {
  event.preventDefault();
  if (state.busy || !state.session) return;
  const answer = elements.answerInput.value.trim();
  if (answer.length < 2) return;
  setBusy(true, "Đang chấm bằng chứng và chuẩn bị câu hỏi tiếp theo...");
  try {
    const session = await request(`${API}/sessions/${state.session.session_id}/answers`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
    showNotice("");
    renderSession(session);
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function loadSummary(sessionId) {
  try {
    let summary;
    try {
      summary = await request(`${API}/sessions/${sessionId}/evidence-summary`);
    } catch {
      summary = await request(`${API}/sessions/${sessionId}/evidence-summary`, { method: "POST" });
    }
    renderSummary(summary);
    showNotice("Hoàn tất. Kết quả chỉ phản ánh bằng chứng thu được trong phiên này.");
  } catch (error) {
    showNotice(`Không tải được tổng hợp: ${error.message}`, "error");
  }
}

function renderSummary(summary) {
  const overall = summary.competencies.overall_readiness || {};
  elements.overallScore.textContent = overall.score == null ? "—" : overall.score.toFixed(1);
  elements.competencyGrid.replaceChildren();
  for (const [key, label] of Object.entries(competencyLabels)) {
    const value = summary.competencies[key] || {};
    const card = document.createElement("article");
    card.className = "competency-card";
    const title = document.createElement("span");
    title.textContent = label;
    const score = document.createElement("strong");
    score.textContent = value.score == null ? "Chưa đủ" : value.score.toFixed(2);
    const status = document.createElement("small");
    status.textContent = `${value.status || "unknown"} · tin cậy ${Math.round((value.confidence || 0) * 100)}%`;
    card.append(title, score, status);
    elements.competencyGrid.append(card);
  }
  elements.summaryPanel.hidden = false;
}

function updateStartState() {
  elements.startButton.disabled = state.busy || !elements.cvSelect.value || !elements.jobSelect.value;
}

elements.setupForm.addEventListener("submit", startInterview);
elements.answerForm.addEventListener("submit", submitAnswer);
elements.cvSelect.addEventListener("change", updateStartState);
elements.jobSelect.addEventListener("change", updateStartState);
elements.answerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.answerForm.requestSubmit();
  }
});
window.addEventListener("online", loadSetup);

loadSetup();
