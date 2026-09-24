/**
 * VerifyAI — Fake News Detection Frontend
 * Handles API communication, UI interactions, and result rendering.
 */

const API_BASE = "http://localhost:5000/api";

// ── DOM Elements ────────────────────────────────────────────
const newsInput = document.getElementById("news-input");
const charCount = document.getElementById("char-count");
const modelSelect = document.getElementById("model-select");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const retryBtn = document.getElementById("retry-btn");

const loadingState = document.getElementById("loading-state");
const resultsSection = document.getElementById("results-section");
const singleResult = document.getElementById("single-result");
const allResults = document.getElementById("all-results");
const errorState = document.getElementById("error-state");
const errorText = document.getElementById("error-text");

const modelsGrid = document.getElementById("models-grid");
const statusDot = document.querySelector(".status-dot");
const statusText = document.querySelector(".status-text");

// ── Character Counter ───────────────────────────────────────
newsInput.addEventListener("input", () => {
    const len = newsInput.value.length;
    charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? "s" : ""}`;
});

// ── Clear Button ────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
    newsInput.value = "";
    charCount.textContent = "0 characters";
    hideAll();
    newsInput.focus();
});

// ── Analyze Button ──────────────────────────────────────────
analyzeBtn.addEventListener("click", analyze);
retryBtn.addEventListener("click", analyze);

// Allow Ctrl+Enter to submit
newsInput.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
        analyze();
    }
});

async function analyze() {
    const text = newsInput.value.trim();
    if (!text) {
        shakeElement(newsInput);
        newsInput.focus();
        return;
    }

    const selectedModel = modelSelect.value;

    // Show loading
    hideAll();
    loadingState.classList.remove("hidden");
    analyzeBtn.disabled = true;

    try {
        let result;
        if (selectedModel === "all") {
            result = await fetchAPI("/predict-all", { text });
            renderAllResults(result);
        } else {
            result = await fetchAPI("/predict", { text, model: selectedModel });
            renderSingleResult(result);
        }
    } catch (err) {
        showError(err.message || "Failed to connect to API. Is the backend running?");
    } finally {
        analyzeBtn.disabled = false;
    }
}

// ── API Helper ──────────────────────────────────────────────
async function fetchAPI(endpoint, body) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `API error: ${res.status}`);
    }

    return res.json();
}

// ── Render Single Result ────────────────────────────────────
function renderSingleResult(data) {
    hideAll();
    resultsSection.classList.remove("hidden");
    singleResult.classList.remove("hidden");

    const verdict = document.getElementById("result-verdict");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictLabel = document.getElementById("verdict-label");
    const verdictModel = document.getElementById("verdict-model");
    const confidenceValue = document.getElementById("confidence-value");
    const ringFill = document.getElementById("ring-fill");
    const details = document.getElementById("result-details");

    const isFake = data.label === "FAKE";
    verdict.className = `result-verdict ${isFake ? "fake" : "real"}`;
    verdictIcon.innerHTML = isFake
        ? '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M8 8l12 12M20 8L8 20" stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>'
        : '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M7 14l5 5 9-9" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    verdictIcon.style.color = isFake ? "var(--danger)" : "var(--success)";

    verdictLabel.textContent = data.label;
    verdictModel.textContent = data.model;

    // Confidence ring animation
    const confidence = data.confidence || 0;
    const percent = Math.round(confidence * 100);
    confidenceValue.textContent = `${percent}%`;

    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (confidence * circumference);
    ringFill.style.strokeDasharray = circumference;
    // Reset then animate
    ringFill.style.transition = "none";
    ringFill.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            ringFill.style.transition = "stroke-dashoffset 1s ease";
            ringFill.style.strokeDashoffset = offset;
        });
    });

    // Detail chips
    details.innerHTML = "";
    const chips = [
        { label: "Text Length", value: data.text_length?.toLocaleString() || "—" },
        { label: "Processed", value: data.cleaned_length?.toLocaleString() || "—" },
    ];

    if (data.probabilities) {
        chips.push(
            { label: "P(Fake)", value: `${(data.probabilities.fake * 100).toFixed(1)}%` },
            { label: "P(Real)", value: `${(data.probabilities.real * 100).toFixed(1)}%` }
        );
    }

    chips.forEach((c) => {
        const chip = document.createElement("div");
        chip.className = "detail-chip";
        chip.innerHTML = `<span class="label">${c.label}</span><span class="value">${c.value}</span>`;
        details.appendChild(chip);
    });
}

// ── Render All Results ──────────────────────────────────────
function renderAllResults(data) {
    hideAll();
    resultsSection.classList.remove("hidden");
    allResults.classList.remove("hidden");

    const grid = document.getElementById("results-grid");
    grid.innerHTML = "";

    data.results.forEach((r) => {
        const isFake = r.label === "FAKE";
        const card = document.createElement("div");
        card.className = `model-result-card ${isFake ? "fake" : "real"}`;

        const confidence = r.confidence ? `${(r.confidence * 100).toFixed(1)}%` : "N/A";

        card.innerHTML = `
            <div class="card-model-name">${r.model}</div>
            <div class="card-label">${r.label}</div>
            <div class="card-confidence">Confidence: <strong>${confidence}</strong></div>
        `;
        grid.appendChild(card);
    });
}

// ── Error / Hide Helpers ────────────────────────────────────
function showError(message) {
    hideAll();
    errorState.classList.remove("hidden");
    errorText.textContent = message;
}

function hideAll() {
    loadingState.classList.add("hidden");
    resultsSection.classList.add("hidden");
    singleResult.classList.add("hidden");
    allResults.classList.add("hidden");
    errorState.classList.add("hidden");
}

function shakeElement(el) {
    el.style.animation = "none";
    el.offsetHeight; // trigger reflow
    el.style.animation = "shake 0.4s ease";
    el.style.borderColor = "var(--danger)";
    setTimeout(() => {
        el.style.borderColor = "";
        el.style.animation = "";
    }, 600);
}

// Add shake keyframes dynamically
const shakeStyle = document.createElement("style");
shakeStyle.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-6px); }
        40% { transform: translateX(6px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
`;
document.head.appendChild(shakeStyle);

// ── Load Model Performance ──────────────────────────────────
async function loadModelPerformance() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) throw new Error("Failed to load models");

        const models = await res.json();
        renderModelCards(models);
        setStatus(true);
    } catch (err) {
        setStatus(false);
        // Show fallback data
        renderModelCards([
            { key: "random_forest", name: "Random Forest", accuracy: 0.9982, precision: 0.9974, recall: 0.9989, f1_score: 0.9981 },
            { key: "svm", name: "SVM (LinearSVC)", accuracy: 0.9963, precision: 0.9960, recall: 0.9962, f1_score: 0.9961 },
            { key: "logistic", name: "Logistic Regression", accuracy: 0.9914, precision: 0.9871, recall: 0.9948, f1_score: 0.9910 },
            { key: "naive_bayes", name: "Naive Bayes", accuracy: 0.9501, precision: 0.9435, recall: 0.9519, f1_score: 0.9477 },
        ]);
    }
}

function renderModelCards(models) {
    modelsGrid.innerHTML = "";

    models.forEach((m, i) => {
        const isBest = i === 0;
        const card = document.createElement("div");
        card.className = `model-perf-card${isBest ? " best" : ""}`;

        const metrics = [
            { label: "Accuracy", value: m.accuracy },
            { label: "Precision", value: m.precision },
            { label: "Recall", value: m.recall },
            { label: "F1-Score", value: m.f1_score },
        ];

        card.innerHTML = `
            ${isBest ? '<span class="perf-badge best-badge">Best Model</span>' : ""}
            <div class="perf-model-name">${m.name}</div>
            <div class="perf-metrics">
                ${metrics
                    .map(
                        (metric) => `
                    <div class="perf-metric">
                        <span class="perf-metric-label">${metric.label}</span>
                        <div class="perf-metric-bar">
                            <div class="perf-metric-bar-fill" style="width: 0%" data-width="${((metric.value || 0) * 100).toFixed(1)}%"></div>
                        </div>
                        <span class="perf-metric-value">${metric.value ? (metric.value * 100).toFixed(2) + "%" : "—"}</span>
                    </div>`
                    )
                    .join("")}
            </div>
        `;
        modelsGrid.appendChild(card);
    });

    // Animate bars after render
    requestAnimationFrame(() => {
        document.querySelectorAll(".perf-metric-bar-fill").forEach((bar) => {
            bar.style.width = bar.dataset.width;
        });
    });
}

function setStatus(connected) {
    if (connected) {
        statusDot.classList.remove("disconnected");
        statusText.textContent = "API Connected";
    } else {
        statusDot.classList.add("disconnected");
        statusText.textContent = "API Offline";
    }
}

// ── Smooth Scroll for Nav Links ─────────────────────────────
document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
        const href = link.getAttribute("href");
        if (href.startsWith("#")) {
            e.preventDefault();
            document.querySelector(href)?.scrollIntoView({ behavior: "smooth" });
            document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
            link.classList.add("active");
        }
    });
});

// ── Initialize ──────────────────────────────────────────────
loadModelPerformance();
