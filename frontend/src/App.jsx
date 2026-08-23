import "./App.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowUp,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  GitBranch,
  LogOut,
  Menu,
  MessageSquare,
  Network,
  Plus,
  Search,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  User,
  X,
  Zap,
} from "lucide-react";


const V4_BENCHMARK = {
  queries: 1000,
  methods: {
    "BM25": { precision: 0.0806, recall: 0.3768499030, mrr: 0.3240269850, ndcg: 0.2968246628, latency: 1.5912427475 },
    "Dense": { precision: 0.1136, recall: 0.5323211471, mrr: 0.4119190476, ndcg: 0.4019954995, latency: 0.0645039609 },
    "Fixed Hybrid": { precision: 0.1111, recall: 0.5189062682, mrr: 0.4357845238, ndcg: 0.4073481337, latency: 1.6557467084 },
    "Adaptive Hybrid": { precision: 0.1148, recall: 0.5381200002, mrr: 0.4340384921, ndcg: 0.4158529424, latency: 1.6557467084 },
  },
  improvement: {
    weightChanged: 1000, weightChangedPercent: 100,
    mrrImproved: 138, mrrEqual: 768, mrrDecreased: 94,
    ndcgImproved: 209, ndcgEqual: 675, ndcgDecreased: 116,
    precisionImproved: 67, precisionEqual: 904, precisionDecreased: 29,
    recallImproved: 67, recallEqual: 904, recallDecreased: 29,
  },
  domains: [
    ["Argumentation", 0.0792, 0.7920, 0.2498603175, 0.3799806967],
    ["Finance", 0.0984, 0.4467904762, 0.4390746032, 0.3696645031],
    ["Medical", 0.1960, 0.1344895247, 0.4302619048, 0.2610803261],
    ["Scientific", 0.0856, 0.7792, 0.6169571429, 0.6526862436],
  ],
};

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function getStoredToken() {
  return localStorage.getItem("ahr_token");
}

function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem("ahr_user") || "null");
  } catch {
    return null;
  }
}

async function apiRequest(path, options = {}) {
  const token = getStoredToken();

  const headers = {
    ...(options.headers || {}),
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  if (
    options.body &&
    !(options.body instanceof FormData) &&
    !headers["Content-Type"]
  ) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(apiUrl(path), {
    ...options,
    headers,
  });

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        `Request failed with status ${response.status}`
    );
  }

  return data;
}

function formatTime(value) {
  if (!value) return "";

  try {
    return new Date(value).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function formatDate(value) {
  if (!value) return "";

  try {
    return new Date(value).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return "";
  }
}

function formatMs(value) {
  if (value === undefined || value === null) return "—";

  const number = Number(value);

  if (!Number.isFinite(number)) return "—";

  if (number < 1000) {
    return `${number.toFixed(1)}ms`;
  }

  return `${(number / 1000).toFixed(2)}s`;
}

function formatPercent(value) {
  if (value === undefined || value === null) return "—";

  const number = Number(value);

  if (!Number.isFinite(number)) return "—";

  return `${(number * 100).toFixed(1)}%`;
}

function getGenerationMs(rag) {
  const metrics = rag?.retrieval_metrics || {};
  const candidates = [
    rag?.latency?.generation_ms,
    metrics.generation_ms,
    metrics.generation_latency_ms,
    metrics.generation_time_ms,
    rag?.generation_ms,
    rag?.generation_latency_ms,
    rag?.generation_time_ms,
    rag?.generation_metrics?.generation_ms,
    rag?.generation_metrics?.latency_ms,
    rag?.generation?.generation_ms,
    rag?.generation?.latency_ms,
    rag?.timings?.generation_ms,
  ];

  for (const value of candidates) {
    const number = Number(value);
    if (Number.isFinite(number) && number > 0) {
      return number;
    }
  }

  // Some backend responses expose only total runtime and retrieval runtime.
  // Use their positive difference as a fallback instead of displaying 0ms.
  const totalMs = Number(
    rag?.latency?.total_ms ??
    metrics.total_ms ??
      rag?.total_ms ??
      rag?.timings?.total_ms
  );
  const retrievalMs = Number(
    rag?.latency?.retrieval_ms ??
    metrics.total_retrieval_ms ??
      rag?.retrieval_ms ??
      rag?.timings?.retrieval_ms
  );

  if (
    Number.isFinite(totalMs) &&
    Number.isFinite(retrievalMs) &&
    totalMs > retrievalMs
  ) {
    return totalMs - retrievalMs;
  }

  return null;
}

function formatMetricValue(value, key) {
  const number = Number(value);

  if (!Number.isFinite(number)) return "—";

  return key === "latency"
    ? `${number.toFixed(3)}s`
    : number.toFixed(4);
}

function formatImprovement(baseline, adaptive) {
  const base = Number(baseline);
  const next = Number(adaptive);

  if (!Number.isFinite(base) || !Number.isFinite(next) || base === 0) {
    return { text: "—", positive: false, negative: false };
  }

  const percent = ((next - base) / Math.abs(base)) * 100;

  return {
    text: `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%`,
    positive: percent > 0,
    negative: percent < 0,
  };
}

function initials(name = "User") {
  return (
    name
      .split(" ")
      .map((part) => part[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "U"
  );
}

function renderMarkdown(text = "") {
  const lines = text.split("\n");

  return (
    <div className="markdown-content">
      {lines.map((line, index) => {
        if (!line.trim()) {
          return <div key={index} className="md-space" />;
        }

        if (line.startsWith("### ")) {
          return (
            <h4 key={index}>
              {line.replace("### ", "")}
            </h4>
          );
        }

        if (line.startsWith("## ")) {
          return (
            <h3 key={index}>
              {line.replace("## ", "")}
            </h3>
          );
        }

        if (line.startsWith("# ")) {
          return (
            <h2 key={index}>
              {line.replace("# ", "")}
            </h2>
          );
        }

        if (/^\d+\.\s/.test(line)) {
          return (
            <div className="md-number" key={index}>
              {line}
            </div>
          );
        }

        if (line.startsWith("- ")) {
          return (
            <div className="md-bullet" key={index}>
              <span>•</span>
              <span>{line.slice(2)}</span>
            </div>
          );
        }

        const parts = line.split(/(\*\*.*?\*\*)/g);

        return (
          <p key={index}>
            {parts.map((part, partIndex) => {
              if (
                part.startsWith("**") &&
                part.endsWith("**")
              ) {
                return (
                  <strong key={partIndex}>
                    {part.slice(2, -2)}
                  </strong>
                );
              }

              return <React.Fragment key={partIndex}>{part}</React.Fragment>;
            })}
          </p>
        );
      })}
    </div>
  );
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();

    setError("");
    setLoading(true);

    try {
      let data;

      if (mode === "login") {
        data = await apiRequest("/auth/login", {
          method: "POST",
          body: JSON.stringify({
            username_or_email: usernameOrEmail,
            password,
          }),
        });
      } else {
        data = await apiRequest("/auth/register", {
          method: "POST",
          body: JSON.stringify({
            username,
            email,
            password,
          }),
        });
      }

      localStorage.setItem("ahr_token", data.token);
      localStorage.setItem(
        "ahr_user",
        JSON.stringify(data.user)
      );

      onAuthenticated(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-background-orb orb-one" />
      <div className="auth-background-orb orb-two" />

      <div className="auth-card">
        <div className="auth-brand">
          <div className="brand-icon large">
            <Sparkles size={25} />
          </div>

          <div>
            <h1>Adaptive Hybrid RAG</h1>
            <p>Document Intelligence</p>
          </div>
        </div>

        <div className="auth-heading">
          <h2>
            {mode === "login"
              ? "Welcome back"
              : "Create your account"}
          </h2>

          <p>
            {mode === "login"
              ? "Sign in to continue to your private document workspace."
              : "Create an account to keep your documents and conversations private."}
          </p>
        </div>

        <form onSubmit={submit} className="auth-form">
          {mode === "register" && (
            <>
              <label>
                Username
                <div className="input-wrap">
                  <User size={17} />
                  <input
                    value={username}
                    onChange={(e) =>
                      setUsername(e.target.value)
                    }
                    placeholder="Choose username"
                    required
                  />
                </div>
              </label>

              <label>
                Email
                <div className="input-wrap">
                  <MessageSquare size={17} />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) =>
                      setEmail(e.target.value)
                    }
                    placeholder="you@example.com"
                    required
                  />
                </div>
              </label>
            </>
          )}

          {mode === "login" && (
            <label>
              Username or email
              <div className="input-wrap">
                <User size={17} />
                <input
                  value={usernameOrEmail}
                  onChange={(e) =>
                    setUsernameOrEmail(e.target.value)
                  }
                  placeholder="paras3"
                  required
                />
              </div>
            </label>
          )}

          <label>
            Password
            <div className="input-wrap">
              <Settings2 size={17} />
              <input
                type="password"
                value={password}
                onChange={(e) =>
                  setPassword(e.target.value)
                }
                placeholder="Minimum 8 characters"
                minLength={8}
                required
              />
            </div>
          </label>

          {error && (
            <div className="auth-error">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="primary-button auth-submit"
            disabled={loading}
          >
            {loading ? (
              "Please wait..."
            ) : (
              <>
                <span>{mode === "login" ? "Sign in" : "Create account"}</span>
                <ChevronRight size={17} aria-hidden="true" />
              </>
            )}
          </button>
        </form>

        <div className="auth-switch">
          {mode === "login" ? (
            <>
              Don't have an account?
              <button
                onClick={() => {
                  setMode("register");
                  setError("");
                }}
              >
                Create one
              </button>
            </>
          ) : (
            <>
              Already have an account?
              <button
                onClick={() => {
                  setMode("login");
                  setError("");
                }}
              >
                Sign in
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  subValue,
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>

      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <strong>{value}</strong>

        {subValue && (
          <span className="metric-sub">
            {subValue}
          </span>
        )}
      </div>
    </div>
  );
}

function PipelineStep({
  title,
  value,
  description,
  icon,
  active,
  last,
}) {
  return (
    <div className="pipeline-step-wrap">
      <div
        className={`pipeline-step ${
          active ? "active" : ""
        }`}
      >
        <div className="pipeline-step-icon">
          {icon}
        </div>

        <div className="pipeline-step-content">
          <strong>{title}</strong>

          {value && (
            <span className="pipeline-value">
              {value}
            </span>
          )}

          {description && (
            <small>{description}</small>
          )}
        </div>
      </div>

      {!last && (
        <div className="pipeline-arrow">
          <ChevronRight size={17} />
        </div>
      )}
    </div>
  );
}

function ResearchDashboard({ rag }) {
  const [open, setOpen] = useState(true);

  const retrievalMetrics =
    rag?.retrieval_metrics || {};

  const retrievalConfig =
    rag?.retrieval_config || {};

  const models = rag?.models || {};

  const reranking =
    rag?.reranking || {};

  const context =
    rag?.context || {};

  const bm25 = Number(rag?.weights?.bm25 || 0);
  const dense = Number(rag?.weights?.dense || 0);

  const totalRetrieval =
    Number(retrievalMetrics.total_retrieval_ms || 0);

  const generation = Number(
  rag?.latency?.generation_ms ??
  rag?.retrieval_metrics?.generation_ms ??
  0
);

  const total =
    Number(
      retrievalMetrics.total_ms ||
        totalRetrieval + generation
    );

  const pipelineHasData = Boolean(rag);

  if (!open) {
    return (
      <button
        className="research-collapsed"
        onClick={() => setOpen(true)}
      >
        <div className="research-title-icon">
          <GitBranch size={20} />
        </div>

        <div>
          <strong>
            Research & Implementation
          </strong>

          <span>
            Inspect retrieval pipeline and live measurements
          </span>
        </div>

        <ChevronDown size={19} />
      </button>
    );
  }

  return (
    <section className="research-dashboard">
      <div className="research-header">
        <div className="research-header-left">
          <div className="research-title-icon">
            <GitBranch size={20} />
          </div>

          <div>
            <h2>Research & Implementation</h2>
            <p>
              Inspect the adaptive retrieval pipeline,
              model configuration and live measurements.
            </p>
          </div>
        </div>

        <button
          className="collapse-button"
          onClick={() => setOpen(false)}
        >
          Close
          <ChevronDown size={16} />
        </button>
      </div>

      {!pipelineHasData ? (
        <div className="dashboard-empty">
          <Activity size={30} />

          <div>
            <strong>No live execution yet</strong>
            <p>
              Send a question to populate the research
              dashboard with actual retrieval and generation
              measurements.
            </p>
          </div>
        </div>
      ) : (
        <>
          <div className="metric-grid">
            <MetricCard
              icon={<Search size={20} />}
              label="BM25 weight"
              value={formatPercent(bm25)}
            />

            <MetricCard
              icon={<BrainCircuit size={20} />}
              label="Dense weight"
              value={formatPercent(dense)}
            />

            <MetricCard
              icon={<Database size={20} />}
              label="Retrieved chunks"
              value={
                retrievalConfig.final_k ??
                rag?.documents?.length ??
                "—"
              }
            />

            <MetricCard
              icon={<Activity size={20} />}
              label="Retrieval"
              value={formatMs(totalRetrieval)}
            />

            <MetricCard
              icon={<BarChart3 size={20} />}
              label="Reranking"
              value={formatMs(reranking.latency_ms)}
              subValue={
                reranking.applied
                  ? `${reranking.reranked_count || 0} reranked`
                  : "Not applied"
              }
            />

            <MetricCard
              icon={<Sparkles size={20} />}
              label="Generation"
              value={formatMs(generation)}
            />

            <MetricCard
              icon={<Zap size={20} />}
              label="Total latency"
              value={formatMs(total)}
            />

            <MetricCard
              icon={<BrainCircuit size={20} />}
              label="Generation model"
              value={models.generation || "—"}
            />
          </div>

          <div className="dashboard-grid">
            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">
                    ACTUAL RETRIEVAL MEASUREMENTS
                  </span>

                  <h3>
                    Latest execution
                  </h3>
                </div>

                <Activity size={20} />
              </div>

              <div className="measurement-list">
                <MeasurementRow
                  label="BM25 search"
                  value={formatMs(
                    retrievalMetrics.bm25_ms
                  )}
                />

                <MeasurementRow
                  label="Dense search"
                  value={formatMs(
                    retrievalMetrics.dense_ms
                  )}
                />

                <MeasurementRow
                  label="Feature extraction"
                  value={formatMs(
                    retrievalMetrics.feature_extraction_ms
                  )}
                />

                <MeasurementRow
                  label="Adaptive prediction"
                  value={formatMs(
                    retrievalMetrics.adaptive_prediction_ms
                  )}
                />

                <MeasurementRow
                  label="Fusion"
                  value={formatMs(
                    retrievalMetrics.fusion_ms
                  )}
                />

                <MeasurementRow
                  label="Other retrieval overhead"
                  value={formatMs(
                    Math.max(
                      0,
                      totalRetrieval -
                        Number(
                          retrievalMetrics.bm25_ms || 0
                        ) -
                        Number(
                          retrievalMetrics.dense_ms || 0
                        ) -
                        Number(
                          retrievalMetrics.feature_extraction_ms ||
                            0
                        ) -
                        Number(
                          retrievalMetrics.adaptive_prediction_ms ||
                            0
                        ) -
                        Number(
                          retrievalMetrics.fusion_ms || 0
                        )
                    )
                  )}
                />

                <div className="measurement-total">
                  <span>Total retrieval</span>
                  <strong>
                    {formatMs(totalRetrieval)}
                  </strong>
                </div>
              </div>
            </div>

            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">
                    CANDIDATE FLOW
                  </span>

                  <h3>
                    Retrieval stages
                  </h3>
                </div>

                <Network size={20} />
              </div>

              <div className="candidate-flow">
                <CandidateBox
                  title="BM25"
                  value={
                    retrievalConfig.bm25_candidates ??
                    retrievalConfig.candidate_k ??
                    "—"
                  }
                />

                <CandidateBox
                  title="DENSE"
                  value={
                    retrievalConfig.dense_candidates ??
                    retrievalConfig.candidate_k ??
                    "—"
                  }
                />

                <CandidateBox
                  title="FUSED"
                  value={
                    retrievalConfig.fused_candidates ??
                    "—"
                  }
                />

                <CandidateBox
                  title="FINAL"
                  value={
                    retrievalConfig.final_k ??
                    "—"
                  }
                />
              </div>

              <div className="model-info">
                <ModelLine
                  label="Embedding"
                  value={models.embedding}
                />

                <ModelLine
                  label="Adaptive predictor"
                  value={models.adaptive_predictor}
                />

                <ModelLine
                  label="Reranker"
                  value={models.reranker}
                />

                <ModelLine
                  label="Generation"
                  value={models.generation}
                />

                <ModelLine
                  label="Reranking"
                  value={
                    reranking.applied
                      ? "Applied"
                      : "Not applied"
                  }
                />
              </div>
            </div>
          </div>

          <div className="performance-observation">
            <div className="observation-icon">
              <Activity size={19} />
            </div>

            <div>
              <strong>
                Current performance observation
              </strong>

              <span>
                Retrieval completed in{" "}
                <b>{formatMs(totalRetrieval)}</b>, while
                generation took{" "}
                <b>{formatMs(generation)}</b>.
                {generation == null
                  ? " Generation timing was not included in the latest backend response."
                  : generation > totalRetrieval
                    ? " The current runtime bottleneck is local LLM generation."
                    : " Retrieval currently represents the larger portion of runtime."}
              </span>
            </div>
          </div>

          <div className="dashboard-grid lower-grid">
            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">
                    ADAPTIVE RETRIEVAL DISTRIBUTION
                  </span>

                  <h3>
                    Current query weights
                  </h3>
                </div>

                <BarChart3 size={20} />
              </div>

              <WeightBar
                label="BM25"
                value={bm25}
              />

              <WeightBar
                label="Dense"
                value={dense}
              />

              <div className="dashboard-note">
                <Check size={16} />

                <span>
                  Values shown here come from the latest
                  RAG execution response.
                </span>
              </div>
            </div>

            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">
                    EXPERIMENTAL EVALUATION
                  </span>

                  <h3>
                    Baseline vs Adaptive RAG
                  </h3>
                </div>

                <BarChart3 size={20} />
              </div>

              <div className="evaluation-table comparison-table">
                <div className="evaluation-head">
                  <span>METRIC</span>
                  <span>BASELINE</span>
                  <span>ADAPTIVE</span>
                  <span>IMPROVEMENT</span>
                </div>

                {[
                  ["Precision@10", "precision"],
                  ["Recall@10", "recall"],
                  ["MRR", "mrr"],
                  ["NDCG@10", "ndcg"],
                  ["Latency", "latency"],
                ].map(([label, key]) => {
                  const baseline =
                    V4_BENCHMARK.methods["Fixed Hybrid"][key];
                  const adaptive =
                    V4_BENCHMARK.methods["Adaptive Hybrid"][key];
                  const improvement =
                    formatImprovement(baseline, adaptive);

                  return (
                    <div className="evaluation-row" key={label}>
                      <span>{label}</span>
                      <span>
                        {formatMetricValue(baseline, key)}
                      </span>
                      <span className="evaluation-adaptive">
                        {formatMetricValue(adaptive, key)}
                      </span>
                      <span
                        className={`evaluation-improvement ${
                          improvement.positive
                            ? "improvement-positive"
                            : improvement.negative
                              ? "improvement-negative"
                              : ""
                        }`}
                      >
                        {improvement.text}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="dashboard-note">
                <Activity size={16} />

                <span>
                  V4 benchmark: <b>1,000 queries</b>. Adaptive weighting changed
                  on all queries; NDCG improved on 209 and MRR improved on 138.
                </span>
              </div>
            </div>
          </div>

          <div className="dashboard-grid v4-results-grid">
            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">V4 BENCHMARK</span>
                  <h3>Adaptive effect</h3>
                </div>
                <Check size={20} />
              </div>

              <div className="context-stats">
                <ContextStat label="Queries evaluated" value="1,000" />
                <ContextStat label="Adaptive weights changed" value="100%" />
                <ContextStat label="MRR improved" value="138" />
                <ContextStat label="NDCG@10 improved" value="209" />
                <ContextStat label="Precision@10 improved" value="67" />
                <ContextStat label="Recall@10 improved" value="67" />
              </div>

              <div className="dashboard-note">
                <Activity size={16} />
                <span>
                  Improvements are measured against the fixed-hybrid result
                  for each query in the V4 evaluation.
                </span>
              </div>
            </div>

            <div className="dashboard-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">DOMAIN ANALYSIS</span>
                  <h3>Performance by domain</h3>
                </div>
                <BarChart3 size={20} />
              </div>

              <div className="evaluation-table domain-table">
                <div className="evaluation-head">
                  <span>DOMAIN</span>
                  <span>PRECISION@10</span>
                  <span>RECALL@10</span>
                  <span>MRR</span>
                  <span>NDCG@10</span>
                </div>
                {V4_BENCHMARK.domains.map(
                  ([domain, precision, recall, mrr, ndcg]) => (
                    <div className="evaluation-row" key={domain}>
                      <span>{domain}</span>
                      <span>{precision.toFixed(4)}</span>
                      <span>{recall.toFixed(4)}</span>
                      <span>{mrr.toFixed(4)}</span>
                      <span className="evaluation-adaptive">
                        {ndcg.toFixed(4)}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
          </div>

          <div className="context-section dashboard-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">
                  CONTEXT SELECTION
                </span>

                <h3>
                  Adaptive context construction
                </h3>
              </div>

              <FileText size={20} />
            </div>

            <div className="context-stats">
              <ContextStat
                label="Initial reranked chunks"
                value={
                  context.initial_reranked_chunks ??
                  "—"
                }
              />

              <ContextStat
                label="Selected chunks"
                value={
                  context.selected_chunks ??
                  "—"
                }
              />

              <ContextStat
                label="Selection method"
                value={
                  context.selection_method ||
                  "—"
                }
              />

              <ContextStat
                label="Expansion window"
                value={
                  context.window !== undefined
                    ? `±${context.window}`
                    : "—"
                }
              />

              <ContextStat
                label="Selection latency"
                value={formatMs(
                  context.selection_latency_ms
                )}
              />
            </div>
          </div>

          <div className="pipeline-section dashboard-card">
            <div className="card-heading">
              <div>
                <span className="eyebrow">
                  ADAPTIVE HYBRID RAG PIPELINE
                </span>

                <h3>
                  Complete execution flow
                </h3>
              </div>

              <GitBranch size={20} />
            </div>

            <div className="pipeline">
              <PipelineStep
                title="User Query"
                value="Input"
                description="Natural language question"
                icon={<MessageSquare size={18} />}
                active
              />

              <PipelineStep
                title="BM25"
                value={formatPercent(bm25)}
                description="Lexical retrieval"
                icon={<Search size={18} />}
                active
              />

              <PipelineStep
                title="Dense"
                value={formatPercent(dense)}
                description="Semantic retrieval"
                icon={<BrainCircuit size={18} />}
                active
              />

              <PipelineStep
                title="Adaptive Predictor"
                value={formatMs(
                  retrievalMetrics.adaptive_prediction_ms
                )}
                description="Query-aware weighting"
                icon={<Activity size={18} />}
                active
              />

              <PipelineStep
                title="Fusion"
                value={
                  retrievalConfig.fused_candidates
                    ? `${retrievalConfig.fused_candidates} candidates`
                    : "Hybrid"
                }
                description="Combine lexical + semantic"
                icon={<Network size={18} />}
                active
              />

              <PipelineStep
                title="Reranking"
                value={
                  reranking.applied
                    ? `${reranking.reranked_count || 0} chunks`
                    : "Not applied"
                }
                description="Cross-encoder"
                icon={<BarChart3 size={18} />}
                active={reranking.applied}
              />

              <PipelineStep
                title="Context Selection"
                value={
                  context.selected_chunks
                    ? `${context.selected_chunks} chunks`
                    : "Adaptive"
                }
                description={
                  context.selection_method ||
                  "Context construction"
                }
                icon={<FileText size={18} />}
                active
              />

              <PipelineStep
                title="Qwen 2.5"
                value={formatMs(generation)}
                description="Local LLM generation"
                icon={<Sparkles size={18} />}
                active
                last
              />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function MeasurementRow({ label, value }) {
  return (
    <div className="measurement-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CandidateBox({ title, value }) {
  return (
    <div className="candidate-box">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ModelLine({ label, value }) {
  return (
    <div className="model-line">
      <span>{label}</span>
      <strong>{value || "—"}</strong>
    </div>
  );
}

function WeightBar({ label, value }) {
  return (
    <div className="weight-block">
      <div className="weight-label">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>

      <div className="weight-track">
        <div
          className="weight-fill"
          style={{
            width: `${Math.min(
              100,
              Math.max(0, value * 100)
            )}%`,
          }}
        />
      </div>
    </div>
  );
}

function ContextStat({ label, value }) {
  return (
    <div className="context-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function App() {
  const [user, setUser] = useState(getStoredUser);
  const [booting, setBooting] = useState(true);

  const [conversations, setConversations] =
    useState([]);

  const [documents, setDocuments] = useState([]);

  const [activeConversationId, setActiveConversationId] =
    useState(null);

  const [messages, setMessages] = useState([]);

  const [query, setQuery] = useState("");

  const [loadingConversations, setLoadingConversations] =
    useState(false);

  const [loadingConversation, setLoadingConversation] =
    useState(false);

  const [sending, setSending] = useState(false);

  const [uploading, setUploading] = useState(false);

  const [error, setError] = useState("");

  const [latestRag, setLatestRag] =
    useState(null);

  const [researchOpen, setResearchOpen] =
    useState(false);

  const [mobileSidebar, setMobileSidebar] =
    useState(false);

  const [profileOpen, setProfileOpen] =
    useState(false);

  const [documentsOpen, setDocumentsOpen] =
    useState(true);

  const inputRef = useRef(null);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (item) =>
          item.id === activeConversationId
      ),
    [conversations, activeConversationId]
  );

  useEffect(() => {
    if (!user) {
      setBooting(false);
      return;
    }

    verifySession();
  }, []);

  async function verifySession() {
    try {
      const data = await apiRequest("/auth/me");

      setUser(data.user);

      await Promise.all([
        loadConversations(),
        loadDocuments(),
      ]);
    } catch {
      logout();
    } finally {
      setBooting(false);
    }
  }

  async function loadConversations() {
    setLoadingConversations(true);

    try {
      const data = await apiRequest(
        "/conversations"
      );

      setConversations(data.conversations || []);

      if (
        data.conversations?.length &&
        !activeConversationId
      ) {
        setActiveConversationId(
          data.conversations[0].id
        );

        await loadConversation(
          data.conversations[0].id
        );
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingConversations(false);
    }
  }

  async function loadDocuments() {
    try {
      const data = await apiRequest(
        "/user/documents"
      );

      setDocuments(data.documents || []);
    } catch {
      setDocuments([]);
    }
  }

  async function loadConversation(id) {
    if (!id) return;

    setLoadingConversation(true);
    setError("");
    setLatestRag(null);

    try {
      const data = await apiRequest(
        `/conversations/${id}`
      );

      setMessages(data.messages || []);

      setActiveConversationId(id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingConversation(false);
    }
  }

  async function createConversation() {
    setError("");

    try {
      const data = await apiRequest(
        "/conversations",
        {
          method: "POST",
          body: JSON.stringify({
            title: "New Conversation",
          }),
        }
      );

      const conversation =
        data.conversation;

      setConversations((prev) => [
        conversation,
        ...prev,
      ]);

      setActiveConversationId(
        conversation.id
      );

      setMessages([]);
      setLatestRag(null);
      setResearchOpen(false);
      setMobileSidebar(false);

      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    } catch (err) {
      setError(err.message);
    }
  }

  async function sendMessage() {
    const trimmed = query.trim();

    if (!trimmed || sending) return;

    setError("");

    let conversationId =
      activeConversationId;

    try {
      if (!conversationId) {
        const data = await apiRequest(
          "/conversations",
          {
            method: "POST",
            body: JSON.stringify({
              title: "New Conversation",
            }),
          }
        );

        conversationId =
          data.conversation.id;

        setConversations((prev) => [
          data.conversation,
          ...prev,
        ]);

        setActiveConversationId(
          conversationId
        );
      }

      const temporaryUserMessage = {
        id: `temp-user-${Date.now()}`,
        role: "user",
        content: trimmed,
        created_at:
          new Date().toISOString(),
      };

      setMessages((prev) => [
        ...prev,
        temporaryUserMessage,
      ]);

      setQuery("");
      setSending(true);

      const data = await apiRequest(
        `/conversations/${conversationId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            query: trimmed,
            top_k: 10,
          }),
        }
      );

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer || "",
        created_at:
          new Date().toISOString(),
      };

      setMessages((prev) => [
        ...prev,
        assistantMessage,
      ]);

      setLatestRag(data.rag || null);
      setResearchOpen(true);

      if (data.rag) {
        localStorage.setItem(
          "ahr_latest_rag",
          JSON.stringify(data.rag)
        );
      }

      await loadConversations();

      if (conversationId) {
        setActiveConversationId(
          conversationId
        );
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);

      setTimeout(() => {
        inputRef.current?.focus();
      }, 50);
    }
  }

  async function deleteConversation(
    event,
    id
  ) {
    event.stopPropagation();

    const confirmed = window.confirm(
      "Delete this conversation?"
    );

    if (!confirmed) return;

    try {
      await apiRequest(
        `/conversations/${id}`,
        {
          method: "DELETE",
        }
      );

      const remaining =
        conversations.filter(
          (item) => item.id !== id
        );

      setConversations(remaining);

      if (activeConversationId === id) {
        if (remaining.length) {
          await loadConversation(
            remaining[0].id
          );
        } else {
          setActiveConversationId(null);
          setMessages([]);
          setLatestRag(null);
        }
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function uploadDocument(event) {
    const file =
      event.target.files?.[0];

    if (!file) return;

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      await apiRequest("/user/upload", {
        method: "POST",
        body: formData,
      });

      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  function logout() {
    localStorage.removeItem("ahr_token");
    localStorage.removeItem("ahr_user");
    setUser(null);
    setConversations([]);
    setMessages([]);
    setDocuments([]);
    setLatestRag(null);
    setActiveConversationId(null);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  if (booting) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">
          <Sparkles size={28} />
        </div>

        <strong>
          Adaptive Hybrid RAG
        </strong>

        <span>
          Loading your workspace...
        </span>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthScreen
        onAuthenticated={(nextUser) => {
          setUser(nextUser);
          setBooting(true);

          setTimeout(() => {
            verifySession();
          }, 0);
        }}
      />
    );
  }

  return (
    <div className="app-shell">
      {mobileSidebar && (
        <div
          className="mobile-overlay"
          onClick={() =>
            setMobileSidebar(false)
          }
        />
      )}

      <aside
        className={`sidebar ${
          mobileSidebar
            ? "sidebar-open"
            : ""
        }`}
      >
        <div className="sidebar-brand">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>

          <div>
            <strong>
              Adaptive Hybrid RAG
            </strong>

            <span>
              Document Intelligence
            </span>
          </div>

          <button
            className="mobile-close"
            onClick={() =>
              setMobileSidebar(false)
            }
          >
            <X size={19} />
          </button>
        </div>

        <button
          className="new-conversation"
          onClick={createConversation}
        >
          <Plus size={18} />
          New conversation
        </button>

        <div className="sidebar-section">
          <div className="section-title">
            <span>CONVERSATIONS</span>

            {loadingConversations && (
              <span className="mini-spinner" />
            )}
          </div>

          <div className="conversation-list">
            {conversations.length === 0 ? (
              <div className="sidebar-empty">
                No conversations yet.
              </div>
            ) : (
              conversations.map(
                (conversation) => (
                  <button
                    className={`conversation-item ${
                      activeConversationId ===
                      conversation.id
                        ? "selected"
                        : ""
                    }`}
                    key={conversation.id}
                    onClick={() => {
                      loadConversation(
                        conversation.id
                      );

                      setMobileSidebar(false);
                    }}
                  >
                    <MessageSquare
                      size={15}
                    />

                    <span>
                      {conversation.title ||
                        "New Conversation"}
                    </span>

                    <button
                      className="conversation-delete"
                      onClick={(event) =>
                        deleteConversation(
                          event,
                          conversation.id
                        )
                      }
                    >
                      <X size={14} />
                    </button>
                  </button>
                )
              )
            )}
          </div>
        </div>

        <div className="sidebar-section documents-section">
          <button
            className="section-title-button"
            onClick={() =>
              setDocumentsOpen(
                (value) => !value
              )
            }
          >
            <span>YOUR DOCUMENTS</span>

            <span className="document-actions">
              <ChevronDown
                size={15}
                className={
                  documentsOpen
                    ? "rotate"
                    : ""
                }
              />
            </span>
          </button>

          {documentsOpen && (
            <>
              <div className="document-list">
                {documents.length === 0 ? (
                  <div className="sidebar-empty">
                    No documents uploaded.
                  </div>
                ) : (
                  documents.map(
                    (document) => (
                      <div
                        className="document-item"
                        key={document.file_id}
                      >
                        <FileText
                          size={16}
                        />

                        <div>
                          <strong>
                            {document.file_name}
                          </strong>

                          <span>
                            {document.chunks} chunks
                          </span>
                        </div>
                      </div>
                    )
                  )
                )}
              </div>

              <label className="upload-button">
                {uploading ? (
                  <>
                    <span className="mini-spinner" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload size={16} />
                    Upload document
                  </>
                )}

                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={uploadDocument}
                  disabled={uploading}
                  hidden
                />
              </label>
            </>
          )}
        </div>

        <div className="sidebar-user">
          <div className="user-card">
            <div className="user-avatar">
              {initials(user.username)}
            </div>

            <div className="user-info">
              <strong>
                {user.username}
              </strong>

              <span>
                {user.email}
              </span>
            </div>
          </div>

          <button
            className="logout-button"
            onClick={logout}
          >
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button
            className="mobile-menu"
            onClick={() =>
              setMobileSidebar(true)
            }
          >
            <Menu size={21} />
          </button>

          <div className="topbar-left">
            <div className="connection-status">
              <span />
              Local
            </div>
          </div>

          <button
            className={`research-toggle ${
              researchOpen ? "active" : ""
            }`}
            onClick={() =>
              setResearchOpen(
                (value) => !value
              )
            }
          >
            <GitBranch size={17} />

            <span>
              Research & Implementation
            </span>

            <ChevronDown
              size={16}
              className={
                researchOpen
                  ? "rotate"
                  : ""
              }
            />
          </button>

          <div className="topbar-right">
            <button
              className="profile-button"
              onClick={() =>
                setProfileOpen(
                  (value) => !value
                )
              }
            >
              {user.username}
              <ChevronDown size={14} />
            </button>

            {profileOpen && (
              <div className="profile-menu">
                <div className="profile-menu-user">
                  <div className="user-avatar small">
                    {initials(
                      user.username
                    )}
                  </div>

                  <div>
                    <strong>
                      {user.username}
                    </strong>

                    <span>
                      {user.email}
                    </span>
                  </div>
                </div>

                <button onClick={logout}>
                  <LogOut size={15} />
                  Logout
                </button>
              </div>
            )}
          </div>
        </header>

        <div className="workspace">
          {error && (
            <div className="error-banner">
              <span>{error}</span>

              <button
                onClick={() =>
                  setError("")
                }
              >
                <X size={16} />
              </button>
            </div>
          )}

          <div className="conversation-header">
            <div>
              <h1>
                {activeConversation?.title ||
                  "New Conversation"}
              </h1>

              <p>
                Your private document
                conversation
              </p>
            </div>
          </div>

          <div className="messages-area">
            {loadingConversation ? (
              <div className="conversation-loading">
                <span className="loading-dot" />
                Loading conversation...
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-chat">
                <div className="empty-chat-icon">
                  <Sparkles size={28} />
                </div>

                <h2>
                  Chat with your documents
                </h2>

                <p>
                  Upload a document and ask a
                  question to start your
                  adaptive retrieval pipeline.
                </p>

                <div className="suggestion-row">
                  <button
                    onClick={() =>
                      setQuery(
                        "Summarize this document."
                      )
                    }
                  >
                    Summarize this document
                  </button>

                  <button
                    onClick={() =>
                      setQuery(
                        "What are the main concepts discussed in this document?"
                      )
                    }
                  >
                    Main concepts
                  </button>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  className={`message ${
                    message.role === "user"
                      ? "user-message"
                      : "assistant-message"
                  }`}
                  key={message.id}
                >
                  <div className="message-avatar">
                    {message.role === "user" ? (
                      initials(user.username)
                    ) : (
                      <Sparkles size={17} />
                    )}
                  </div>

                  <div className="message-body">
                    <div className="message-author">
                      {message.role === "user"
                        ? "You"
                        : "Adaptive RAG"}

                      {message.created_at && (
                        <span>
                          {formatTime(
                            message.created_at
                          )}
                        </span>
                      )}
                    </div>

                    <div className="message-content">
                      {message.role ===
                      "assistant"
                        ? renderMarkdown(
                            message.content
                          )
                        : message.content}
                    </div>
                  </div>
                </div>
              ))
            )}

            {sending && (
              <div className="message assistant-message">
                <div className="message-avatar">
                  <Sparkles size={17} />
                </div>

                <div className="message-body">
                  <div className="message-author">
                    Adaptive RAG
                  </div>

                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}

            {researchOpen && (
              <ResearchDashboard
                rag={latestRag}
              />
            )}

            <div className="messages-bottom-space" />
          </div>

          <div className="composer-area">
            <div className="composer">
              <Sparkles
                size={18}
                className="composer-icon"
              />

              <textarea
                ref={inputRef}
                value={query}
                onChange={(event) =>
                  setQuery(
                    event.target.value
                  )
                }
                onKeyDown={handleKeyDown}
                placeholder={
                  documents.length
                    ? "Ask something about your documents..."
                    : "Upload a document to start asking questions..."
                }
                disabled={
                  sending ||
                  documents.length === 0
                }
                rows={1}
              />

              <button
                className="send-button"
                onClick={sendMessage}
                disabled={
                  sending ||
                  !query.trim() ||
                  documents.length === 0
                }
              >
                {sending ? (
                  <span className="button-spinner" />
                ) : (
                  <ArrowUp size={19} />
                )}
              </button>
            </div>

            <div className="composer-hint">
              Press Enter to send · Shift + Enter
              for a new line
            </div>
          </div>
        </div>

        <footer className="statusbar">
          <span>
            BM25{" "}
            {latestRag
              ? formatPercent(
                  latestRag.weights?.bm25
                )
              : "—"}
          </span>

          <span>
            Dense{" "}
            {latestRag
              ? formatPercent(
                  latestRag.weights?.dense
                )
              : "—"}
          </span>

          <span>
            Model{" "}
            {latestRag?.models?.generation ||
              "qwen2.5:7b"}
          </span>

          <span>
            Retrieval{" "}
            {latestRag
              ? formatMs(
                  latestRag
                    .retrieval_metrics
                    ?.total_retrieval_ms
                )
              : "—"}
          </span>

          <span>
            Generation{" "}
            {latestRag
              ? formatMs(
                  latestRag
                    .retrieval_metrics
                    ?.generation_ms
                )
              : "—"}
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;