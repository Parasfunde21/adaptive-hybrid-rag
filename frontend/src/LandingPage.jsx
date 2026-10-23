import { useEffect, useState } from "react";
import {
  ArrowRight,
  BrainCircuit,
  ChevronRight,
  GitBranch,
  Layers3,
  MessageSquareText,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const pipelineSteps = [
  "Document",
  "Text Extraction & Cleaning",
  "Adaptive Chunking",
  "BM25 + Dense Retrieval",
  "17 Retrieval Diagnostics",
  "Adaptive Weight Prediction",
  "Hybrid Score Fusion",
  "Cross-Encoder Reranking",
  "Qwen2.5-7B",
  "Answer + Provenance",
];

const differentiators = [
  {
    icon: <Layers3 size={20} />,
    title: "17 Retrieval-Diagnostic Features",
    description:
      "The system analyzes lexical, semantic, agreement, query-level, and lexical-coverage signals.",
  },
  {
    icon: <BrainCircuit size={20} />,
    title: "Adaptive BM25/Dense Weighting",
    description:
      "An Extra Trees regressor predicts a query-specific BM25 contribution.",
  },
  {
    icon: <Network size={20} />,
    title: "Hybrid Retrieval",
    description:
      "BM25 and dense retrieval are combined instead of relying on only one retrieval strategy.",
  },
  {
    icon: <ShieldCheck size={20} />,
    title: "Provenance-Aware Answers",
    description:
      "The application preserves source information and chunk-level provenance for generated responses.",
  },
];

const techStack = [
  "React",
  "Tailwind CSS",
  "FastAPI",
  "BM25",
  "Sentence-Transformers / MiniLM",
  "ChromaDB",
  "Cross-Encoder",
  "Qwen2.5-7B",
  "Ollama",
];

const capabilities = [
  "Upload domain-specific documents",
  "Ask questions about uploaded content",
  "Retrieve relevant document passages",
  "Use adaptive hybrid retrieval",
  "Generate context-aware answers",
  "View provenance/source information",
];

const benchmarkStats = [
  { label: "Precision@10", value: "0.1148" },
  { label: "Recall@10", value: "0.5381" },
  { label: "nDCG@10", value: "0.4159" },
  { label: "MRR", value: "0.4340" },
];

function LandingPage({ onNavigate }) {
  const [activeSection, setActiveSection] = useState("about");

  useEffect(() => {
    const sections = Array.from(
      document.querySelectorAll("section[id]")
    ).filter((section) =>
      ["about", "problem", "how-it-works", "technology"].includes(section.id)
    );

    if (!sections.length) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntry = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visibleEntry) {
          setActiveSection(visibleEntry.target.id);
        }
      },
      {
        root: null,
        threshold: [0.2, 0.4, 0.6],
        rootMargin: "-18% 0px -42% 0px",
      }
    );

    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, []);

  return (
    <div className="landing-page">
      <header className="landing-header">
        <div className="landing-brand">
          <div className="brand-icon large landing-brand-icon">
            <Sparkles size={22} />
          </div>

          <div>
            <strong>Adaptive Hybrid RAG</strong>
            <span>Query-aware retrieval research system</span>
          </div>
        </div>

        <nav className="landing-nav" aria-label="Main navigation">
          <a href="#about" className={activeSection === "about" ? "landing-nav-item active" : "landing-nav-item"}>
            About
          </a>
          <a href="#problem" className={activeSection === "problem" ? "landing-nav-item active" : "landing-nav-item"}>
            Problem
          </a>
          <a href="#how-it-works" className={activeSection === "how-it-works" ? "landing-nav-item active" : "landing-nav-item"}>
            How it works
          </a>
          <a href="#technology" className={activeSection === "technology" ? "landing-nav-item active" : "landing-nav-item"}>
            Technology
          </a>
          <button type="button" className="landing-nav-button" onClick={() => onNavigate("/login")}>
            Sign in
          </button>
        </nav>
      </header>

      <main className="landing-main">
        <section className="landing-hero">
          <div className="landing-hero-copy">
            <div className="eyebrow">Research-driven question answering</div>

            <h1>Query-Aware Adaptive Hybrid RAG</h1>

            <p className="landing-lead">
              A domain-specific question answering framework that dynamically combines lexical and
              semantic retrieval based on how each query behaves.
            </p>

            <p className="landing-subtext">
              Instead of using a fixed retrieval balance for every query, the system analyzes
              retrieval signals and predicts a query-specific BM25 and dense retrieval weighting.
            </p>

            <div className="landing-actions">
              <button type="button" className="primary-button landing-primary" onClick={() => onNavigate("/login")}>
                <span>Get Started</span>
                <ArrowRight size={18} aria-hidden="true" />
              </button>

              <a href="#how-it-works" className="secondary-button landing-secondary">
                Explore How It Works
              </a>
            </div>
          </div>

          <div className="landing-hero-panel" aria-label="Research overview panel">
            <div className="hero-panel-header">
              <div className="hero-panel-badge">
                <GitBranch size={16} />
                Adaptive workflow
              </div>
            </div>

            <div className="hero-panel-body">
              <div className="hero-metric-row">
                <Search size={18} />
                <span>BM25 retrieval</span>
                <strong>Adaptive</strong>
              </div>

              <div className="hero-metric-row">
                <BrainCircuit size={18} />
                <span>Dense retrieval</span>
                <strong>Adaptive</strong>
              </div>

              <div className="hero-metric-row">
                <Network size={18} />
                <span>Fusion strategy</span>
                <strong>Query-aware</strong>
              </div>

              <div className="hero-metric-row">
                <MessageSquareText size={18} />
                <span>Generation</span>
                <strong>Qwen2.5-7B</strong>
              </div>
            </div>
          </div>
        </section>

        <section id="about" className="landing-section">
          <div className="section-heading">
            <span className="eyebrow">What is this project?</span>
            <h2>What is Query-Aware Adaptive RAG?</h2>
          </div>

          <div className="section-copy">
            <p>
              Traditional hybrid retrieval commonly uses a fixed weighting between BM25 and dense
              retrieval.
            </p>
            <p>
              This project makes that weighting query-aware. For every query, the system analyzes
              retrieval behavior using 17 diagnostic features and uses an Extra Trees regressor to
              predict the appropriate BM25 contribution, with the dense contribution as its
              complement.
            </p>
          </div>
        </section>

        <section id="problem" className="landing-section landing-problems">
          <div className="section-heading">
            <span className="eyebrow">Why adaptive retrieval?</span>
            <h2>Why Adaptive Retrieval?</h2>
          </div>

          <div className="problem-grid">
            <article className="problem-card">
              <span className="problem-kicker">Fixed Retrieval Weights</span>
              <p>A single BM25/dense balance is applied to every query.</p>
            </article>

            <article className="problem-card">
              <span className="problem-kicker">Different Queries Behave Differently</span>
              <p>Some queries benefit more from lexical matching while others benefit more from semantic similarity.</p>
            </article>

            <article className="problem-card">
              <span className="problem-kicker">Retrieval Quality Affects RAG</span>
              <p>Poor retrieval can result in irrelevant context being passed to the language model.</p>
            </article>
          </div>

          <div className="solution-callout">
            <strong>Query-Aware Fusion</strong>
            <span>The system adapts the retrieval balance according to query-specific retrieval diagnostics.</span>
          </div>
        </section>

        <section id="how-it-works" className="landing-section">
          <div className="section-heading">
            <span className="eyebrow">How it works</span>
            <h2>How the System Works</h2>
          </div>

          <div className="pipeline-flow" aria-label="Research pipeline flow">
            {pipelineSteps.map((step, index) => (
              <div key={step} className="pipeline-node-wrap">
                <div className="pipeline-node">
                  <span>{step}</span>
                </div>

                {index < pipelineSteps.length - 1 && (
                  <div className="pipeline-arrow" aria-hidden="true">
                    <ChevronRight size={18} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="landing-section">
          <div className="section-heading">
            <span className="eyebrow">What makes it different</span>
            <h2>Query-Aware Adaptive Retrieval</h2>
          </div>

          <div className="feature-grid">
            {differentiators.map(({ icon, title, description }) => (
              <article key={title} className="feature-card">
                <div className="feature-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{description}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="technology" className="landing-section">
          <div className="section-heading">
            <span className="eyebrow">Technology stack</span>
            <h2>Technology Stack</h2>
          </div>

          <div className="stack-grid">
            {techStack.map((item) => (
              <div key={item} className="stack-pill">
                {item}
              </div>
            ))}
          </div>
        </section>

        <section className="landing-section">
          <div className="section-heading">
            <span className="eyebrow">Use the system</span>
            <h2>Use the System</h2>
          </div>

          <div className="capability-list">
            {capabilities.map((item) => (
              <div key={item} className="capability-item">
                <div className="bullet-check">
                  <CheckmarkIcon />
                </div>
                <span>{item}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="landing-section landing-research">
          <div className="section-heading">
            <span className="eyebrow">Research and results</span>
            <h2>Evaluated on a 1,000-Query Benchmark</h2>
          </div>

          <p className="research-domain">
            Evaluated across four BEIR domains: ArguAna, FiQA, NFCorpus, and SciFact.
          </p>

          <div className="benchmark-grid">
            {benchmarkStats.map(({ label, value }) => (
              <div key={label} className="benchmark-card">
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          <div className="research-text">
            <p>
              Adaptive Hybrid — 0.1148 for Precision@10, 0.5381 for Recall@10, 0.4159 for nDCG@10,
              and 0.4340 for MRR.
            </p>
            <p>
              The adaptive approach improved three of the four ranking metrics over the static
              0.50/0.50 hybrid baseline, while MRR was slightly lower.
            </p>
          </div>
        </section>

        <section className="landing-cta">
          <div>
            <span className="eyebrow">Explore the framework</span>
            <h2>Explore Query-Aware Adaptive RAG</h2>
            <p>
              Upload your documents, ask questions, and explore how adaptive hybrid retrieval
              supports domain-specific question answering.
            </p>
          </div>

          <button type="button" className="primary-button landing-primary" onClick={() => onNavigate("/login")}>
            <span>Get Started</span>
            <ArrowRight size={18} aria-hidden="true" />
          </button>
        </section>
      </main>

      <footer className="landing-footer">
        <div className="landing-footer-brand">
          <div className="brand-icon large landing-brand-icon">
            <Sparkles size={22} />
          </div>
          <div>
            <strong>Adaptive Hybrid RAG</strong>
            <span>Query-aware hybrid retrieval for domain-specific question answering.</span>
          </div>
        </div>

        <div className="landing-footer-meta">
          <span>React</span>
          <span>FastAPI</span>
          <span>BM25</span>
          <span>ChromaDB</span>
          <span>Qwen2.5-7B</span>
        </div>

        <div className="landing-footer-links">
          <a href="https://github.com/Parasfunde21/adaptive-hybrid-rag.git" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

function CheckmarkIcon() {
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 13.2 4.3 10l-1.1 1.1 4.3 4.3 9.5-9.5-1.1-1.1z" /></svg>;
}

export default LandingPage;
