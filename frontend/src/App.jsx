import { Fragment, useEffect, useRef, useState } from "react";

const DEFAULT_API_BASE = "http://127.0.0.1:8001";
const EXAMPLES_PAGE_SIZE = 200;
const DEFAULT_DATASET_PATH =
  "/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/examples/output/rf_dataset_v2";

const RUN_ROWS = [
  {
    name: "mr_yolo_pyramid_v4",
    model: "MR_YOLO",
    dataset: "rf_dataset_thesis_v2",
    status: "Training",
    metric: "mAP50 0.941",
    eta: "01h 18"
  },
  {
    name: "yolov8_cfg512_clean",
    model: "YOLOv8",
    dataset: "rf_dataset_cfg512",
    status: "Queued",
    metric: "Warmup",
    eta: "En attente"
  },
  {
    name: "mr_tf_sep_ablation",
    model: "MR_TF",
    dataset: "rf_dataset_fused",
    status: "Completed",
    metric: "mAP50 0.928",
    eta: "Termine"
  }
];

const DATASET_ROWS = [
  {
    name: "rf_dataset_thesis_v2",
    split: "train / val",
    samples: "100k / 20k",
    classes: "20",
    note: "dataset principal pour les experiences multi-resolution"
  },
  {
    name: "rf_dataset_cfg512",
    split: "train / val",
    samples: "84k / 16k",
    classes: "20",
    note: "jeu de reference pour les variantes mono-resolution"
  },
  {
    name: "rf_dataset_noise_aug",
    split: "train / val",
    samples: "120k / 24k",
    classes: "20",
    note: "augmentation forte pour tester la robustesse faible SNR"
  }
];

const EVALUATION_METRICS = [
  { key: "map50_95", label: "mAP50:95", family: "Performance", mode: "max" },
  { key: "map50", label: "mAP50", family: "Performance", mode: "max" },
  { key: "avg_recall_low_snr", label: "Recall low SNR", family: "Recall", mode: "max" },
  { key: "avg_recall_medium_snr", label: "Recall medium SNR", family: "Recall", mode: "max" },
  { key: "avg_recall_high_snr", label: "Recall high SNR", family: "Recall", mode: "max" },
  { key: "val_loss", label: "Validation loss", family: "Loss", mode: "min" },
  { key: "train_loss", label: "Training loss", family: "Loss", mode: "min" },
  { key: "loss_box_val", label: "Box loss val", family: "Loss", mode: "min" },
  { key: "loss_cls_val", label: "Cls loss val", family: "Loss", mode: "min" },
  { key: "loss_dfl_val", label: "DFL loss val", family: "Loss", mode: "min" },
];

const DEFAULT_EVALUATION_RUN_INPUTS = [
  {
    path: "/Users/tailleesarah/Documents/thèse/icml/detector2026/runs/examples_of_training/tf_attn_yolon_specificres_cfg512",
    label: "TF Attn YOLOn",
  },
  {
    path: "/Users/tailleesarah/Documents/thèse/icml/detector2026/runs/examples_of_training/yolov11n_specificres",
    label: "YOLOv11n",
  },
];

const EVALUATION_VIEWS = [
  { id: "map_vs_model_cost", label: "mAP vs FLOPs / Params" },
  { id: "confusion_matrices", label: "Confusion matrices" },
  { id: "best_recall_vs_snr", label: "Best recall vs snr" },
  { id: "f1_threshold_curves", label: "F1 / Precision / Recall" },
  { id: "map_vs_epochs", label: "Map vs epochs" },
  { id: "loss_vs_epochs", label: "Loss vs epochs" },
  { id: "recall_vs_epochs", label: "Recall vs epochs" },
];

const EVALUATION_SCATTER_X_OPTIONS = [
  { id: "flops", label: "FLOPs" },
  { id: "params", label: "Params" },
];

const EVALUATION_SCATTER_METRIC_OPTIONS = [
  { id: "map50_95", label: "mAP50:95", marker: "circle" },
  { id: "map50", label: "mAP50", marker: "square" },
];

const EVALUATION_CONFUSION_SNR_OPTIONS = [
  { id: "low_snr", label: "Low SNR" },
  { id: "medium_snr", label: "Medium SNR" },
  { id: "high_snr", label: "High SNR" },
];

const EVALUATION_CONFUSION_MODES = [
  { id: "single", label: "Matrice seule" },
  { id: "difference", label: "Difference A - B" },
];

const EVALUATION_CONFUSION_NORMALIZATION_OPTIONS = [
  { id: "none", label: "Aucune" },
  { id: "row", label: "Par ligne" },
  { id: "column", label: "Par colonne" },
];

const EVALUATION_CURVE_SCOPE_OPTIONS = [
  { id: "global", label: "Global" },
  { id: "per_class", label: "Par classe" },
];

function normalizeApiBase(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    return "";
  }
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
}

function resolveApiUrl(apiBase, path) {
  const normalizedBase = normalizeApiBase(apiBase);
  if (!normalizedBase) {
    return path;
  }
  if (path.startsWith("/")) {
    return `${normalizedBase}${path}`;
  }
  return `${normalizedBase}/${path}`;
}

function extractApiError(data, fallbackStatus) {
  if (typeof data?.detail === "string") {
    return { message: data.detail, raw: data.detail, diagnostics: null };
  }
  if (data?.detail && typeof data.detail === "object") {
    return {
      message: data.detail.message ?? `HTTP ${fallbackStatus}`,
      raw: JSON.stringify(data.detail, null, 2),
      diagnostics: data.detail.diagnostics ?? null,
    };
  }
  return {
    message: `HTTP ${fallbackStatus}`,
    raw: JSON.stringify(data ?? {}, null, 2),
    diagnostics: null,
  };
}

function PathField({
  label,
  value,
  onChange,
  apiFetch,
  pickerKind = "directory",
  pickerTitle = "Choisir un chemin",
  placeholder = "",
  note,
  className = "",
}) {
  const [pickerState, setPickerState] = useState({ status: "idle", error: "" });
  const helperText =
    note ??
    (pickerKind === "file"
      ? "Saisie libre ou selection native du fichier."
      : "Saisie libre ou selection native du dossier.");
  const isOpeningPicker = pickerState.status === "loading";

  async function handleBrowse() {
    if (!apiFetch || isOpeningPicker) {
      return;
    }

    setPickerState({ status: "loading", error: "" });
    try {
      const response = await apiFetch("/system/path-picker", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: pickerKind,
          path: value,
          title: pickerTitle,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(extractApiError(data, response.status).message);
      }
      if (!data?.cancelled && typeof data?.path === "string" && data.path.trim()) {
        onChange(data.path);
      }
      setPickerState({ status: "idle", error: "" });
    } catch (error) {
      setPickerState({
        status: "error",
        error: error instanceof Error ? error.message : "Impossible d'ouvrir le selecteur natif.",
      });
    }
  }

  return (
    <label className={`field ${className}`.trim()}>
      <span>{label}</span>
      <div className="path-field-shell">
        <div className="path-field-row">
          <input
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
          />
          <button
            type="button"
            className="ghost-button path-field-browse-button"
            onClick={() => void handleBrowse()}
            disabled={isOpeningPicker}
          >
            {isOpeningPicker ? "Ouverture..." : "Parcourir"}
          </button>
        </div>
        <div className="path-field-meta">
          <small>{helperText}</small>
          {pickerState.error ? <small className="path-field-error">{pickerState.error}</small> : null}
        </div>
      </div>
    </label>
  );
}

function Navigation({ page, onNavigate, isConnected, connectionLabel, onDisconnect }) {
  const items = [
    { id: "connect", label: "Connexion" },
    { id: "overview", label: "Accueil" },
    { id: "datasets", label: "Datasets", requiresConnection: true },
    { id: "training", label: "Training", requiresConnection: true },
    { id: "evaluation", label: "Evaluation", requiresConnection: true },
    { id: "artifacts", label: "Artefacts", requiresConnection: true }
  ];

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="topbar-kicker">RF Studio</span>
        <strong>detector2026</strong>
        <small>Entrainement, evaluation et supervision des detecteurs RF</small>
      </div>
      <nav className="nav-tabs" aria-label="Navigation principale">
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-tab ${page === item.id ? "nav-tab-active" : ""}`}
            onClick={() => onNavigate(item.id)}
            disabled={Boolean(item.requiresConnection && !isConnected)}
            title={item.requiresConnection && !isConnected ? "Connexion backend requise" : ""}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <div className="connection-indicator">
        <div className="connection-summary">
          <span className={`status-badge ${isConnected ? "status-done" : "status-idle"}`}>
            {isConnected ? "Backend connecte" : "Backend non connecte"}
          </span>
          <small>{connectionLabel}</small>
        </div>
        {isConnected ? (
          <button type="button" className="ghost-button topbar-logout" onClick={onDisconnect}>
            Deconnecter
          </button>
        ) : null}
      </div>
    </header>
  );
}

function ConnectionPage({
  credentialInput,
  passwordInput,
  onCredentialInputChange,
  onPasswordInputChange,
  connectionInput,
  onConnectionInputChange,
  onConnect,
  onContinue,
  status,
  message,
  connectionLabel,
  latencyMs
}) {
  const isChecking = status === "checking";
  const isConnected = status === "connected";
  const isError = status === "error";
  const statusCopy = isConnected
    ? "Connexion etablie"
    : isChecking
      ? "Verification en cours"
      : isError
        ? "Echec de connexion"
        : "Connexion requise";

  return (
    <div className="page-stack">
      <section className="connect-minimal-wrap">
        <div className="connect-minimal-card">
          <div className="connect-minimal-head">
            <span className="eyebrow">Connexion</span>
            <h1>Acces a detector2026</h1>
            <p>Authentifiez-vous puis verifiez la connexion au back-end local.</p>
          </div>

          <div className="connect-minimal-form">
            <label className="field">
              <span>Identifiant</span>
              <input
                type="text"
                value={credentialInput}
                onChange={(event) => onCredentialInputChange(event.target.value)}
                autoComplete="username"
              />
            </label>
            <label className="field">
              <span>Mot de passe</span>
              <input
                type="password"
                value={passwordInput}
                onChange={(event) => onPasswordInputChange(event.target.value)}
                autoComplete="current-password"
              />
            </label>
            <label className="field">
              <span>Adresse du back-end</span>
              <input
                type="text"
                value={connectionInput}
                onChange={(event) => onConnectionInputChange(event.target.value)}
                placeholder={DEFAULT_API_BASE}
              />
            </label>
          </div>

          <div className="connect-minimal-status">
            <div className="connect-state-row">
              <span className={`connect-pulse connect-pulse-${isConnected ? "done" : isChecking ? "checking" : isError ? "error" : "idle"}`} />
              <span className={`status-badge ${isConnected ? "status-done" : isChecking ? "status-running" : isError ? "status-error" : "status-idle"}`}>
                {statusCopy}
              </span>
              <span className="connect-latency">{latencyMs !== null ? `${latencyMs} ms` : "latence --"}</span>
            </div>
            <p>{message || "Identifiants locaux attendus: admin / admin. Verification back-end via /health."}</p>
            <p>
              Endpoint: <code>{connectionLabel}</code>
            </p>
          </div>

          <div className="hero-actions connect-minimal-actions">
            <button type="button" className="primary-button" onClick={onConnect} disabled={isChecking}>
              {isChecking ? "Verification..." : "Se connecter"}
            </button>
            <button type="button" className="secondary-button" onClick={onContinue} disabled={!isConnected}>
              Ouvrir l&apos;accueil
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function OverviewPage({ onOpenTraining, onOpenDatasets }) {
  const overviewCards = [
    {
      eyebrow: "Etape 1",
      title: "Connexion backend",
      description: "Verifie l'API locale pour deverrouiller les espaces Datasets, Training, Evaluation et Artefacts."
    },
    {
      eyebrow: "Etape 2",
      title: "Pilotage des runs",
      description: "Configure les experiences, choisis le modele, le dataset et suis la progression de l'entrainement."
    },
    {
      eyebrow: "Etape 3",
      title: "Lecture des resultats",
      description: "Compare les checkpoints, lis les metriques et inspecte rapidement les artefacts produits."
    }
  ];

  return (
    <div className="page-stack">
      <header className="hero overview-hero">
        <div className="hero-copy">
          <span className="eyebrow">Accueil</span>
          <h1>Superviser les detecteurs RF avec la meme UX que le simulateur</h1>
          <p>
            detector2026 reprend le socle d&apos;interface de ICML2026DataSimulator:
            une page de connexion claire, une topbar legible, une navigation simple
            et des vues organisees pour piloter datasets, runs, evaluation et artefacts.
          </p>
          <div className="hero-actions">
            <button type="button" className="primary-button" onClick={onOpenTraining}>
              Ouvrir Training
            </button>
            <button type="button" className="secondary-button" onClick={onOpenDatasets}>
              Ouvrir Datasets
            </button>
          </div>
        </div>

        <aside className="hero-panel overview-panel">
          <div className="section-title">
            <span>Valeur produit</span>
            <h2>Un poste de pilotage unique</h2>
            <p>
              L&apos;objectif est de garder la meme logique de navigation que le simulateur,
              mais appliquee aux workflows d&apos;entrainement et d&apos;analyse de modeles.
            </p>
          </div>
          <div className="overview-kpis">
            <article>
              <strong>Connexion</strong>
              <span>Backend local, credentials simples, verification /health</span>
            </article>
            <article>
              <strong>Runs</strong>
              <span>Vue centralisee sur les experiences, statuts et checkpoints</span>
            </article>
            <article>
              <strong>Evaluation</strong>
              <span>mAP, recall SNR, comparaisons et inspection des sorties</span>
            </article>
          </div>
        </aside>
      </header>

      <section className="overview-flow-section">
        <div className="overview-flow-head">
          <h2>Le meme langage d&apos;interface, oriente entrainement et evaluation</h2>
        </div>
        <div className="overview-flow-mask">
          <div className="overview-flow-track">
            {[...overviewCards, ...overviewCards].map((card, index) => (
              <article
                key={`${card.eyebrow}-${card.title}-${index}`}
                className="panel overview-flow-card"
                aria-hidden={index >= overviewCards.length}
              >
                <span className="eyebrow">{card.eyebrow}</span>
                <h3>{card.title}</h3>
                <p>{card.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function DetectorPage({ eyebrow, title, description, stats, children }) {
  return (
    <div className="page-stack">
      <header className="hero detector-hero">
        <div className="hero-copy">
          <span className="eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        <aside className="hero-panel detector-summary-panel">
          <div className="section-title">
            <span>Resume</span>
            <h2>Lecture rapide</h2>
            <p>Les cartes de synthese reprennent le meme rythme de lecture que le simulateur.</p>
          </div>
          <div className="stats-grid">
            {stats.map((item) => (
              <article key={item.label} className="stat-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <small>{item.hint}</small>
              </article>
            ))}
          </div>
        </aside>
      </header>
      {children}
    </div>
  );
}

function formatCount(value) {
  return Number(value ?? 0).toLocaleString("fr-FR");
}

function formatLargeNumber(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "-";
  }
  if (Math.abs(numericValue) >= 1_000_000_000) {
    return `${(numericValue / 1_000_000_000).toFixed(2)}G`;
  }
  if (Math.abs(numericValue) >= 1_000_000) {
    return `${(numericValue / 1_000_000).toFixed(2)}M`;
  }
  if (Math.abs(numericValue) >= 1_000) {
    return `${(numericValue / 1_000).toFixed(2)}k`;
  }
  return numericValue.toLocaleString("fr-FR");
}

function normalizeThresholdInput(value, fallback = 0.1) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return fallback;
  }
  return Math.max(0, Math.min(1, numericValue));
}

function formatDecimal(value, digits = 2) {
  return Number(value ?? 0).toLocaleString("fr-FR", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits
  });
}

function formatPercent(value) {
  return `${formatDecimal((value ?? 0) * 100, 1)}%`;
}

function formatThreshold(value) {
  return formatDecimal(value, 3);
}

function defaultClassLabel(classKey) {
  const numericClassId = Number(classKey);
  if (Number.isInteger(numericClassId)) {
    return `Class ${numericClassId}`;
  }
  return String(classKey ?? "");
}

function collectCurveClassOptions(curves) {
  const optionMap = new Map();
  (curves ?? []).forEach((curve) => {
    (curve.classOptions ?? []).forEach((option) => {
      if (!option?.key || optionMap.has(option.key)) {
        return;
      }
      optionMap.set(option.key, option.label || defaultClassLabel(option.key));
    });
  });
  return Array.from(optionMap.entries()).map(([key, label]) => ({ key, label }));
}

function resolveCurveClassLabel(classOptions, classKey) {
  return classOptions.find((option) => option.key === classKey)?.label ?? defaultClassLabel(classKey);
}

function toXYXY(box) {
  return {
    x1: box.xc - box.w / 2,
    y1: box.yc - box.h / 2,
    x2: box.xc + box.w / 2,
    y2: box.yc + box.h / 2,
  };
}

function boxIoU(a, b) {
  const boxA = toXYXY(a);
  const boxB = toXYXY(b);
  const interW = Math.max(0, Math.min(boxA.x2, boxB.x2) - Math.max(boxA.x1, boxB.x1));
  const interH = Math.max(0, Math.min(boxA.y2, boxB.y2) - Math.max(boxA.y1, boxB.y1));
  const interArea = interW * interH;
  const areaA = Math.max(0, boxA.x2 - boxA.x1) * Math.max(0, boxA.y2 - boxA.y1);
  const areaB = Math.max(0, boxB.x2 - boxB.x1) * Math.max(0, boxB.y2 - boxB.y1);
  const union = areaA + areaB - interArea;
  return union > 0 ? interArea / union : 0;
}

function buildPredictionAnalysis(preview) {
  const groundTruth = preview?.boxes ?? [];
  const predictions = preview?.predictions ?? [];
  const matchedGt = new Set();
  const matchedPred = new Set();
  const matches = [];

  groundTruth.forEach((gtBox, gtIndex) => {
    let bestIndex = -1;
    let bestIoU = 0;
    predictions.forEach((predBox, predIndex) => {
      if (matchedPred.has(predIndex) || predBox.class_id !== gtBox.class_id) {
        return;
      }
      const iou = boxIoU(gtBox, predBox);
      if (iou >= 0.3 && iou > bestIoU) {
        bestIoU = iou;
        bestIndex = predIndex;
      }
    });
    if (bestIndex >= 0) {
      matchedGt.add(gtIndex);
      matchedPred.add(bestIndex);
      matches.push({
        gtBox,
        predBox: predictions[bestIndex],
        iou: bestIoU,
      });
    }
  });

  const truePositives = matches.length;
  const falseNegatives = Math.max(0, groundTruth.length - truePositives);
  const falsePositives = Math.max(0, predictions.length - truePositives);
  const averageConfidence = predictions.length
    ? predictions.reduce((sum, item) => sum + Number(item.confidence ?? 0), 0) / predictions.length
    : 0;
  const averageIoU = matches.length
    ? matches.reduce((sum, item) => sum + item.iou, 0) / matches.length
    : 0;

  return {
    truePositives,
    falseNegatives,
    falsePositives,
    precision: predictions.length ? truePositives / predictions.length : 0,
    recall: groundTruth.length ? truePositives / groundTruth.length : 0,
    averageConfidence,
    averageIoU,
    matches,
  };
}

function formatMetricValue(metricKey, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  if (metricKey.includes("map") || metricKey.includes("recall")) {
    return formatPercent(value);
  }
  return formatDecimal(value, 3);
}

function formatBytes(value) {
  const bytes = Number(value ?? 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 o";
  }
  const units = ["o", "Ko", "Mo", "Go", "To"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const scaled = bytes / 1024 ** exponent;
  return `${formatDecimal(scaled, exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function classColor(classId) {
  const palette = [
    "#c8742f",
    "#3f566b",
    "#8b4c18",
    "#5e7e95",
    "#d5965d",
    "#516879",
    "#a5571f",
    "#758998"
  ];
  return palette[Math.abs(Number(classId ?? 0)) % palette.length];
}


function DatasetsPage({ apiFetch }) {
  const [datasetPath, setDatasetPath] = useState(DEFAULT_DATASET_PATH);
  const [statsState, setStatsState] = useState({ status: "idle", data: null, error: null });
  const [examplesState, setExamplesState] = useState({ status: "idle", data: null, error: null });
  const [previewState, setPreviewState] = useState({ status: "idle", data: null, error: null });
  const [activeSplit, setActiveSplit] = useState("train");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [cfgIndex, setCfgIndex] = useState(0);
  const [showBoxes, setShowBoxes] = useState(true);
  const [hoveredBoxIndex, setHoveredBoxIndex] = useState(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);

  async function loadExamples(path, split, { reset = false } = {}) {
    const previousSamples = reset ? [] : (examplesState.data?.samples ?? []);
    const offset = reset ? 0 : previousSamples.length;
    setExamplesState((current) => ({
      status: reset ? "loading" : "loading_more",
      data: reset ? null : current.data,
      error: null
    }));

    try {
      const response = await apiFetch(
        `/dataset/examples?path=${encodeURIComponent(path)}&split=${encodeURIComponent(split)}&offset=${offset}&limit=${EXAMPLES_PAGE_SIZE}`
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(extractApiError(data, response.status).message);
      }
      const mergedSamples = reset ? data.samples : [...previousSamples, ...data.samples];
      setExamplesState({
        status: "ready",
        data: { ...data, samples: mergedSamples, loaded: mergedSamples.length },
        error: null
      });
      if (reset) {
        setCurrentIndex(0);
        setCfgIndex(0);
      }
      return data.samples.length;
    } catch (error) {
      setExamplesState((current) => ({
        status: "error",
        data: reset ? null : current.data,
        error: error instanceof Error ? error.message : "Erreur inconnue."
      }));
      if (reset) {
        setPreviewState({ status: "idle", data: null, error: null });
      }
      return 0;
    }
  }

  async function loadStats(path, split = activeSplit) {
    const nextPath = path?.trim();
    if (!nextPath) {
      return;
    }
    setStatsState((current) => ({ ...current, status: "loading", error: null }));

    try {
      const response = await apiFetch(
        `/dataset/stats?path=${encodeURIComponent(nextPath)}&split=${encodeURIComponent(split)}`
      );
      const data = await response.json();
      if (!response.ok) {
        const errorDetail = extractApiError(data, response.status);
        setStatsState({ status: "error", data: null, error: errorDetail });
        setExamplesState({ status: "idle", data: null, error: null });
        setPreviewState({ status: "idle", data: null, error: null });
        return;
      }
      setStatsState({ status: "ready", data, error: null });
      setActiveSplit(data.split);
      await loadExamples(nextPath, data.split, { reset: true });
    } catch (error) {
      setStatsState({
        status: "error",
        data: null,
        error: {
          message: error instanceof Error ? error.message : "Erreur inconnue.",
          raw: null,
          diagnostics: null,
        }
      });
      setExamplesState({ status: "idle", data: null, error: null });
      setPreviewState({ status: "idle", data: null, error: null });
    }
  }

  useEffect(() => {
    void loadStats(datasetPath, activeSplit);
  }, []);

  const data = statsState.data;
  const splitStats = data?.split_stats ?? null;
  const widthHistogram = splitStats?.histograms?.width ?? [];
  const heightHistogram = splitStats?.histograms?.height ?? [];
  const areaHistogram = splitStats?.histograms?.area ?? [];
  const snrHistogram = splitStats?.histograms?.snr ?? [];
  const storageBreakdown = splitStats?.storage?.by_folder ?? [];
  const activeSamples = examplesState.data?.samples ?? [];
  const totalSamples = examplesState.data?.total ?? activeSamples.length;
  const hasMoreSamples = Boolean(examplesState.data?.has_more);
  const currentSample = activeSamples[currentIndex] ?? null;
  const preview = previewState.data;
  const inspectedBox =
    preview?.boxes?.[hoveredBoxIndex ?? -1] ??
    preview?.boxes?.[selectedBoxIndex ?? -1] ??
    preview?.boxes?.[0] ??
    null;

  useEffect(() => {
    async function loadPreview() {
      if (!currentSample) {
        setPreviewState({ status: "idle", data: null, error: null });
        return;
      }
      setPreviewState((current) => ({ status: "loading", data: current.data, error: null }));
      try {
        const response = await apiFetch(
          `/dataset/example?path=${encodeURIComponent(datasetPath)}&split=${encodeURIComponent(activeSplit)}&sample_id=${encodeURIComponent(currentSample.sample_id)}&cfg_index=${cfgIndex}`
        );
        const result = await response.json();
        if (!response.ok) {
          throw new Error(extractApiError(result, response.status).message);
        }
        setPreviewState({ status: "ready", data: result, error: null });
      } catch (error) {
        setPreviewState({
          status: "error",
          data: null,
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    }

    void loadPreview();
  }, [activeSplit, apiFetch, cfgIndex, currentSample, datasetPath]);

  async function goToNextSample() {
    if (!activeSamples.length) {
      return;
    }
    const atLoadedEnd = currentIndex >= activeSamples.length - 1;
    if (!atLoadedEnd) {
      setCurrentIndex((current) => Math.min(current + 1, activeSamples.length - 1));
      return;
    }
    if (!hasMoreSamples || examplesState.status === "loading_more") {
      return;
    }
    const added = await loadExamples(datasetPath, activeSplit, { reset: false });
    if (added > 0) {
      setCurrentIndex((current) => current + 1);
    }
  }

  return (
    <div className="page-stack">
      <header className="hero display-hero">
        <div className="hero-copy">
          <span className="eyebrow">Datasets</span>
          <h1>Cataloguer, verifier et inspecter les jeux de donnees</h1>
          <p>
            Cette page garde la meme densite visuelle que l&apos;analyse dataset du simulateur,
            mais focalisee ici sur les jeux utilises pour detector2026.
          </p>
          <div className="hero-actions">
            <button type="button" className="primary-button" onClick={() => void loadStats(datasetPath, activeSplit)}>
              Charger le dataset
            </button>
            <button type="button" className="ghost-button" onClick={() => setCurrentIndex(0)}>
              Revenir au premier sample
            </button>
          </div>
        </div>

        <aside className="hero-panel dataset-hero-panel">
          <PathField
            label="Dossier du dataset"
            value={datasetPath}
            onChange={setDatasetPath}
            apiFetch={apiFetch}
            pickerKind="directory"
            pickerTitle="Choisir un dossier de dataset"
            placeholder="/abs/path/to/dataset"
          />
          <div className="dataset-path-note">
            Le chemin peut etre absolu ou relatif a la racine du projet.
          </div>
          <label className="field">
            <span>Split visualise</span>
            <select
              value={activeSplit}
              onChange={(event) => {
                const nextSplit = event.target.value;
                setActiveSplit(nextSplit);
                void loadStats(datasetPath, nextSplit);
              }}
            >
              {(data?.split_names ?? ["train", "val"]).map((splitName) => (
                <option key={splitName} value={splitName}>
                  {splitName}
                </option>
              ))}
            </select>
          </label>
          <div className="terminal">
            {statsState.status === "loading" ? <p>Analyse du dataset en cours...</p> : null}
            {statsState.status === "error" ? <p>{statsState.error?.message}</p> : null}
            {statsState.status === "ready" ? (
              <>
                <p>Dataset analyse: {data.dataset_path}</p>
                <p>Split analyse: {data.split}</p>
                <p>
                  {formatCount(data.totals.sample_count)} scenarios et {formatCount(data.totals.box_count)} boites.
                </p>
              </>
            ) : null}
            {statsState.status === "idle" ? <p>Aucune analyse lancee.</p> : null}
          </div>
        </aside>
      </header>

      {statsState.status === "error" ? (
        <section className="dashboard-grid">
          <section className="panel panel-span-12">
            <div className="section-title">
              <span>Diagnostic</span>
              <h2>Structure du dataset et erreur detectee</h2>
              <p>Le front affiche ici les details concrets si le chemin est vide, incorrect ou mal formate.</p>
            </div>
            <pre className="json-preview">
              {statsState.error?.raw ??
                JSON.stringify(
                  {
                    message: statsState.error?.message ?? "Erreur inconnue.",
                    diagnostics: statsState.error?.diagnostics ?? null,
                  },
                  null,
                  2
                )}
            </pre>
          </section>
        </section>
      ) : null}

      <section className="stats-grid detector-stats-grid">
        <article className="stat-card">
          <span>Scenarios</span>
          <strong>{formatCount(data?.totals?.sample_count)}</strong>
          <small>Nombre total d&apos;echantillons du split {activeSplit}</small>
        </article>
        <article className="stat-card">
          <span>Boites</span>
          <strong>{formatCount(data?.totals?.box_count)}</strong>
          <small>Annotations de detection chargees</small>
        </article>
        <article className="stat-card">
          <span>Stockage</span>
          <strong>{formatBytes(data?.totals?.total_bytes)}</strong>
          <small>Volume disque du split selectionne</small>
        </article>
      </section>

      <section className="dashboard-grid">
        <section className="panel panel-span-8">
          <div className="section-title">
            <span>Repartition</span>
            <h2>Classes dominantes</h2>
            <p>Distribution des classes detectees dans le split {activeSplit}.</p>
          </div>
          <RichBarList
            items={data?.class_distribution ?? []}
            labelKey="class_name"
            valueKey="count"
            valueFormatter={(value, item) => `${formatCount(value)} · ${formatPercent(item.ratio)}`}
          />
        </section>
        <section className="panel panel-span-4">
          <div className="section-title">
            <span>PSNR</span>
            <h2>Qualite moyenne par configuration STFT</h2>
            <p>Moyenne, minimum et maximum des PSNR observes selon les resolutions disponibles.</p>
          </div>
          <RichBarList
            items={data?.psnr_overview ?? []}
            labelKey="cfg"
            valueKey="mean"
            valueFormatter={(value, item) =>
              `moy ${formatDecimal(value)} dB · min ${formatDecimal(item.min)} dB · max ${formatDecimal(item.max)} dB`
            }
          />
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>SNR</span>
            <h2>Distribution des SNR</h2>
            <p>Histogramme des niveaux SNR observes dans le split {activeSplit}.</p>
          </div>
          <RichBarList items={snrHistogram} labelKey="label" valueKey="count" />
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Memoire</span>
            <h2>Occupation disque</h2>
            <p>Repartition du volume du split par type de fichier.</p>
          </div>
          <RichBarList
            items={storageBreakdown}
            labelKey="folder"
            valueKey="bytes"
            valueFormatter={(value) => formatBytes(value)}
          />
        </section>
        <section className="panel panel-span-12">
          <div className="section-title">
            <span>Exemples</span>
            <h2>Spectres annotes</h2>
            <p>Visualiseur d&apos;echantillons avec boites englobantes, classes et navigation.</p>
          </div>

          <div className="viewer-toolbar">
            <div className="viewer-nav">
              <button
                type="button"
                className="secondary-button"
                onClick={() => setCurrentIndex((current) => Math.max(0, current - 1))}
                disabled={currentIndex === 0}
              >
                ← Precedent
              </button>
              <strong>{activeSamples.length ? `${currentIndex + 1} / ${totalSamples}` : "0 / 0"}</strong>
              <button
                type="button"
                className="secondary-button"
                onClick={() => void goToNextSample()}
                disabled={!activeSamples.length || (!hasMoreSamples && currentIndex >= activeSamples.length - 1)}
              >
                Suivant →
              </button>
            </div>

            <div className="viewer-configs">
              {(preview?.cfg_labels ?? []).map((label, index) => (
                <button
                  key={label}
                  type="button"
                  className={`nav-tab ${cfgIndex === index ? "nav-tab-active" : ""}`}
                  onClick={() => setCfgIndex(index)}
                >
                  {label}
                </button>
              ))}
              <button type="button" className="ghost-button" onClick={() => setShowBoxes((current) => !current)}>
                {showBoxes ? "Boites visibles" : "Boites masquees"}
              </button>
            </div>
          </div>

          {examplesState.status === "loading" ? <p className="dashboard-empty">Chargement de la liste des exemples...</p> : null}
          {examplesState.status === "loading_more" ? <p className="dashboard-empty">Chargement d&apos;un paquet supplementaire...</p> : null}
          {examplesState.status === "error" ? <p className="dashboard-empty">{examplesState.error}</p> : null}
          {!activeSamples.length && examplesState.status === "ready" ? (
            <p className="dashboard-empty">Aucun exemple disponible pour ce split.</p>
          ) : null}

          {preview ? (
            <div className="sample-viewer">
              <div className="sample-stage">
                <img src={preview.image.data_url} alt={`Spectre ${preview.sample_id}`} className="sample-image" />
                <div className={`sample-overlay ${showBoxes ? "" : "sample-overlay-hidden"}`}>
                  {preview.boxes.map((box, index) => {
                    const color = classColor(box.class_id);
                    const isSelected = selectedBoxIndex === index;
                    const isHovered = hoveredBoxIndex === index;
                    return (
                      <div
                        key={`${box.class_id}-${index}`}
                        className={`sample-box ${isSelected ? "sample-box-selected" : ""} ${isHovered ? "sample-box-hovered" : ""}`}
                        style={{
                          left: `${(box.xc - box.w / 2) * 100}%`,
                          top: `${(box.yc - box.h / 2) * 100}%`,
                          width: `${box.w * 100}%`,
                          height: `${box.h * 100}%`,
                          borderColor: color
                        }}
                        onMouseEnter={() => setHoveredBoxIndex(index)}
                        onMouseLeave={() => setHoveredBoxIndex((current) => (current === index ? null : current))}
                        onClick={() => setSelectedBoxIndex(index)}
                      >
                        <span className="sample-box-label" style={{ background: color }}>
                          {box.class_name}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="sample-sidebar">
                <div className="metric-line">
                  <span>Sample</span>
                  <strong>{preview.sample_id}</strong>
                </div>
                <div className="metric-line">
                  <span>Split</span>
                  <strong>{preview.split}</strong>
                </div>
                <div className="metric-line">
                  <span>Boites</span>
                  <strong>{formatCount(preview.box_count)}</strong>
                </div>
                <div className="metric-line">
                  <span>Resolution</span>
                  <strong>{preview.cfg_labels?.[preview.cfg_index] ?? `cfg${preview.cfg_index}`}</strong>
                </div>

                {inspectedBox ? (
                  <div className="sample-detail-card">
                    <span>Boite selectionnee</span>
                    <strong>{inspectedBox.class_name}</strong>
                    <p>
                      Centre ({formatDecimal(inspectedBox.xc, 3)}, {formatDecimal(inspectedBox.yc, 3)}) ·
                      taille {formatDecimal(inspectedBox.w, 3)} × {formatDecimal(inspectedBox.h, 3)}
                    </p>
                    {inspectedBox.snr !== null && inspectedBox.snr !== undefined ? (
                      <p>SNR: {formatDecimal(inspectedBox.snr, 2)} dB</p>
                    ) : null}
                    {Object.keys(inspectedBox.psnr ?? {}).length ? (
                      <div className="psnr-chip-list">
                        {Object.entries(inspectedBox.psnr).map(([cfgName, cfgValue]) => (
                          <span key={cfgName} className="class-chip">
                            {cfgName}: {formatDecimal(cfgValue, 2)} dB
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <div className="class-chip-list">
                  {preview.class_names.map((className, index) => (
                    <span
                      key={className}
                      className="class-chip"
                      style={{ borderColor: classColor(index), color: classColor(index) }}
                    >
                      {className}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : null}

          {previewState.status === "loading" ? <p className="dashboard-empty">Chargement du spectre...</p> : null}
          {previewState.status === "error" ? <p className="dashboard-empty">{previewState.error}</p> : null}
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Boites</span>
            <h2>Distribution des largeurs</h2>
            <p>Histogramme des largeurs normalisees des boites englobantes.</p>
          </div>
          <RichBarList items={widthHistogram} labelKey="label" valueKey="count" />
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Boites</span>
            <h2>Distribution des hauteurs</h2>
            <p>Histogramme des hauteurs normalisees des boites englobantes.</p>
          </div>
          <RichBarList items={heightHistogram} labelKey="label" valueKey="count" />
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Boites</span>
            <h2>Distribution des surfaces</h2>
            <p>Histogramme des surfaces normalisees des boites englobantes.</p>
          </div>
          <RichBarList items={areaHistogram} labelKey="label" valueKey="count" />
        </section>
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Resume</span>
            <h2>Mesures des boites</h2>
            <p>Quelques statistiques descriptives utiles pour controler la coherence du split.</p>
          </div>
          <div className="metric-stack">
            <div className="metric-line">
              <span>Largeur mediane</span>
              <strong>{formatDecimal(splitStats?.box_metrics?.width?.median)}</strong>
            </div>
            <div className="metric-line">
              <span>Hauteur mediane</span>
              <strong>{formatDecimal(splitStats?.box_metrics?.height?.median)}</strong>
            </div>
            <div className="metric-line">
              <span>Surface mediane</span>
              <strong>{formatDecimal(splitStats?.box_metrics?.area?.median)}</strong>
            </div>
            <div className="metric-line">
              <span>SNR moyen</span>
              <strong>{formatDecimal(splitStats?.box_metrics?.snr?.mean)} dB</strong>
            </div>
            <div className="metric-line">
              <span>Boites / scenario</span>
              <strong>{formatDecimal(splitStats?.box_metrics?.boxes_per_sample?.mean)}</strong>
            </div>
          </div>
        </section>
      </section>
    </div>
  );
}

function TrainingPage({ apiFetch }) {
  const [modelsState, setModelsState] = useState({ status: "loading", models: [], error: null });
  const [runsState, setRunsState] = useState({ status: "loading", runs: [], resources: null, error: null });
  const [datasetState, setDatasetState] = useState({ status: "idle", valid: false, errors: [], numClasses: null, diagnostics: null });
  const [launchError, setLaunchError] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [formState, setFormState] = useState({
    runName: "detector2026-run",
    datasetPath: DEFAULT_DATASET_PATH,
    outputRoot: "./runs/",
    epochs: 300,
    batchSize: 64,
    patience: 30,
    learningRate: "1e-3",
    datasetMode: "fused",
    numClasses: "20",
    widthMult: "0.5",
    regMax: "16",
    backboneMode: "TFSep_pyramid",
    outfusionChannelsMult: "1",
    device: "cuda:0",
    resKey: "cfg512",
    resWidth: "256",
    resHeight: "256",
  });

  const selectedModel = modelsState.models.find((model) => model.id === selectedModelId) ?? null;
  const activeRuns = runsState.runs.filter((run) => run.status === "running" || run.status === "finishing");

  async function refreshRuns() {
    try {
      const response = await apiFetch("/training/runs");
      const data = await response.json();
      if (!response.ok) {
        throw new Error(extractApiError(data, response.status).message);
      }
      setRunsState({ status: "ready", runs: data.runs ?? [], resources: data.resources ?? null, error: null });
    } catch (error) {
      setRunsState((current) => ({
        ...current,
        status: "error",
        error: error instanceof Error ? error.message : "Erreur inconnue."
      }));
    }
  }

  useEffect(() => {
    async function loadModels() {
      try {
        const response = await apiFetch("/training/models");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(extractApiError(data, response.status).message);
        }
        const models = data.models ?? [];
        setModelsState({ status: "ready", models, error: null });
        if (models.length) {
          setSelectedModelId(models[0].id);
          const defaults = models[0].default_config ?? {};
          setFormState((current) => ({
            ...current,
            widthMult: String(defaults.width_mult ?? current.widthMult),
            regMax: String(defaults.reg_max ?? current.regMax),
            backboneMode: String(defaults.backbone_mode ?? current.backboneMode),
            outfusionChannelsMult: String(defaults.outfusion_channels_mult ?? current.outfusionChannelsMult),
            device: String(defaults.device ?? current.device),
            resKey: String(defaults.res_key ?? current.resKey),
            resWidth: String(defaults.res_hw?.[0] ?? current.resWidth),
            resHeight: String(defaults.res_hw?.[1] ?? current.resHeight),
            datasetMode: String(models[0].dataset_modes?.[0] ?? current.datasetMode),
          }));
        }
      } catch (error) {
        setModelsState({
          status: "error",
          models: [],
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    }

    void loadModels();
  }, [apiFetch]);

  useEffect(() => {
    let stopped = false;

    void refreshRuns();
    const timer = setInterval(() => {
      if (!stopped) {
        void refreshRuns();
      }
    }, 4000);

    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [apiFetch]);

  useEffect(() => {
    const datasetPath = String(formState.datasetPath ?? "").trim();
    if (!datasetPath) {
      setDatasetState({ status: "idle", valid: false, errors: [], numClasses: null, diagnostics: null });
      return undefined;
    }

    const timer = setTimeout(() => {
      void (async () => {
        setDatasetState((current) => ({ ...current, status: "checking" }));
        try {
          const response = await apiFetch(`/training/dataset-info?path=${encodeURIComponent(datasetPath)}`);
          const data = await response.json();
          if (!response.ok) {
            throw new Error(extractApiError(data, response.status).message);
          }
          const nextState = {
            status: "ready",
            valid: Boolean(data.valid),
            errors: data.errors ?? [],
            numClasses: data.num_classes ?? null,
            diagnostics: data.diagnostics ?? null,
          };
          setDatasetState(nextState);
          if (data.valid && Number.isFinite(Number(data.num_classes)) && Number(data.num_classes) > 0) {
            setFormState((current) => ({ ...current, numClasses: String(data.num_classes) }));
          }
        } catch (error) {
          setDatasetState({
            status: "error",
            valid: false,
            errors: [error instanceof Error ? error.message : "Validation dataset indisponible."],
            numClasses: null,
            diagnostics: null,
          });
        }
      })();
    }, 350);

    return () => clearTimeout(timer);
  }, [apiFetch, formState.datasetPath]);

  function applyModelSelection(modelId) {
    setSelectedModelId(modelId);
    const model = modelsState.models.find((item) => item.id === modelId);
    if (!model) {
      return;
    }
    const defaults = model.default_config ?? {};
    setFormState((current) => ({
      ...current,
      widthMult: String(defaults.width_mult ?? current.widthMult),
      regMax: String(defaults.reg_max ?? current.regMax),
      backboneMode: String(defaults.backbone_mode ?? current.backboneMode),
      outfusionChannelsMult: String(defaults.outfusion_channels_mult ?? current.outfusionChannelsMult),
      device: String(defaults.device ?? current.device),
      resKey: String(defaults.res_key ?? current.resKey),
      resWidth: String(defaults.res_hw?.[0] ?? current.resWidth),
      resHeight: String(defaults.res_hw?.[1] ?? current.resHeight),
      datasetMode: String(model.dataset_modes?.[0] ?? current.datasetMode),
    }));
  }

  async function launchTraining() {
    if (!selectedModel) {
      return;
    }
    if (!datasetState.valid) {
      throw new Error("Le dataset n'est pas valide. Corrige le chemin avant de lancer l'entrainement.");
    }
    setLaunchError(null);
    const payload = {
      run_name: formState.runName,
      model_id: selectedModel.id,
      dataset_path: formState.datasetPath,
      output_root: formState.outputRoot,
      training: {
        epochs: Number(formState.epochs),
        batch_size: Number(formState.batchSize),
        patience: Number(formState.patience),
        learning_rate: Number(formState.learningRate),
        dataset_mode: formState.datasetMode,
      },
      model_config: {
        num_classes: Number(formState.numClasses),
        width_mult: Number(formState.widthMult),
        reg_max: Number(formState.regMax),
        backbone_mode: formState.backboneMode,
        outfusion_channels_mult: Number(formState.outfusionChannelsMult),
        device: formState.device,
        res_key: formState.resKey,
        res_hw: [Number(formState.resWidth), Number(formState.resHeight)],
      },
    };

    const response = await apiFetch("/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(extractApiError(data, response.status).message);
    }
    setRunsState((current) => ({
      ...current,
      runs: [data, ...(current.runs ?? [])],
    }));
  }

  async function stopTraining(runId) {
    const response = await apiFetch(`/training/${encodeURIComponent(runId)}/cancel`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(extractApiError(data, response.status).message);
    }
    await refreshRuns();
  }

  async function handleLaunchTraining() {
    try {
      await launchTraining();
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Erreur inconnue.");
    }
  }

  const datasetIndicatorClass =
    datasetState.status === "checking"
      ? "dataset-health-dot-checking"
      : datasetState.valid
        ? "dataset-health-dot-valid"
        : datasetState.status === "idle"
          ? "dataset-health-dot-idle"
          : "dataset-health-dot-invalid";
  const datasetIndicatorTitle =
    datasetState.status === "checking"
      ? "Validation du dataset en cours..."
      : datasetState.valid
        ? `Dataset valide. ${datasetState.numClasses ?? "-"} classes detectees.`
        : (datasetState.errors ?? []).join("\n") || "Dataset invalide.";

  return (
    <div className="page-stack">
      <header className="hero training-hero">
        <div className="hero-copy">
          <span className="eyebrow">Training</span>
          <h1>Lancer un entrainement sans se perdre dans la configuration</h1>
          <p>
            Cet ecran est repense pour rester simple: choix du modele, reglage des
            parametres essentiels, lancement du run et suivi immediat des ressources.
          </p>
        </div>
        <aside className="hero-panel training-summary-panel">
          <div className="section-title">
            <span>Etat live</span>
            <h2>Vue immediate</h2>
            <p>Lis en un coup d&apos;oeil les runs actifs et la charge CPU/GPU du poste.</p>
          </div>
          <div className="stats-grid training-live-grid">
            <article className="stat-card">
              <span>Runs actifs</span>
              <strong>{formatCount(activeRuns.length)}</strong>
              <small>jobs en execution ou finalisation</small>
            </article>
            <article className="stat-card">
              <span>CPU</span>
              <strong>{formatDecimal(runsState.resources?.cpu?.utilization_percent)}%</strong>
              <small>{formatCount(runsState.resources?.cpu?.logical_cores)} coeurs logiques</small>
            </article>
            <article className="stat-card">
              <span>GPU principal</span>
              <strong>{runsState.resources?.gpus?.[0]?.device ?? "cpu"}</strong>
              <small>{formatDecimal(runsState.resources?.gpus?.[0]?.memory_utilization_percent)}% memoire</small>
            </article>
          </div>
        </aside>
      </header>

      <section className="dashboard-grid training-dashboard-grid">
        <section className="panel panel-span-8 training-config-panel">
          <div className="section-title">
            <span>Configuration</span>
            <h2>Choisir un modele et regler l&apos;essentiel</h2>
            <p>Le backend fournit les modeles disponibles et leurs options de base.</p>
          </div>

          {modelsState.status === "error" ? <p className="dashboard-empty">{modelsState.error}</p> : null}
          {launchError ? <div className="training-error-banner">{launchError}</div> : null}

          <div className="training-form-grid">
            <label className="field field-full-span">
              <span>Modele</span>
              <select value={selectedModelId} onChange={(event) => applyModelSelection(event.target.value)}>
                {modelsState.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field field-full-span">
              <span>Nom du run</span>
              <input value={formState.runName} onChange={(event) => setFormState((current) => ({ ...current, runName: event.target.value }))} />
            </label>

            <PathField
              className="field-full-span"
              label={(
                <span className="field-label-row">
                  <span>Dataset</span>
                  <span className="dataset-health-indicator" title={datasetIndicatorTitle} aria-label={datasetIndicatorTitle}>
                    <span className={`dataset-health-dot ${datasetIndicatorClass}`} />
                  </span>
                </span>
              )}
              value={formState.datasetPath}
              onChange={(nextValue) => setFormState((current) => ({ ...current, datasetPath: nextValue }))}
              apiFetch={apiFetch}
              pickerKind="directory"
              pickerTitle="Choisir un dataset pour l'entrainement"
              note="Saisie libre ou selection native du dataset d'entrainement."
            />

            <PathField
              className="field-full-span"
              label="Dossier de sortie"
              value={formState.outputRoot}
              onChange={(nextValue) => setFormState((current) => ({ ...current, outputRoot: nextValue }))}
              apiFetch={apiFetch}
              pickerKind="directory"
              pickerTitle="Choisir un dossier de sortie"
            />

            <label className="field">
              <span>Mode dataset</span>
              <select value={formState.datasetMode} onChange={(event) => setFormState((current) => ({ ...current, datasetMode: event.target.value }))}>
                {(selectedModel?.dataset_modes ?? []).map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Num classes</span>
              <input value={formState.numClasses} readOnly />
            </label>

            <label className="field">
              <span>Device</span>
              <select value={formState.device} onChange={(event) => setFormState((current) => ({ ...current, device: event.target.value }))}>
                {(selectedModel?.options?.devices ?? ["cpu"]).map((device) => (
                  <option key={device} value={device}>
                    {device}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Epochs</span>
              <input type="number" value={formState.epochs} onChange={(event) => setFormState((current) => ({ ...current, epochs: event.target.value }))} />
            </label>

            <label className="field">
              <span>Batch size</span>
              <input type="number" value={formState.batchSize} onChange={(event) => setFormState((current) => ({ ...current, batchSize: event.target.value }))} />
            </label>

            <label className="field">
              <span>Patience</span>
              <input type="number" value={formState.patience} onChange={(event) => setFormState((current) => ({ ...current, patience: event.target.value }))} />
            </label>

            <label className="field">
              <span>Learning rate</span>
              <input value={formState.learningRate} onChange={(event) => setFormState((current) => ({ ...current, learningRate: event.target.value }))} />
            </label>

            <label className="field">
              <span>Width mult</span>
              <input value={formState.widthMult} onChange={(event) => setFormState((current) => ({ ...current, widthMult: event.target.value }))} />
            </label>

            <label className="field">
              <span>Reg max</span>
              <input type="number" value={formState.regMax} onChange={(event) => setFormState((current) => ({ ...current, regMax: event.target.value }))} />
            </label>

            {selectedModel?.id === "mr_yolo" ? (
              <>
                <label className="field">
                  <span>Backbone</span>
                  <select value={formState.backboneMode} onChange={(event) => setFormState((current) => ({ ...current, backboneMode: event.target.value }))}>
                    {(selectedModel?.options?.backbone_modes ?? []).map((mode) => (
                      <option key={mode} value={mode}>
                        {mode}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Outfusion channels</span>
                  <input type="number" value={formState.outfusionChannelsMult} onChange={(event) => setFormState((current) => ({ ...current, outfusionChannelsMult: event.target.value }))} />
                </label>
              </>
            ) : null}

            {selectedModel?.id !== "mr_yolo" ? (
              <>
                <label className="field">
                  <span>Resolution key</span>
                  <select value={formState.resKey} onChange={(event) => setFormState((current) => ({ ...current, resKey: event.target.value }))}>
                    {(selectedModel?.options?.res_keys ?? ["cfg512"]).map((key) => (
                      <option key={key} value={key}>
                        {key}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Resolution H</span>
                  <input type="number" value={formState.resWidth} onChange={(event) => setFormState((current) => ({ ...current, resWidth: event.target.value }))} />
                </label>
                <label className="field">
                  <span>Resolution W</span>
                  <input type="number" value={formState.resHeight} onChange={(event) => setFormState((current) => ({ ...current, resHeight: event.target.value }))} />
                </label>
              </>
            ) : null}
          </div>

          <div className="hero-actions training-actions">
            <button type="button" className="primary-button" onClick={() => void handleLaunchTraining()}>
              Lancer l&apos;entrainement
            </button>
          </div>
        </section>

        <section className="panel panel-span-4 training-resources-panel">
          <div className="section-title">
            <span>Ressources</span>
            <h2>CPU / GPU</h2>
            <p>Lecture simple de l&apos;etat machine pendant les runs.</p>
          </div>
          <div className="metric-stack training-resource-stack">
            <div className="metric-line">
              <span>CPU usage</span>
              <strong>{formatDecimal(runsState.resources?.cpu?.utilization_percent)}%</strong>
            </div>
            <div className="metric-line">
              <span>CPU loadavg</span>
              <strong>{(runsState.resources?.cpu?.loadavg ?? []).join(" / ") || "-"}</strong>
            </div>
            {(runsState.resources?.gpus ?? []).map((gpu) => (
              <div key={gpu.device} className="metric-line">
                <span>{gpu.name}</span>
                <strong>
                  {gpu.device} · {formatDecimal(gpu.memory_utilization_percent)}% memoire
                </strong>
              </div>
            ))}
          </div>
        </section>

        <section className="panel panel-span-12 training-runs-panel">
          <div className="section-title">
            <span>Runs</span>
            <h2>Entrainements en cours</h2>
            <p>Vue temps reel des runs lances depuis l&apos;application.</p>
          </div>
          {runsState.error ? <p className="dashboard-empty">{runsState.error}</p> : null}
          <div className="training-run-list">
            {(runsState.runs ?? []).map((run) => (
              <article key={run.run_id} className="training-run-card">
                <div className="training-run-head">
                  <strong>{run.run_name}</strong>
                  <span
                    className={`status-badge ${
                      run.status === "running" || run.status === "finishing"
                        ? "status-running"
                        : run.status === "canceled"
                          ? "status-idle"
                          : run.status === "failed"
                            ? "status-error"
                          : "status-done"
                    }`}
                  >
                    {run.status}
                  </span>
                </div>
                <p>{run.model_id} · {run.dataset_path}</p>
                <small>Racine runs: {run.output_root ?? "./runs/"}</small>
                <small>Sortie: {run.output_dir ?? "-"}</small>
                {run.live_status ? (
                  <section className="training-live-run-panel">
                    <div className="training-live-run-head">
                      <div>
                        <span className="training-live-run-kicker">Suivi live</span>
                        <strong>
                          Epoch {run.live_status.epoch}/{run.live_status.epochs_total}
                        </strong>
                      </div>
                      <span className={`training-phase-chip training-phase-${run.live_status.phase ?? "idle"}`}>
                        {run.live_status.phase === "train"
                          ? "train"
                          : run.live_status.phase === "val"
                            ? "validation"
                            : "attente"}
                      </span>
                    </div>
                    <div className="training-live-run-grid">
                      <article>
                        <span>Batch</span>
                        <strong>
                          {formatCount(run.live_status.batch_current)}/{formatCount(run.live_status.batch_total)}
                        </strong>
                      </article>
                      <article>
                        <span>Progression epoch</span>
                        <strong>{formatCount(run.live_status.epoch_progress_percent)}%</strong>
                      </article>
                      <article>
                        <span>{run.live_status.metric_name === "val_loss" ? "Val loss" : "Loss"}</span>
                        <strong>{run.live_status.metric_value ?? "-"}</strong>
                      </article>
                    </div>
                    <div className="training-live-bars">
                      <div>
                        <div className="training-live-bar-label">
                          <span>Dans l&apos;epoch</span>
                          <strong>{formatCount(run.live_status.epoch_progress_percent)}%</strong>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${run.live_status.epoch_progress_percent ?? 0}%` }} />
                        </div>
                      </div>
                      <div>
                        <div className="training-live-bar-label">
                          <span>Run global</span>
                          <strong>{run.progress ?? 0}%</strong>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${run.progress ?? 0}%` }} />
                        </div>
                      </div>
                    </div>
                    <small>
                      {run.live_status.elapsed ? `Temps ecoule: ${run.live_status.elapsed}` : "Suivi live en cours"}
                    </small>
                  </section>
                ) : (
                  <>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${run.progress ?? 0}%` }} />
                    </div>
                    <small>{run.progress ?? 0}%</small>
                  </>
                )}
                {run.status === "failed" ? (
                  <div className="training-run-error">
                    <strong>Echec du run</strong>
                    <p>{run.error_message ?? "Le worker d'entrainement s'est termine anormalement."}</p>
                    {run.log_tail ? (
                      <details className="training-run-log-details">
                        <summary>Dernieres lignes du log</summary>
                        <pre>{run.log_tail}</pre>
                      </details>
                    ) : null}
                  </div>
                ) : null}
                {run.status === "running" || run.status === "finishing" ? (
                  <div className="training-run-actions">
                    <button
                      type="button"
                      className="ghost-button training-stop-button"
                      onClick={() => void stopTraining(run.run_id)}
                    >
                      Stopper
                    </button>
                  </div>
                ) : null}
              </article>
            ))}
            {!runsState.runs?.length ? <p className="dashboard-empty">Aucun entrainement lance pour le moment.</p> : null}
          </div>
        </section>
      </section>
    </div>
  );
}

function EvaluationPage({ apiFetch }) {
  const [runInputs, setRunInputs] = useState(DEFAULT_EVALUATION_RUN_INPUTS);
  const [loadedRunsState, setLoadedRunsState] = useState({ status: "idle", runs: [], error: null });
  const [visualizationId, setVisualizationId] = useState("map_vs_model_cost");
  const [scatterXAxis, setScatterXAxis] = useState("flops");
  const [scatterMetricSelection, setScatterMetricSelection] = useState({
    map50_95: true,
    map50: false,
  });
  const [recallSNRState, setRecallSNRState] = useState({ status: "idle", curves: [], error: null });
  const [f1StatsState, setF1StatsState] = useState({ status: "idle", curves: [], error: null });
  const [evaluationCurveScope, setEvaluationCurveScope] = useState("global");
  const [evaluationClassKey, setEvaluationClassKey] = useState("");
  const [confusionSnrBand, setConfusionSnrBand] = useState("low_snr");
  const [confusionMode, setConfusionMode] = useState("single");
  const [confusionNormalization, setConfusionNormalization] = useState("none");
  const [confusionPrimaryRunLabel, setConfusionPrimaryRunLabel] = useState("");
  const [confusionReferenceRunLabel, setConfusionReferenceRunLabel] = useState("");
  const [confusionMatrixState, setConfusionMatrixState] = useState({ status: "idle", matrices: [], error: null });

  const selectedVisualization = EVALUATION_VIEWS.find((item) => item.id === visualizationId) ?? EVALUATION_VIEWS[0];
  const selectedScatterMetricKeys = EVALUATION_SCATTER_METRIC_OPTIONS
    .filter((item) => scatterMetricSelection[item.id])
    .map((item) => item.id);
  const currentCurveClassOptions = visualizationId === "best_recall_vs_snr"
    ? collectCurveClassOptions(recallSNRState.curves)
    : visualizationId === "f1_threshold_curves"
      ? collectCurveClassOptions(f1StatsState.curves)
      : [];

  async function loadRunsFromInputs() {
    const entries = runInputs
      .map((item, index) => ({
        id: item.id ?? `run-${index}`,
        path: String(item.path ?? "").trim(),
        label: String(item.label ?? "").trim(),
      }))
      .filter((item) => item.path);

    if (!entries.length) {
      setLoadedRunsState({ status: "error", runs: [], error: "Ajoute au moins un chemin de run avant de charger." });
      return;
    }

    setLoadedRunsState({ status: "loading", runs: [], error: null });
    try {
      const runs = await Promise.all(
        entries.map(async (entry, index) => {
          const response = await apiFetch(`/evaluation/run?path=${encodeURIComponent(entry.path)}`);
          const data = await response.json();
          if (!response.ok) {
            throw new Error(`${entry.path}: ${extractApiError(data, response.status).message}`);
          }
          return {
            id: entry.id || `loaded-${index}`,
            path: entry.path,
            label: entry.label || data.summary?.run_name || `Run ${index + 1}`,
            detail: data,
          };
        })
      );
      setLoadedRunsState({ status: "ready", runs, error: null });
    } catch (error) {
      setLoadedRunsState({
        status: "error",
        runs: [],
        error: error instanceof Error ? error.message : "Erreur inconnue."
      });
    }
  }

  useEffect(() => {
    void loadRunsFromInputs();
  }, []);

  useEffect(() => {
    if (visualizationId !== "best_recall_vs_snr" || !loadedRunsState.runs.length) {
      setRecallSNRState({ status: "idle", curves: [], error: null });
      return;
    }

    void (async () => {
      setRecallSNRState({ status: "loading", curves: [], error: null });
      try {
        const curves = await Promise.all(
          loadedRunsState.runs.map(async (run) => {
            const epoch = run.detail?.summary?.best_snapshots?.avg_recall_low_snr?.epoch;
            if (!epoch) {
              throw new Error(`Aucun epoch de best recall disponible pour ${run.label}.`);
            }
            const response = await apiFetch(
              `/evaluation/run/recall-snr?path=${encodeURIComponent(run.path)}&epoch=${encodeURIComponent(epoch)}`
            );
            const data = await response.json();
            if (!response.ok) {
              throw new Error(`${run.label}: ${extractApiError(data, response.status).message}`);
            }
            return {
              label: run.label,
              epoch,
              global: data.global ?? null,
              perClass: data.per_class ?? {},
              classOptions: data.class_options ?? [],
            };
          })
        );
        setRecallSNRState({ status: "ready", curves, error: null });
      } catch (error) {
        setRecallSNRState({
          status: "error",
          curves: [],
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    })();
  }, [apiFetch, loadedRunsState.runs, visualizationId]);

  useEffect(() => {
    if (visualizationId !== "f1_threshold_curves" || !loadedRunsState.runs.length) {
      setF1StatsState({ status: "idle", curves: [], error: null });
      return;
    }

    void (async () => {
      setF1StatsState({ status: "loading", curves: [], error: null });
      try {
        const curves = await Promise.all(
          loadedRunsState.runs.map(async (run) => {
            const epoch = run.detail?.summary?.best_snapshots?.checkpoint?.epoch;
            if (!epoch) {
              throw new Error(`Aucun epoch de best checkpoint disponible pour ${run.label}.`);
            }
            const response = await apiFetch(
              `/evaluation/run/f1-stats?path=${encodeURIComponent(run.path)}&epoch=${encodeURIComponent(epoch)}`
            );
            const data = await response.json();
            if (!response.ok) {
              throw new Error(`${run.label}: ${extractApiError(data, response.status).message}`);
            }
            return {
              label: run.label,
              epoch,
              global: data.global ?? null,
              perClass: data.per_class ?? {},
              classOptions: data.class_options ?? [],
            };
          })
        );
        setF1StatsState({ status: "ready", curves, error: null });
      } catch (error) {
        setF1StatsState({
          status: "error",
          curves: [],
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    })();
  }, [apiFetch, loadedRunsState.runs, visualizationId]);

  useEffect(() => {
    if (evaluationCurveScope !== "per_class") {
      return;
    }
    if (!currentCurveClassOptions.length) {
      setEvaluationClassKey("");
      return;
    }
    setEvaluationClassKey((current) => (
      currentCurveClassOptions.some((option) => option.key === current)
        ? current
        : currentCurveClassOptions[0].key
    ));
  }, [evaluationCurveScope, visualizationId, recallSNRState.curves, f1StatsState.curves]);

  useEffect(() => {
    const matrices = confusionMatrixState.matrices ?? [];
    if (!matrices.length) {
      setConfusionPrimaryRunLabel("");
      setConfusionReferenceRunLabel("");
      return;
    }

    setConfusionPrimaryRunLabel((current) => {
      if (current && matrices.some((item) => item.label === current)) {
        return current;
      }
      return matrices[0].label;
    });

    setConfusionReferenceRunLabel((current) => {
      if (
        current &&
        matrices.some((item) => item.label === current) &&
        current !== (confusionPrimaryRunLabel || matrices[0].label)
      ) {
        return current;
      }
      const fallback = matrices.find((item) => item.label !== (confusionPrimaryRunLabel || matrices[0].label));
      return fallback?.label ?? "";
    });
  }, [confusionMatrixState.matrices, confusionPrimaryRunLabel]);

  useEffect(() => {
    if (visualizationId !== "confusion_matrices" || !loadedRunsState.runs.length) {
      setConfusionMatrixState({ status: "idle", matrices: [], error: null });
      return;
    }

    void (async () => {
      setConfusionMatrixState({ status: "loading", matrices: [], error: null });
      try {
        const matrices = await Promise.all(
          loadedRunsState.runs.map(async (run) => {
            const epoch = run.detail?.summary?.best_snapshots?.checkpoint?.epoch;
            if (!epoch) {
              throw new Error(`Aucun epoch de best checkpoint disponible pour ${run.label}.`);
            }
            const response = await apiFetch(
              `/evaluation/run/confusion-matrices?path=${encodeURIComponent(run.path)}&epoch=${encodeURIComponent(epoch)}`
            );
            const data = await response.json();
            if (!response.ok) {
              throw new Error(`${run.label}: ${extractApiError(data, response.status).message}`);
            }
            return {
              label: run.label,
              epoch,
              classLabels: data.class_labels ?? [],
              matrices: data.matrices ?? {},
            };
          })
        );
        setConfusionMatrixState({ status: "ready", matrices, error: null });
      } catch (error) {
        setConfusionMatrixState({
          status: "error",
          matrices: [],
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    })();
  }, [apiFetch, loadedRunsState.runs, visualizationId]);

  function updateRunInput(index, key, value) {
    setRunInputs((current) => current.map((item, itemIndex) => (
      itemIndex === index ? { ...item, [key]: value } : item
    )));
  }

  function addRunInput() {
    setRunInputs((current) => [...current, { path: "", label: "" }]);
  }

  function removeRunInput(index) {
    setRunInputs((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }

  function toggleScatterMetric(metricId) {
    setScatterMetricSelection((current) => {
      const nextValue = !current[metricId];
      const activeCount = Object.values(current).filter(Boolean).length;
      if (!nextValue && activeCount === 1) {
        return current;
      }
      return {
        ...current,
        [metricId]: nextValue,
      };
    });
  }

  const loadedRuns = loadedRunsState.runs ?? [];
  const comparisonRows = loadedRuns.map((run) => {
    const checkpoint = run.detail?.summary?.best_snapshots?.checkpoint;
    return [
      run.label,
      run.detail?.summary?.run_name ?? "-",
      String(checkpoint?.epoch ?? "-"),
      formatMetricValue("map50_95", checkpoint?.metrics?.map50_95),
      formatMetricValue("map50", checkpoint?.metrics?.map50),
      formatLargeNumber(checkpoint?.model_info?.params),
      formatLargeNumber(checkpoint?.model_info?.flops),
    ];
  });

  return (
    <div className="page-stack">
      <header className="hero detector-hero">
        <div className="hero-copy">
          <span className="eyebrow">Evaluation</span>
          <h1>Comparer plusieurs runs a partir de chemins explicites</h1>
          <p>
            Cette page fonctionne en trois temps: tu donnes les chemins et labels des runs,
            tu choisis la visualisation a produire, puis l&apos;ecran rend uniquement le plot
            ou les valeurs demandees.
          </p>
        </div>
      </header>

      <section className="dashboard-grid">
        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Selection</span>
            <h2>Runs a comparer</h2>
            <p>Saisis les chemins complets des runs et le label qui sera utilise dans les plots.</p>
          </div>
          <div className="evaluation-manual-list">
            {runInputs.map((item, index) => (
              <div key={`run-input-${index}`} className="evaluation-manual-card">
                <PathField
                  className="field-full-span"
                  label="Chemin du run"
                  value={item.path}
                  onChange={(nextValue) => updateRunInput(index, "path", nextValue)}
                  apiFetch={apiFetch}
                  pickerKind="directory"
                  pickerTitle="Choisir un dossier de run"
                  placeholder="/Users/.../runs/examples_of_training/tf_attn_yolon_specificres_cfg512"
                />
                <label className="field">
                  <span>Label du plot</span>
                  <input
                    value={item.label}
                    onChange={(event) => updateRunInput(index, "label", event.target.value)}
                    placeholder={`Run ${index + 1}`}
                  />
                </label>
                <div className="evaluation-manual-actions">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => removeRunInput(index)}
                    disabled={runInputs.length === 1}
                  >
                    Supprimer
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="hero-actions evaluation-selection-actions">
            <button type="button" className="secondary-button" onClick={addRunInput}>
              Ajouter un run
            </button>
            <button type="button" className="primary-button" onClick={() => void loadRunsFromInputs()}>
              Charger la selection
            </button>
          </div>
          {loadedRunsState.error ? <div className="training-error-banner">{loadedRunsState.error}</div> : null}
        </section>

        <section className="panel panel-span-6">
          <div className="section-title">
            <span>Visualisation</span>
            <h2>Ce que tu veux voir</h2>
            <p>Choisis une seule vue a la fois, puis ajuste ses options quand c&apos;est necessaire.</p>
          </div>
          <div className="evaluation-visual-picker">
            {EVALUATION_VIEWS.map((view) => (
              <button
                key={view.id}
                type="button"
                className={`evaluation-visual-card ${visualizationId === view.id ? "evaluation-visual-card-active" : ""}`}
                onClick={() => setVisualizationId(view.id)}
              >
                <strong>{view.label}</strong>
              </button>
            ))}
          </div>
          {visualizationId === "map_vs_model_cost" ? (
            <div className="evaluation-visual-options">
              <div className="evaluation-control-group">
                <strong>Axe des abscisses</strong>
                <p>Le scatter plot utilise soit les FLOPs, soit le nombre de parametres.</p>
                <div className="evaluation-chip-row">
                  {EVALUATION_SCATTER_X_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`evaluation-chip ${scatterXAxis === option.id ? "evaluation-chip-active" : ""}`}
                      onClick={() => setScatterXAxis(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="evaluation-control-group">
                <strong>Metriques affichees</strong>
                <p>Coche une ou deux metriques de mAP a tracer sur la meme vue.</p>
                <div className="evaluation-checkbox-grid">
                  {EVALUATION_SCATTER_METRIC_OPTIONS.map((option) => {
                    const isChecked = Boolean(scatterMetricSelection[option.id]);
                    const isLastActive = isChecked && selectedScatterMetricKeys.length === 1;
                    return (
                      <label
                        key={option.id}
                        className={`evaluation-checkbox-card ${isChecked ? "evaluation-checkbox-card-active" : ""}`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleScatterMetric(option.id)}
                          disabled={isLastActive}
                        />
                        <div>
                          <strong>{option.label}</strong>
                          <small>{option.marker === "circle" ? "Marqueur rond" : "Marqueur carre"}</small>
                        </div>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : null}
          {visualizationId === "confusion_matrices" ? (
            <div className="evaluation-visual-options">
              <div className="evaluation-control-group">
                <strong>Mode d&apos;affichage</strong>
                <p>Affiche soit une matrice seule, soit la difference entre deux runs charges.</p>
                <div className="evaluation-chip-row">
                  {EVALUATION_CONFUSION_MODES.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`evaluation-chip ${confusionMode === option.id ? "evaluation-chip-active" : ""}`}
                      onClick={() => setConfusionMode(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="evaluation-control-group">
                <strong>Tranche de SNR</strong>
                <p>Affiche la matrice de confusion calculee sur faible, moyen ou fort SNR.</p>
                <div className="evaluation-chip-row">
                  {EVALUATION_CONFUSION_SNR_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`evaluation-chip ${confusionSnrBand === option.id ? "evaluation-chip-active" : ""}`}
                      onClick={() => setConfusionSnrBand(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="evaluation-control-group">
                <strong>Normalisation</strong>
                <p>Choisis si la matrice doit rester brute, etre normalisee par ligne, ou par colonne.</p>
                <div className="evaluation-chip-row">
                  {EVALUATION_CONFUSION_NORMALIZATION_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`evaluation-chip ${confusionNormalization === option.id ? "evaluation-chip-active" : ""}`}
                      onClick={() => setConfusionNormalization(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="evaluation-control-group">
                <strong>Selection des runs</strong>
                <p>Choisis le run principal A, puis eventuellement un run B pour afficher `A - B`.</p>
                <div className="evaluation-select-grid">
                  <label className="field">
                    <span>Run A</span>
                    <select
                      value={confusionPrimaryRunLabel}
                      onChange={(event) => {
                        const nextLabel = event.target.value;
                        setConfusionPrimaryRunLabel(nextLabel);
                        if (nextLabel === confusionReferenceRunLabel) {
                          const fallback = (confusionMatrixState.matrices ?? []).find((item) => item.label !== nextLabel);
                          setConfusionReferenceRunLabel(fallback?.label ?? "");
                        }
                      }}
                    >
                      {(confusionMatrixState.matrices ?? []).map((item) => (
                        <option key={`primary-${item.label}`} value={item.label}>
                          {item.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  {confusionMode === "difference" ? (
                    <label className="field">
                      <span>Run B</span>
                      <select
                        value={confusionReferenceRunLabel}
                        onChange={(event) => setConfusionReferenceRunLabel(event.target.value)}
                      >
                        {(confusionMatrixState.matrices ?? [])
                          .filter((item) => item.label !== confusionPrimaryRunLabel)
                          .map((item) => (
                            <option key={`reference-${item.label}`} value={item.label}>
                              {item.label}
                            </option>
                          ))}
                      </select>
                    </label>
                  ) : null}
                </div>
              </div>
            </div>
          ) : null}
          {visualizationId === "best_recall_vs_snr" || visualizationId === "f1_threshold_curves" ? (
            <div className="evaluation-visual-options">
              <div className="evaluation-control-group">
                <strong>Portee des courbes</strong>
                <p>Affiche soit la courbe globale, soit la courbe d&apos;une classe particuliere.</p>
                <div className="evaluation-chip-row">
                  {EVALUATION_CURVE_SCOPE_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      className={`evaluation-chip ${evaluationCurveScope === option.id ? "evaluation-chip-active" : ""}`}
                      onClick={() => setEvaluationCurveScope(option.id)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              {evaluationCurveScope === "per_class" ? (
                <div className="evaluation-control-group">
                  <strong>Classe observee</strong>
                  <p>Ce parametre est partage par les vues qui exposent une declinaison par classe.</p>
                  <label className="field">
                    <span>Classe</span>
                    <select
                      value={evaluationClassKey}
                      onChange={(event) => setEvaluationClassKey(event.target.value)}
                      disabled={!currentCurveClassOptions.length}
                    >
                      {!currentCurveClassOptions.length ? (
                        <option value="">Aucune classe disponible</option>
                      ) : null}
                      {currentCurveClassOptions.map((option) => (
                        <option key={option.key} value={option.key}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              ) : null}
            </div>
          ) : null}
        </section>

        <section className="panel panel-span-12">
          <div className="section-title">
            <span>Rendu</span>
            <h2>{selectedVisualization.label}</h2>
            <p>Le contenu change uniquement selon la visualisation selectionnee.</p>
          </div>
          {loadedRunsState.status === "loading" ? <p className="dashboard-empty">Chargement des runs...</p> : null}
          {loadedRunsState.status === "ready" && !loadedRuns.length ? <p className="dashboard-empty">Aucun run charge.</p> : null}
          {loadedRuns.length ? (
            <EvaluationRender
              loadedRuns={loadedRuns}
              visualizationId={visualizationId}
              scatterXAxis={scatterXAxis}
              selectedScatterMetricKeys={selectedScatterMetricKeys}
              recallSNRState={recallSNRState}
              f1StatsState={f1StatsState}
              evaluationCurveScope={evaluationCurveScope}
              evaluationClassKey={evaluationClassKey}
              evaluationClassOptions={currentCurveClassOptions}
              confusionSnrBand={confusionSnrBand}
              confusionMode={confusionMode}
              confusionNormalization={confusionNormalization}
              confusionPrimaryRunLabel={confusionPrimaryRunLabel}
              confusionReferenceRunLabel={confusionReferenceRunLabel}
              confusionMatrixState={confusionMatrixState}
            />
          ) : null}
        </section>

        <section className="panel panel-span-12">
          <div className="section-title">
            <span>Tableau</span>
            <h2>Repere rapide</h2>
            <p>Ce tableau reste visible pour situer rapidement les snapshots `best.pt` charges.</p>
          </div>
          <Table
            headers={["Label", "Run", "Epoch best.pt", "mAP50:95", "mAP50", "Params", "FLOPs"]}
            rows={comparisonRows}
          />
        </section>
      </section>
    </div>
  );
}

function EvaluationRender({
  loadedRuns,
  visualizationId,
  scatterXAxis,
  selectedScatterMetricKeys,
  recallSNRState,
  f1StatsState,
  evaluationCurveScope,
  evaluationClassKey,
  evaluationClassOptions,
  confusionSnrBand,
  confusionMode,
  confusionNormalization,
  confusionPrimaryRunLabel,
  confusionReferenceRunLabel,
  confusionMatrixState
}) {
  if (visualizationId === "map_vs_model_cost") {
    const palette = ["#c8742f", "#3f566b", "#8b4c18", "#5e7e95"];
    const xAxisOption = EVALUATION_SCATTER_X_OPTIONS.find((item) => item.id === scatterXAxis) ?? EVALUATION_SCATTER_X_OPTIONS[0];
    const selectedMetricOptions = EVALUATION_SCATTER_METRIC_OPTIONS.filter((item) => selectedScatterMetricKeys.includes(item.id));
    const yLabel = selectedMetricOptions.length === 1 ? selectedMetricOptions[0].label : "mAP";
    const chartTitle = `${yLabel} vs ${xAxisOption.label}`;
    const chartSubtitle =
      selectedMetricOptions.length === 2
        ? "Chaque run peut afficher deux marqueurs sur le meme axe X: un pour mAP50:95 et un pour mAP50."
        : `Chaque point represente le best checkpoint du run place selon ${xAxisOption.label}.`;

    const runSummaries = loadedRuns
      .map((run, index) => {
        const checkpoint = run.detail?.summary?.best_snapshots?.checkpoint;
        const x = checkpoint?.model_info?.[scatterXAxis];
        if (x === null || x === undefined) {
          return null;
        }
        return {
          label: run.label,
          color: palette[index % palette.length],
          x: Number(x),
          epoch: checkpoint?.epoch,
          metrics: {
            map50_95: checkpoint?.metrics?.map50_95,
            map50: checkpoint?.metrics?.map50,
          },
        };
      })
      .filter(Boolean);

    const points = runSummaries.flatMap((run) =>
      selectedMetricOptions.flatMap((metricOption) => {
        const y = run.metrics?.[metricOption.id];
        if (y === null || y === undefined) {
          return [];
        }
        return [{
          label: `${run.label}${selectedMetricOptions.length > 1 ? ` · ${metricOption.label}` : ""}`,
          runLabel: run.label,
          color: run.color,
          x: run.x,
          y: Number(y),
          epoch: run.epoch,
          metricKey: metricOption.id,
          metricLabel: metricOption.label,
          marker: metricOption.marker,
        }];
      })
    );

    return (
      <div className="evaluation-render-stack">
        <ScatterComparisonChart
          title={chartTitle}
          subtitle={chartSubtitle}
          xLabel={xAxisOption.label}
          yLabel={yLabel}
          xMetricKey={scatterXAxis}
          yMetricKey={selectedMetricOptions[0]?.id ?? "map50_95"}
          selectedMetricKeys={selectedScatterMetricKeys}
          runSummaries={runSummaries}
          points={points}
        />
      </div>
    );
  }

  if (visualizationId === "confusion_matrices") {
    if (confusionMatrixState.status === "loading") {
      return <p className="dashboard-empty">Chargement des matrices de confusion...</p>;
    }
    if (confusionMatrixState.error) {
      return <p className="dashboard-empty">{confusionMatrixState.error}</p>;
    }

    const items = confusionMatrixState.matrices ?? [];
    const primaryItem = items.find((item) => item.label === confusionPrimaryRunLabel) ?? items[0] ?? null;
    const referenceItem = confusionMode === "difference"
      ? items.find((item) => item.label === confusionReferenceRunLabel && item.label !== primaryItem?.label) ?? null
      : null;

    if (!primaryItem) {
      return <p className="dashboard-empty">Aucune matrice de confusion chargee.</p>;
    }
    if (confusionMode === "difference" && !referenceItem) {
      return <p className="dashboard-empty">Charge au moins deux runs pour afficher une difference de matrices.</p>;
    }

    return (
      <div className="evaluation-render-stack">
        <ConfusionMatrixView
          title={`Confusion matrices · ${(EVALUATION_CONFUSION_SNR_OPTIONS.find((item) => item.id === confusionSnrBand)?.label) ?? confusionSnrBand}`}
          mode={confusionMode}
          normalization={confusionNormalization}
          primaryItem={primaryItem}
          referenceItem={referenceItem}
          snrBand={confusionSnrBand}
        />
      </div>
    );
  }

  if (visualizationId === "best_recall_vs_snr") {
    if (recallSNRState.status === "loading") {
      return <p className="dashboard-empty">Chargement des courbes recall vs snr...</p>;
    }
    if (recallSNRState.error) {
      return <p className="dashboard-empty">{recallSNRState.error}</p>;
    }

    const selectedClassLabel = resolveCurveClassLabel(evaluationClassOptions, evaluationClassKey);
    const series = (recallSNRState.curves ?? []).flatMap((curve, index) => {
      const selectedCurve = evaluationCurveScope === "per_class"
        ? curve.perClass?.[evaluationClassKey]
        : curve.global;
      if (!selectedCurve) {
        return [];
      }
      return [{
        label: `${curve.label} · epoch ${curve.epoch}`,
        color: ["#c8742f", "#3f566b", "#8b4c18", "#5e7e95"][index % 4],
        points: (selectedCurve.recall ?? []).map((value, pointIndex) => ({
          x: Number(selectedCurve.snr_bins?.[pointIndex] ?? pointIndex),
          y: Number(value),
        })),
      }];
    });

    if (!series.length && evaluationCurveScope === "per_class") {
      return <p className="dashboard-empty">Aucune courbe recall vs snr disponible pour {selectedClassLabel}.</p>;
    }

    return (
      <div className="evaluation-render-stack">
        <ComparisonChart
          title={evaluationCurveScope === "per_class" ? `Recall vs snr · ${selectedClassLabel}` : "Best recall vs snr"}
          subtitle={
            evaluationCurveScope === "per_class"
              ? "Courbe recall_snr par classe pour le meilleur epoch de recall faible SNR."
              : "Courbe globale recall_snr pour le meilleur epoch de recall faible SNR."
          }
          xLabel="SNR"
          yLabel="Recall"
          metricKey="avg_recall_low_snr"
          series={series}
        />
      </div>
    );
  }

  if (visualizationId === "f1_threshold_curves") {
    if (f1StatsState.status === "loading") {
      return <p className="dashboard-empty">Chargement des courbes f1_stats...</p>;
    }
    if (f1StatsState.error) {
      return <p className="dashboard-empty">{f1StatsState.error}</p>;
    }

    const selectedClassLabel = resolveCurveClassLabel(evaluationClassOptions, evaluationClassKey);
    const selectedCurves = (f1StatsState.curves ?? []).flatMap((curve, index) => {
      const selectedStats = evaluationCurveScope === "per_class"
        ? curve.perClass?.[evaluationClassKey]
        : curve.global;
      if (!selectedStats) {
        return [];
      }
      return [{
        label: `${curve.label} · epoch ${curve.epoch}`,
        color: ["#c8742f", "#3f566b", "#8b4c18", "#5e7e95"][index % 4],
        stats: selectedStats,
      }];
    });

    if (!selectedCurves.length) {
      return (
        <p className="dashboard-empty">
          {evaluationCurveScope === "per_class"
            ? `Aucune courbe f1_stats disponible pour ${selectedClassLabel}.`
            : "Pas assez de donnees pour afficher f1_stats."}
        </p>
      );
    }

    const thresholdRecallSeries = selectedCurves.map((item) => ({
      label: item.label,
      color: item.color,
      points: (item.stats.thr ?? []).map((threshold, pointIndex) => ({
        x: Number(threshold),
        y: Number(item.stats.recall?.[pointIndex] ?? 0),
      })),
    }));
    const thresholdPrecisionSeries = selectedCurves.map((item) => ({
      label: item.label,
      color: item.color,
      points: (item.stats.thr ?? []).map((threshold, pointIndex) => ({
        x: Number(threshold),
        y: Number(item.stats.precision?.[pointIndex] ?? 0),
      })),
    }));
    const thresholdF1Series = selectedCurves.map((item) => ({
      label: item.label,
      color: item.color,
      points: (item.stats.thr ?? []).map((threshold, pointIndex) => ({
        x: Number(threshold),
        y: Number(item.stats.f1?.[pointIndex] ?? 0),
      })),
    }));
    const precisionRecallSeries = selectedCurves.map((item) => ({
      label: item.label,
      color: item.color,
      points: (item.stats.recall ?? []).map((recallValue, pointIndex) => ({
        x: Number(recallValue),
        y: Number(item.stats.precision?.[pointIndex] ?? 0),
      })),
    }));

    return (
      <div className="evaluation-render-stack evaluation-plot-grid">
        <ComparisonChart
          title={evaluationCurveScope === "per_class" ? `Recall vs threshold · ${selectedClassLabel}` : "Recall vs threshold"}
          subtitle="Courbe recall issue de f1_stats."
          xLabel="Confidence threshold"
          yLabel="Recall"
          metricKey="avg_recall_low_snr"
          series={thresholdRecallSeries}
          formatXValue={formatThreshold}
          formatYValue={formatPercent}
        />
        <ComparisonChart
          title={evaluationCurveScope === "per_class" ? `Precision vs threshold · ${selectedClassLabel}` : "Precision vs threshold"}
          subtitle="Courbe precision issue de f1_stats."
          xLabel="Confidence threshold"
          yLabel="Precision"
          metricKey="precision"
          series={thresholdPrecisionSeries}
          formatXValue={formatThreshold}
          formatYValue={formatPercent}
        />
        <ComparisonChart
          title={evaluationCurveScope === "per_class" ? `F1 vs threshold · ${selectedClassLabel}` : "F1 vs threshold"}
          subtitle="Courbe F1 issue de f1_stats."
          xLabel="Confidence threshold"
          yLabel="F1"
          metricKey="f1"
          series={thresholdF1Series}
          formatXValue={formatThreshold}
          formatYValue={formatPercent}
        />
        <ComparisonChart
          title={evaluationCurveScope === "per_class" ? `Precision vs recall · ${selectedClassLabel}` : "Precision vs recall"}
          subtitle="Trace precision / recall reconstituee a partir de f1_stats."
          xLabel="Recall"
          yLabel="Precision"
          metricKey="precision"
          series={precisionRecallSeries}
          formatXValue={formatPercent}
          formatYValue={formatPercent}
        />
      </div>
    );
  }

  if (visualizationId === "map_vs_epochs") {
    return (
      <div className="evaluation-render-stack evaluation-plot-grid">
        <ComparisonChart
          title="mAP50 vs epochs"
          subtitle="Evolution du mAP50 par epoch."
          xLabel="Epoch"
          yLabel="mAP50"
          metricKey="map50"
          series={buildEpochSeries(loadedRuns, "map50")}
        />
        <ComparisonChart
          title="mAP50:95 vs epochs"
          subtitle="Evolution du mAP50:95 par epoch."
          xLabel="Epoch"
          yLabel="mAP50:95"
          metricKey="map50_95"
          series={buildEpochSeries(loadedRuns, "map50_95")}
        />
      </div>
    );
  }

  if (visualizationId === "loss_vs_epochs") {
    return (
      <div className="evaluation-render-stack evaluation-plot-grid">
        <ComparisonChart
          title="Training loss vs epochs"
          subtitle="Evolution de la loss d'entrainement."
          xLabel="Epoch"
          yLabel="Training loss"
          metricKey="train_loss"
          series={buildEpochSeries(loadedRuns, "train_loss")}
        />
        <ComparisonChart
          title="Validation loss vs epochs"
          subtitle="Evolution de la loss de validation."
          xLabel="Epoch"
          yLabel="Validation loss"
          metricKey="val_loss"
          series={buildEpochSeries(loadedRuns, "val_loss")}
        />
      </div>
    );
  }

  return (
    <div className="evaluation-render-stack evaluation-plot-grid">
      <ComparisonChart
        title="Recall low SNR vs epochs"
        subtitle="Evolution du recall faible SNR."
        xLabel="Epoch"
        yLabel="Recall low SNR"
        metricKey="avg_recall_low_snr"
        series={buildEpochSeries(loadedRuns, "avg_recall_low_snr")}
      />
      <ComparisonChart
        title="Recall medium SNR vs epochs"
        subtitle="Evolution du recall moyen SNR."
        xLabel="Epoch"
        yLabel="Recall medium SNR"
        metricKey="avg_recall_medium_snr"
        series={buildEpochSeries(loadedRuns, "avg_recall_medium_snr")}
      />
      <ComparisonChart
        title="Recall high SNR vs epochs"
        subtitle="Evolution du recall fort SNR."
        xLabel="Epoch"
        yLabel="Recall high SNR"
        metricKey="avg_recall_high_snr"
        series={buildEpochSeries(loadedRuns, "avg_recall_high_snr")}
      />
    </div>
  );
}

function ScatterComparisonChart({
  title,
  subtitle,
  xLabel,
  yLabel,
  xMetricKey,
  yMetricKey,
  selectedMetricKeys,
  runSummaries,
  points
}) {
  const [hiddenRunLabels, setHiddenRunLabels] = useState({});
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [inspectedRunLabel, setInspectedRunLabel] = useState(null);
  const runSignature = (runSummaries ?? []).map((run) => run.label).join("|");

  useEffect(() => {
    setHiddenRunLabels({});
    setHoveredPoint(null);
    setInspectedRunLabel(null);
  }, [runSignature]);

  if (!points?.length) {
    return <p className="dashboard-empty">Pas assez de donnees pour afficher {title.toLowerCase()}.</p>;
  }

  const visibleRunSummaries = (runSummaries ?? []).filter((run) => !hiddenRunLabels[run.label]);
  const visiblePoints = points.filter((point) => !hiddenRunLabels[point.runLabel]);
  if (!visiblePoints.length) {
    return <p className="dashboard-empty">Au moins un run doit rester visible pour afficher {title.toLowerCase()}.</p>;
  }

  const width = 960;
  const height = 360;
  const padding = 42;
  const minX = Math.min(...visiblePoints.map((point) => point.x));
  const maxX = Math.max(...visiblePoints.map((point) => point.x));
  const minY = Math.min(...visiblePoints.map((point) => point.y));
  const maxY = Math.max(...visiblePoints.map((point) => point.y));
  const safeMaxX = minX === maxX ? maxX + 1 : maxX;
  const safeMaxY = minY === maxY ? maxY + 1 : maxY;
  const xTicks = [0, 1, 2, 3].map((index) => minX + ((safeMaxX - minX) / 3) * index);
  const yTicks = [0, 1, 2, 3].map((index) => minY + ((safeMaxY - minY) / 3) * (3 - index));
  const activeRunLabel = hoveredPoint?.runLabel ?? inspectedRunLabel;
  const selectedMetricOptions = EVALUATION_SCATTER_METRIC_OPTIONS.filter((item) => selectedMetricKeys.includes(item.id));

  function pointToCoords(point) {
    const x = padding + ((point.x - minX) / Math.max(1e-9, safeMaxX - minX)) * (width - padding * 2);
    const y = height - padding - ((point.y - minY) / Math.max(1e-9, safeMaxY - minY)) * (height - padding * 2);
    return { x, y };
  }

  function renderPoint(point, coords) {
    if (point.marker === "square") {
      return (
        <>
          <rect x={coords.x - 8} y={coords.y - 8} width="16" height="16" rx="3" fill={point.color} opacity="0.14" />
          <rect x={coords.x - 5.5} y={coords.y - 5.5} width="11" height="11" rx="2" fill={point.color} opacity="0.94" />
        </>
      );
    }
    return (
      <>
        <circle cx={coords.x} cy={coords.y} r="12" fill={point.color} opacity="0.14" />
        <circle cx={coords.x} cy={coords.y} r="7" fill={point.color} opacity="0.94" />
      </>
    );
  }

  function toggleRunVisibility(runLabel) {
    setHiddenRunLabels((current) => {
      const nextHidden = !current[runLabel];
      const visibleCount = (runSummaries ?? []).filter((run) => !current[run.label]).length;
      if (nextHidden && visibleCount === 1) {
        return current;
      }
      return {
        ...current,
        [runLabel]: nextHidden,
      };
    });
  }

  return (
    <section className="evaluation-plot-card">
      <div className="evaluation-chart-meta">
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
        <span>
          {xLabel} {formatLargeNumber(minX)} → {formatLargeNumber(safeMaxX)} · {yLabel} {formatMetricValue(yMetricKey, minY)} → {formatMetricValue(yMetricKey, safeMaxY)}
        </span>
      </div>
      <div className="evaluation-axis-guide">
        <span>Axe X: {xLabel}</span>
        <span>Axe Y: {yLabel}</span>
        <span>Chaque point = best checkpoint d&apos;un run</span>
        {selectedMetricOptions.map((metricOption) => (
          <span key={metricOption.id}>
            {metricOption.marker === "circle" ? "Rond" : "Carre"} = {metricOption.label}
          </span>
        ))}
        <span>{visibleRunSummaries.length} / {(runSummaries ?? []).length} runs visibles</span>
      </div>
      <div className="evaluation-chart-frame">
        <svg
          className="evaluation-chart"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={title}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {[0, 1, 2, 3].map((index) => {
            const y = padding + ((height - padding * 2) / 3) * index;
            return <line key={`h-${index}`} x1={padding} y1={y} x2={width - padding} y2={y} className="evaluation-chart-grid" />;
          })}
          {[0, 1, 2, 3].map((index) => {
            const x = padding + ((width - padding * 2) / 3) * index;
            return <line key={`v-${index}`} x1={x} y1={padding} x2={x} y2={height - padding} className="evaluation-chart-grid" />;
          })}
          {hoveredPoint ? (
            <>
              <line
                x1={padding}
                y1={hoveredPoint.coords.y}
                x2={width - padding}
                y2={hoveredPoint.coords.y}
                className="evaluation-chart-guide-line"
              />
              <line
                x1={hoveredPoint.coords.x}
                y1={padding}
                x2={hoveredPoint.coords.x}
                y2={height - padding}
                className="evaluation-chart-guide-line"
              />
            </>
          ) : null}
          {visiblePoints.map((point) => {
            const coords = pointToCoords(point);
            const isActive = activeRunLabel ? activeRunLabel === point.runLabel : true;
            const pointOpacity = isActive ? 1 : 0.26;
            const shouldShowLabel = hoveredPoint
              ? hoveredPoint.runLabel === point.runLabel && hoveredPoint.metricKey === point.metricKey
              : activeRunLabel === point.runLabel;
            return (
              <g
                key={`${point.runLabel}-${point.metricKey}`}
                opacity={pointOpacity}
                onMouseEnter={() => setHoveredPoint({ ...point, coords })}
              >
                {renderPoint(point, coords)}
                <circle
                  cx={coords.x}
                  cy={coords.y}
                  r="16"
                  fill="transparent"
                  className="evaluation-chart-point-hitbox"
                />
                {shouldShowLabel ? (
                  <text x={coords.x + 10} y={coords.y - 10} className="evaluation-chart-point-label">
                    {point.label}
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={width / 2} y={height - 8} textAnchor="middle" className="evaluation-chart-axis-title">
            {xLabel}
          </text>
          <text
            x={14}
            y={height / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${height / 2})`}
            className="evaluation-chart-axis-title"
          >
            {yLabel}
          </text>
          {xTicks.map((tick, index) => {
            const x = padding + ((tick - minX) / Math.max(1e-9, safeMaxX - minX)) * (width - padding * 2);
            return (
              <text key={`x-tick-${index}`} x={x} y={height - 20} textAnchor="middle" className="evaluation-chart-tick">
                {formatLargeNumber(tick)}
              </text>
            );
          })}
          {yTicks.map((tick, index) => {
            const y = padding + ((height - padding * 2) / 3) * index + 4;
            return (
              <text key={`y-tick-${index}`} x={padding - 10} y={y} textAnchor="end" className="evaluation-chart-tick">
                {formatMetricValue(yMetricKey, tick)}
              </text>
            );
          })}
        </svg>
        {hoveredPoint ? (
          <div
            className="evaluation-chart-tooltip"
            style={{
              left: `${(hoveredPoint.coords.x / width) * 100}%`,
              top: `${(hoveredPoint.coords.y / height) * 100}%`,
            }}
          >
            <strong>{hoveredPoint.runLabel}</strong>
            <span>{hoveredPoint.metricLabel}</span>
            <span>{xLabel}: {formatLargeNumber(hoveredPoint.x)}</span>
            <span>{hoveredPoint.metricLabel}: {formatMetricValue(hoveredPoint.metricKey, hoveredPoint.y)}</span>
            {hoveredPoint.epoch !== null && hoveredPoint.epoch !== undefined ? <span>Epoch: {formatCount(hoveredPoint.epoch)}</span> : null}
          </div>
        ) : null}
      </div>
      <div className="evaluation-chart-legend">
        {(runSummaries ?? []).map((run) => (
          <button
            key={run.label}
            type="button"
            className={`evaluation-chart-legend-item ${hiddenRunLabels[run.label] ? "evaluation-chart-legend-item-muted" : ""} ${activeRunLabel === run.label ? "evaluation-chart-legend-item-active" : ""}`}
            onClick={() => toggleRunVisibility(run.label)}
            onMouseEnter={() => setInspectedRunLabel(run.label)}
            onMouseLeave={() => setInspectedRunLabel(null)}
          >
            <span className="evaluation-chart-legend-swatch" style={{ backgroundColor: run.color }} />
            <div>
              <strong>{run.label}</strong>
              <small>
                {xLabel} {formatLargeNumber(run.x)}
              </small>
              <div className="evaluation-chart-legend-metrics">
                {selectedMetricOptions.map((metricOption) => (
                  <span key={`${run.label}-${metricOption.id}`}>
                    {metricOption.label} {formatMetricValue(metricOption.id, run.metrics?.[metricOption.id])}
                  </span>
                ))}
              </div>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}

function buildEpochSeries(loadedRuns, metricKey) {
  const palette = ["#c8742f", "#3f566b", "#8b4c18", "#5e7e95"];
  return loadedRuns
    .map((run, index) => ({
      label: run.label,
      color: palette[index % palette.length],
      points: (run.detail?.epoch_rows ?? [])
        .filter((row) => row?.[metricKey] !== null && row?.[metricKey] !== undefined)
        .map((row) => ({ x: Number(row.epoch), y: Number(row[metricKey]) })),
    }))
    .filter((item) => item.points.length >= 2);
}

function ComparisonChart({
  title,
  subtitle,
  xLabel,
  yLabel,
  metricKey,
  series,
  formatXValue = formatCount,
  formatYValue = (value) => formatMetricValue(metricKey, value),
}) {
  const [hiddenSeriesLabels, setHiddenSeriesLabels] = useState({});
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [inspectedSeriesLabel, setInspectedSeriesLabel] = useState(null);
  const seriesSignature = (series ?? []).map((item) => item.label).join("|");

  useEffect(() => {
    setHiddenSeriesLabels({});
    setHoveredPoint(null);
    setInspectedSeriesLabel(null);
  }, [seriesSignature]);

  if (!series?.length) {
    return <p className="dashboard-empty">Pas assez de donnees pour afficher {title.toLowerCase()}.</p>;
  }

  const visibleSeries = series.filter((item) => !hiddenSeriesLabels[item.label]);
  if (!visibleSeries.length) {
    return <p className="dashboard-empty">Au moins une serie doit rester visible pour afficher {title.toLowerCase()}.</p>;
  }

  const width = 960;
  const height = 340;
  const padding = 34;
  const allPoints = visibleSeries.flatMap((item) => item.points);
  const minX = Math.min(...allPoints.map((point) => point.x));
  const maxX = Math.max(...allPoints.map((point) => point.x));
  const minY = Math.min(...allPoints.map((point) => point.y));
  const maxY = Math.max(...allPoints.map((point) => point.y));
  const safeMaxX = minX === maxX ? maxX + 1 : maxX;
  const safeMaxY = minY === maxY ? maxY + 1 : maxY;
  const xTicks = [0, 1, 2, 3].map((index) => minX + ((safeMaxX - minX) / 3) * index);
  const yTicks = [0, 1, 2, 3].map((index) => minY + ((safeMaxY - minY) / 3) * (3 - index));
  const activeSeriesLabel = hoveredPoint?.seriesLabel ?? inspectedSeriesLabel;

  function pointToCoords(point) {
    const x = padding + ((point.x - minX) / Math.max(1e-9, safeMaxX - minX)) * (width - padding * 2);
    const y = height - padding - ((point.y - minY) / Math.max(1e-9, safeMaxY - minY)) * (height - padding * 2);
    return { x, y };
  }

  function toggleSeriesVisibility(label) {
    setHiddenSeriesLabels((current) => {
      const nextHidden = !current[label];
      const visibleCount = series.filter((item) => !current[item.label]).length;
      if (nextHidden && visibleCount === 1) {
        return current;
      }
      return {
        ...current,
        [label]: nextHidden,
      };
    });
  }

  return (
    <section className="evaluation-plot-card">
      <div className="evaluation-chart-meta">
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
        <span>
          {xLabel} {formatXValue(minX)} → {formatXValue(safeMaxX)} · {yLabel} {formatYValue(minY)} → {formatYValue(safeMaxY)}
        </span>
      </div>
      <div className="evaluation-axis-guide">
        <span>Survole un point pour lire la valeur exacte</span>
        <span>Clique sur un run dans la legende pour le masquer</span>
        <span>{visibleSeries.length} / {series.length} series visibles</span>
      </div>
      <div className="evaluation-chart-frame">
        <svg
          className="evaluation-chart"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={title}
          onMouseLeave={() => setHoveredPoint(null)}
        >
          {[0, 1, 2, 3].map((index) => {
            const y = padding + ((height - padding * 2) / 3) * index;
            return <line key={`h-${index}`} x1={padding} y1={y} x2={width - padding} y2={y} className="evaluation-chart-grid" />;
          })}
          {[0, 1, 2, 3].map((index) => {
            const x = padding + ((width - padding * 2) / 3) * index;
            return <line key={`v-${index}`} x1={x} y1={padding} x2={x} y2={height - padding} className="evaluation-chart-grid" />;
          })}
          {hoveredPoint ? (
            <>
              <line
                x1={padding}
                y1={hoveredPoint.coords.y}
                x2={width - padding}
                y2={hoveredPoint.coords.y}
                className="evaluation-chart-guide-line"
              />
              <line
                x1={hoveredPoint.coords.x}
                y1={padding}
                x2={hoveredPoint.coords.x}
                y2={height - padding}
                className="evaluation-chart-guide-line"
              />
            </>
          ) : null}
          {visibleSeries.map((item) => {
            const isActive = activeSeriesLabel ? activeSeriesLabel === item.label : true;
            return (
              <g key={item.label} opacity={isActive ? 1 : 0.22}>
                <polyline
                  points={item.points.map((point) => {
                    const coords = pointToCoords(point);
                    return `${coords.x},${coords.y}`;
                  }).join(" ")}
                  fill="none"
                  stroke={item.color}
                  strokeWidth={isActive ? "3.5" : "2.4"}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {item.points.map((point, pointIndex) => {
                  const coords = pointToCoords(point);
                  const isHovered =
                    hoveredPoint?.seriesLabel === item.label &&
                    hoveredPoint?.x === point.x &&
                    hoveredPoint?.y === point.y;
                  return (
                    <g
                      key={`${item.label}-${pointIndex}`}
                      onMouseEnter={() => setHoveredPoint({ ...point, coords, color: item.color, seriesLabel: item.label })}
                    >
                      <circle
                        cx={coords.x}
                        cy={coords.y}
                        r={isHovered ? "5.5" : "4"}
                        fill={item.color}
                        className="evaluation-chart-point-node"
                      />
                      <circle
                        cx={coords.x}
                        cy={coords.y}
                        r="12"
                        fill="transparent"
                        className="evaluation-chart-point-hitbox"
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text x={width / 2} y={height - 8} textAnchor="middle" className="evaluation-chart-axis-title">
            {xLabel}
          </text>
          <text
            x={14}
            y={height / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${height / 2})`}
            className="evaluation-chart-axis-title"
          >
            {yLabel}
          </text>
          {xTicks.map((tick, index) => {
            const x = padding + ((tick - minX) / Math.max(1e-9, safeMaxX - minX)) * (width - padding * 2);
            return (
              <text key={`x-tick-${index}`} x={x} y={height - 20} textAnchor="middle" className="evaluation-chart-tick">
                {formatXValue(tick)}
              </text>
            );
          })}
          {yTicks.map((tick, index) => {
            const y = padding + ((height - padding * 2) / 3) * index + 4;
            return (
              <text key={`y-tick-${index}`} x={padding - 10} y={y} textAnchor="end" className="evaluation-chart-tick">
                {formatYValue(tick)}
              </text>
            );
          })}
        </svg>
        {hoveredPoint ? (
          <div
            className="evaluation-chart-tooltip"
            style={{
              left: `${(hoveredPoint.coords.x / width) * 100}%`,
              top: `${(hoveredPoint.coords.y / height) * 100}%`,
            }}
          >
            <strong>{hoveredPoint.seriesLabel}</strong>
            <span>{xLabel}: {formatXValue(hoveredPoint.x)}</span>
            <span>{yLabel}: {formatYValue(hoveredPoint.y)}</span>
          </div>
        ) : null}
      </div>
      <div className="evaluation-chart-legend">
        {series.map((item) => (
          <button
            key={item.label}
            type="button"
            className={`evaluation-chart-legend-item ${hiddenSeriesLabels[item.label] ? "evaluation-chart-legend-item-muted" : ""} ${activeSeriesLabel === item.label ? "evaluation-chart-legend-item-active" : ""}`}
            onClick={() => toggleSeriesVisibility(item.label)}
            onMouseEnter={() => setInspectedSeriesLabel(item.label)}
            onMouseLeave={() => setInspectedSeriesLabel(null)}
          >
            <span className="evaluation-chart-legend-swatch" style={{ backgroundColor: item.color }} />
            <div>
              <strong>{item.label}</strong>
              <small>{formatCount(item.points.length)} points</small>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}

function normalizeConfusionMatrix(matrix, normalization) {
  const numericMatrix = matrix.map((row) => row.map((value) => Number(value ?? 0)));
  if (normalization === "row") {
    return numericMatrix.map((row) => {
      const rowSum = row.reduce((sum, value) => sum + value, 0);
      if (rowSum <= 0) {
        return row.map(() => 0);
      }
      return row.map((value) => value / rowSum);
    });
  }
  if (normalization === "column") {
    const columnSums = numericMatrix[0].map((_, colIndex) => numericMatrix.reduce((sum, row) => sum + row[colIndex], 0));
    return numericMatrix.map((row) => row.map((value, colIndex) => {
      const columnSum = columnSums[colIndex];
      return columnSum > 0 ? value / columnSum : 0;
    }));
  }
  return numericMatrix;
}

function formatConfusionValue(value, normalization) {
  if (normalization === "none") {
    return formatCount(value);
  }
  return formatPercent(value);
}

function ConfusionMatrixView({ title, mode, normalization, primaryItem, referenceItem, snrBand }) {
  if (!primaryItem) {
    return <p className="dashboard-empty">Pas assez de donnees pour afficher {title.toLowerCase()}.</p>;
  }

  return (
    <div className="evaluation-render-stack">
      <div className="evaluation-chart-meta">
        <div>
          <strong>{title}</strong>
          <span>
            {mode === "difference"
              ? "Une seule heatmap affiche la difference A - B sur l'epoch du best checkpoint de chaque run."
              : "Une seule heatmap affiche le run selectionne. Survole une case pour lire la valeur exacte."}
          </span>
        </div>
      </div>
      <ConfusionMatrixCard
        key={`${primaryItem.label}-${referenceItem?.label ?? "single"}-${snrBand}-${normalization}`}
        mode={mode}
        normalization={normalization}
        label={primaryItem.label}
        compareLabel={referenceItem?.label ?? null}
        epoch={primaryItem.epoch}
        compareEpoch={referenceItem?.epoch ?? null}
        matrix={primaryItem.matrices?.[snrBand]}
        compareMatrix={referenceItem?.matrices?.[snrBand] ?? null}
        classLabels={primaryItem.classLabels}
      />
    </div>
  );
}

function ConfusionMatrixCard({ mode, normalization, label, compareLabel, epoch, compareEpoch, matrix, compareMatrix, classLabels }) {
  const [hoveredCell, setHoveredCell] = useState(null);

  if (!Array.isArray(matrix) || !matrix.length) {
    return <p className="dashboard-empty">Aucune matrice disponible pour {label}.</p>;
  }
  if (mode === "difference" && (!Array.isArray(compareMatrix) || compareMatrix.length !== matrix.length)) {
    return <p className="dashboard-empty">Impossible de calculer la difference de matrices pour {label}.</p>;
  }

  const labels = Array.isArray(classLabels) && classLabels.length === matrix.length
    ? classLabels
    : matrix.map((_, index) => (index === matrix.length - 1 ? "bg" : `c${index}`));

  const normalizedBaseMatrix = normalizeConfusionMatrix(matrix, normalization);
  const normalizedCompareMatrix = compareMatrix ? normalizeConfusionMatrix(compareMatrix, normalization) : null;
  const displayMatrix = mode === "difference"
    ? normalizedBaseMatrix.map((row, rowIndex) => row.map((value, colIndex) => Number(value ?? 0) - Number(normalizedCompareMatrix?.[rowIndex]?.[colIndex] ?? 0)))
    : normalizedBaseMatrix;

  const flatValues = displayMatrix.flatMap((row) => row);
  const maxValue = mode === "difference"
    ? Math.max(...flatValues.map((value) => Math.abs(value)), 1)
    : Math.max(...flatValues, 1);

  function cellBackground(value) {
    const numericValue = Number(value ?? 0);
    if (mode === "difference") {
      const ratio = Math.min(1, Math.abs(numericValue) / maxValue);
      const alpha = 0.08 + ratio * 0.84;
      if (numericValue > 0) {
        return `rgba(200, 116, 47, ${alpha})`;
      }
      if (numericValue < 0) {
        return `rgba(63, 86, 107, ${alpha})`;
      }
      return "rgba(241, 245, 249, 0.9)";
    }
    const ratio = Math.max(0, numericValue / maxValue);
    const alpha = 0.08 + ratio * 0.84;
    return `rgba(200, 116, 47, ${alpha})`;
  }

  return (
    <section className="evaluation-plot-card">
      <div className="evaluation-chart-meta">
        <div>
          <strong>{mode === "difference" ? `${label} - ${compareLabel}` : label}</strong>
          <span>
            {mode === "difference"
              ? `Epochs ${formatCount(epoch)} - ${formatCount(compareEpoch)} · lignes = verite terrain, colonnes = predictions`
              : `Epoch ${formatCount(epoch)} · lignes = verite terrain, colonnes = predictions`}
          </span>
        </div>
        <span>{mode === "difference" ? `Amplitude max ${formatConfusionValue(maxValue, normalization)}` : `Max ${formatConfusionValue(maxValue, normalization)}`}</span>
      </div>
      <div className="evaluation-axis-guide">
        <span>{mode === "difference" ? "Orange: A > B" : "Plus c'est fonce, plus le compte est eleve"}</span>
        {mode === "difference" ? <span>Bleu: A &lt; B</span> : null}
        <span>
          {normalization === "row"
            ? "Chaque ligne somme a 100%"
            : normalization === "column"
              ? "Chaque colonne somme a 100%"
              : "Matrice brute"}
        </span>
        <span>{mode === "difference" ? "Valeur affichee = A - B" : normalization === "none" ? "Valeur affichee = nombre d'exemples" : "Valeur affichee = pourcentage"}</span>
      </div>
      <div className="evaluation-confusion-frame">
        <div
          className="evaluation-confusion-scroll"
          onMouseLeave={() => setHoveredCell(null)}
        >
          <div
            className="evaluation-confusion-matrix"
            style={{ gridTemplateColumns: `110px repeat(${displayMatrix.length}, minmax(40px, 1fr))` }}
          >
            <div className="evaluation-confusion-corner">GT \ Pred</div>
            {labels.map((cellLabel) => (
              <div key={`col-${cellLabel}`} className="evaluation-confusion-axis-cell evaluation-confusion-axis-top" title={cellLabel}>
                {cellLabel}
              </div>
            ))}
            {displayMatrix.map((row, rowIndex) => (
              <Fragment key={`row-${labels[rowIndex]}`}>
                <div className="evaluation-confusion-axis-cell evaluation-confusion-axis-left" title={labels[rowIndex]}>
                  {labels[rowIndex]}
                </div>
                {row.map((value, colIndex) => {
                  const numericValue = Number(value ?? 0);
                  const isActive = hoveredCell?.rowIndex === rowIndex && hoveredCell?.colIndex === colIndex;
                  const baseValue = Number(matrix?.[rowIndex]?.[colIndex] ?? 0);
                  const referenceValue = Number(compareMatrix?.[rowIndex]?.[colIndex] ?? 0);
                  const displayedBaseValue = Number(normalizedBaseMatrix?.[rowIndex]?.[colIndex] ?? 0);
                  const displayedReferenceValue = Number(normalizedCompareMatrix?.[rowIndex]?.[colIndex] ?? 0);
                  return (
                    <button
                      key={`${rowIndex}-${colIndex}`}
                      type="button"
                      className={`evaluation-confusion-cell ${isActive ? "evaluation-confusion-cell-active" : ""}`}
                      style={{ background: cellBackground(numericValue) }}
                      onMouseEnter={() => setHoveredCell({
                        rowIndex,
                        colIndex,
                        rowLabel: labels[rowIndex],
                        colLabel: labels[colIndex],
                        value: numericValue,
                        baseValue,
                        referenceValue,
                        displayedBaseValue,
                        displayedReferenceValue,
                      })}
                    >
                      {mode === "difference" && numericValue > 0 ? "+" : ""}
                      {formatConfusionValue(numericValue, normalization)}
                    </button>
                  );
                })}
              </Fragment>
            ))}
          </div>
        </div>
        {hoveredCell ? (
          <div className="evaluation-confusion-tooltip">
            <strong>{mode === "difference" ? `${label} - ${compareLabel}` : label}</strong>
            <span>GT: {hoveredCell.rowLabel}</span>
            <span>Prediction: {hoveredCell.colLabel}</span>
            {mode === "difference" ? (
              <>
                <span>A brut: {formatCount(hoveredCell.baseValue)}</span>
                <span>B brut: {formatCount(hoveredCell.referenceValue)}</span>
                <span>A affiche: {formatConfusionValue(hoveredCell.displayedBaseValue, normalization)}</span>
                <span>B affiche: {formatConfusionValue(hoveredCell.displayedReferenceValue, normalization)}</span>
                <span>Delta: {hoveredCell.value > 0 ? "+" : ""}{formatConfusionValue(hoveredCell.value, normalization)}</span>
              </>
            ) : (
              <>
                <span>Brut: {formatCount(hoveredCell.baseValue)}</span>
                <span>Affiche: {formatConfusionValue(hoveredCell.value, normalization)}</span>
              </>
            )}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function ArtifactsPage({ apiFetch }) {
  const [checkpointPath, setCheckpointPath] = useState("");
  const [datasetPath, setDatasetPath] = useState(DEFAULT_DATASET_PATH);
  const [datasetSplit, setDatasetSplit] = useState("val");
  const [confidenceThreshold, setConfidenceThreshold] = useState("0.10");
  const [appliedConfig, setAppliedConfig] = useState({
    checkpointPath: "",
    datasetPath: DEFAULT_DATASET_PATH,
    datasetSplit: "val",
    confidenceThreshold: "0.10",
  });
  const [examplesState, setExamplesState] = useState({ status: "idle", data: null, error: null });
  const [previewState, setPreviewState] = useState({ status: "idle", data: null, error: null });
  const [currentIndex, setCurrentIndex] = useState(0);
  const [cfgIndex, setCfgIndex] = useState(0);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showTP, setShowTP] = useState(true);
  const [showFP, setShowFP] = useState(true);
  const [showFN, setShowFN] = useState(true);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
  const [hoveredBoxIndex, setHoveredBoxIndex] = useState(null);

  const activeSamples = examplesState.data?.samples ?? [];
  const preview = previewState.data;
  const currentSample = activeSamples[currentIndex] ?? null;
  const predictionAnalysis = preview?.analysis?.summary
    ? {
        precision: Number(preview.analysis.summary.precision ?? 0),
        recall: Number(preview.analysis.summary.recall ?? 0),
        truePositives: Number(preview.analysis.summary.tp_count ?? 0),
        falsePositives: Number(preview.analysis.summary.fp_count ?? 0),
        falseNegatives: Number(preview.analysis.summary.fn_count ?? 0),
        averageConfidence: Number(preview.analysis.summary.average_confidence ?? 0),
        averageIoU: Number(preview.analysis.summary.average_iou ?? 0),
      }
    : buildPredictionAnalysis(preview);
  const hasPendingConfigChanges =
    checkpointPath !== appliedConfig.checkpointPath ||
    datasetPath !== appliedConfig.datasetPath ||
    datasetSplit !== appliedConfig.datasetSplit ||
    confidenceThreshold !== appliedConfig.confidenceThreshold;

  useEffect(() => {
    const nextPath = String(appliedConfig.datasetPath ?? "").trim();
    if (!nextPath) {
      setExamplesState({ status: "idle", data: null, error: null });
      setPreviewState({ status: "idle", data: null, error: null });
      return;
    }

    void (async () => {
      setExamplesState({ status: "loading", data: null, error: null });
      try {
        const response = await apiFetch(
          `/dataset/examples?path=${encodeURIComponent(nextPath)}&split=${encodeURIComponent(appliedConfig.datasetSplit)}&offset=0&limit=50`
        );
        const data = await response.json();
        if (!response.ok) {
          throw new Error(extractApiError(data, response.status).message);
        }
        setExamplesState({ status: "ready", data, error: null });
        setCurrentIndex(0);
        setCfgIndex(0);
        setSelectedBoxIndex(null);
        setHoveredBoxIndex(null);
      } catch (error) {
        setExamplesState({
          status: "error",
          data: null,
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
        setPreviewState({ status: "idle", data: null, error: null });
      }
    })();
  }, [apiFetch, appliedConfig.datasetPath, appliedConfig.datasetSplit]);

  useEffect(() => {
    const nextPath = String(appliedConfig.datasetPath ?? "").trim();
    if (!nextPath || !currentSample) {
      setPreviewState({ status: "idle", data: null, error: null });
      return;
    }

    void (async () => {
      setPreviewState((current) => ({ status: "loading", data: current.data, error: null }));
      try {
        const confValue = normalizeThresholdInput(appliedConfig.confidenceThreshold, 0.1);
        const previewPath = appliedConfig.checkpointPath.trim()
          ? `/artifacts/preview?path=${encodeURIComponent(nextPath)}&split=${encodeURIComponent(appliedConfig.datasetSplit)}&sample_id=${encodeURIComponent(currentSample.sample_id)}&cfg_index=${cfgIndex}&checkpoint_path=${encodeURIComponent(appliedConfig.checkpointPath.trim())}&conf_thres=${encodeURIComponent(confValue)}`
          : `/dataset/example?path=${encodeURIComponent(nextPath)}&split=${encodeURIComponent(appliedConfig.datasetSplit)}&sample_id=${encodeURIComponent(currentSample.sample_id)}&cfg_index=${cfgIndex}`;
        const response = await apiFetch(previewPath);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(extractApiError(data, response.status).message);
        }
        setPreviewState({ status: "ready", data, error: null });
        setSelectedBoxIndex(null);
        setHoveredBoxIndex(null);
      } catch (error) {
        setPreviewState({
          status: "error",
          data: null,
          error: error instanceof Error ? error.message : "Erreur inconnue."
        });
      }
    })();
  }, [apiFetch, appliedConfig.checkpointPath, appliedConfig.confidenceThreshold, appliedConfig.datasetPath, appliedConfig.datasetSplit, cfgIndex, currentSample]);

  function applyArtifactsConfig() {
    setAppliedConfig({
      checkpointPath,
      datasetPath,
      datasetSplit,
      confidenceThreshold,
    });
    setCurrentIndex(0);
    setCfgIndex(0);
    setSelectedBoxIndex(null);
    setHoveredBoxIndex(null);
    setPreviewState({ status: "idle", data: null, error: null });
  }

  return (
    <div className="page-stack">
      <header className="hero detector-hero">
        <div className="hero-copy">
          <span className="eyebrow">Artefacts</span>
          <h1>Visualiser les sorties d&apos;un modele a partir d&apos;un checkpoint</h1>
          <p>
            Cet ecran ne garde que le flux utile: choisir un <code>best.pt</code>, choisir
            la base de donnees sur laquelle appliquer le modele, puis lire les sorties.
          </p>
        </div>
      </header>

      <section className="dashboard-grid">
        <section className="panel panel-span-12 artifacts-config-panel">
          <div className="section-title">
            <span>Configuration</span>
            <h2>Checkpoint et base de donnees</h2>
            <p>Renseigne le chemin du modele entraine et la base de donnees a utiliser pour la visualisation.</p>
          </div>
          <div className="artifacts-form-grid">
            <PathField
              className="field-full-span"
              label="Chemin vers best.pt"
              value={checkpointPath}
              onChange={setCheckpointPath}
              apiFetch={apiFetch}
              pickerKind="file"
              pickerTitle="Choisir un checkpoint"
              placeholder="/Users/.../runs/.../best.pt"
            />
            <PathField
              className="field-full-span"
              label="Base de donnees"
              value={datasetPath}
              onChange={setDatasetPath}
              apiFetch={apiFetch}
              pickerKind="directory"
              pickerTitle="Choisir un dataset"
              placeholder="/Users/.../rf_dataset_v2"
            />
            <label className="field artifacts-split-field">
              <span>Split</span>
              <select value={datasetSplit} onChange={(event) => setDatasetSplit(event.target.value)}>
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
              </select>
            </label>
            <label className="field artifacts-threshold-field">
              <span>Seuil de confiance</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={confidenceThreshold}
                onChange={(event) => setConfidenceThreshold(event.target.value)}
              />
            </label>
          </div>
          <div className="hero-actions artifacts-config-actions">
            <button
              type="button"
              className="primary-button"
              onClick={applyArtifactsConfig}
              disabled={!hasPendingConfigChanges}
            >
              Appliquer
            </button>
          </div>
        </section>

        <section className="panel panel-span-12 artifacts-output-panel">
          <div className="section-title">
            <span>Visualisation</span>
            <h2>Sorties du modele</h2>
            <p>Pour le moment, cette zone affiche le spectre et les boites reelles du dataset selectionne.</p>
          </div>
          <div className="viewer-toolbar">
            <div className="viewer-nav">
              <button
                type="button"
                className="secondary-button"
                onClick={() => setCurrentIndex((current) => Math.max(0, current - 1))}
                disabled={currentIndex === 0}
              >
                ← Precedent
              </button>
              <strong>{activeSamples.length ? `${currentIndex + 1} / ${activeSamples.length}` : "0 / 0"}</strong>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setCurrentIndex((current) => Math.min(current + 1, Math.max(0, activeSamples.length - 1)))}
                disabled={!activeSamples.length || currentIndex >= activeSamples.length - 1}
              >
                Suivant →
              </button>
            </div>

            <div className="viewer-configs">
              {(preview?.cfg_labels ?? []).map((label, index) => (
                <button
                  key={label}
                  type="button"
                  className={`nav-tab ${cfgIndex === index ? "nav-tab-active" : ""}`}
                  onClick={() => setCfgIndex(index)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {examplesState.status === "loading" ? <p className="dashboard-empty">Chargement des exemples...</p> : null}
          {examplesState.status === "error" ? <p className="dashboard-empty">{examplesState.error}</p> : null}
          {previewState.status === "loading" ? <p className="dashboard-empty">Chargement du spectre...</p> : null}
          {previewState.status === "error" ? <p className="dashboard-empty">{previewState.error}</p> : null}

          {preview ? (
            <div className="sample-viewer artifacts-sample-viewer">
              <div className="sample-stage">
                <img src={preview.image.data_url} alt={`Spectre ${preview.sample_id}`} className="sample-image" />
                <div className={`sample-overlay ${showGroundTruth ? "" : "sample-overlay-hidden"}`}>
                  {preview.boxes.map((box, index) => {
                    const isSelected = selectedBoxIndex === index;
                    const isHovered = hoveredBoxIndex === index;
                    return (
                      <div
                        key={`${box.class_id}-${index}`}
                        className={`sample-box ${isSelected ? "sample-box-selected" : ""} ${isHovered ? "sample-box-hovered" : ""}`}
                        style={{
                          left: `${(box.xc - box.w / 2) * 100}%`,
                          top: `${(box.yc - box.h / 2) * 100}%`,
                          width: `${box.w * 100}%`,
                          height: `${box.h * 100}%`,
                          borderColor: "#22d3ee"
                        }}
                        onMouseEnter={() => setHoveredBoxIndex(index)}
                        onMouseLeave={() => setHoveredBoxIndex((current) => (current === index ? null : current))}
                        onClick={() => setSelectedBoxIndex(index)}
                      >
                        <span className="sample-box-label" style={{ background: "#0891b2" }}>
                          {box.class_name}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className={`sample-overlay ${showTP ? "" : "sample-overlay-hidden"}`}>
                  {(preview.analysis?.tp ?? []).map((box, index) => {
                    return (
                      <div
                        key={`tp-${box.class_id}-${index}`}
                        className="sample-box sample-box-prediction sample-box-tp"
                        style={{
                          left: `${(box.xc - box.w / 2) * 100}%`,
                          top: `${(box.yc - box.h / 2) * 100}%`,
                          width: `${box.w * 100}%`,
                          height: `${box.h * 100}%`,
                          borderColor: "#22c55e"
                        }}
                      >
                        <span className="sample-box-label sample-box-label-prediction" style={{ background: "#16a34a" }}>
                          TP {box.class_name} · {formatPercent(box.confidence)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className={`sample-overlay ${showFP ? "" : "sample-overlay-hidden"}`}>
                  {(preview.analysis?.fp ?? []).map((box, index) => {
                    return (
                      <div
                        key={`fp-${box.class_id}-${index}`}
                        className="sample-box sample-box-prediction sample-box-fp"
                        style={{
                          left: `${(box.xc - box.w / 2) * 100}%`,
                          top: `${(box.yc - box.h / 2) * 100}%`,
                          width: `${box.w * 100}%`,
                          height: `${box.h * 100}%`,
                          borderColor: "#ef4444"
                        }}
                      >
                        <span className="sample-box-label sample-box-label-prediction" style={{ background: "#dc2626" }}>
                          FP {box.class_name} · {formatPercent(box.confidence)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className={`sample-overlay ${showGroundTruth ? "" : "sample-overlay-hidden"}`}>
                </div>
                <div className={`sample-overlay ${showFN ? "" : "sample-overlay-hidden"}`}>
                  {(preview.analysis?.fn ?? []).map((box, index) => {
                    return (
                      <div
                        key={`fn-${box.class_id}-${index}`}
                        className="sample-box sample-box-fn"
                        style={{
                          left: `${(box.xc - box.w / 2) * 100}%`,
                          top: `${(box.yc - box.h / 2) * 100}%`,
                          width: `${box.w * 100}%`,
                          height: `${box.h * 100}%`,
                          borderColor: "#a855f7"
                        }}
                      >
                        <span className="sample-box-label" style={{ background: "#9333ea" }}>
                          FN {box.class_name}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="sample-sidebar">
                <div className="artifacts-legend">
                  <button type="button" className={`artifacts-legend-item ${showGroundTruth ? "artifacts-legend-item-active" : ""}`} onClick={() => setShowGroundTruth((current) => !current)}>
                    <span className="artifacts-legend-swatch artifacts-legend-swatch-gt" />
                    <strong>Boites reelles</strong>
                    <small>ground truth du dataset</small>
                  </button>
                  <button type="button" className={`artifacts-legend-item ${showTP ? "artifacts-legend-item-active" : ""}`} onClick={() => setShowTP((current) => !current)}>
                    <span className="artifacts-legend-swatch artifacts-legend-swatch-tp" />
                    <strong>TP</strong>
                    <small>predictions matchees</small>
                  </button>
                  <button type="button" className={`artifacts-legend-item ${showFP ? "artifacts-legend-item-active" : ""}`} onClick={() => setShowFP((current) => !current)}>
                    <span className="artifacts-legend-swatch artifacts-legend-swatch-fp" />
                    <strong>FP</strong>
                    <small>predictions non matchees</small>
                  </button>
                  <button type="button" className={`artifacts-legend-item ${showFN ? "artifacts-legend-item-active" : ""}`} onClick={() => setShowFN((current) => !current)}>
                    <span className="artifacts-legend-swatch artifacts-legend-swatch-fn" />
                    <strong>FN</strong>
                    <small>boites reelles manquees</small>
                  </button>
                </div>
                <div className="metric-line">
                  <span>Sample</span>
                  <strong>{preview.sample_id}</strong>
                </div>
                <div className="metric-line">
                  <span>Split</span>
                  <strong>{appliedConfig.datasetSplit}</strong>
                </div>
                <div className="metric-line">
                  <span>Boites</span>
                  <strong>{formatCount(preview.box_count)}</strong>
                </div>
                <div className="metric-line">
                  <span>Predictions</span>
                  <strong>{formatCount(preview.prediction_count ?? 0)}</strong>
                </div>
                <div className="metric-line">
                  <span>Resolution</span>
                  <strong>{preview.cfg_labels?.[preview.cfg_index] ?? `cfg${preview.cfg_index}`}</strong>
                </div>
                {preview.inference ? (
                  <div className="sample-detail-card">
                    <span>Inference</span>
                    <strong>{preview.inference.model_name}</strong>
                    <p>
                      conf {formatDecimal(preview.inference.conf_thres, 2)} ·
                      iou {formatDecimal(preview.inference.iou_thres, 2)} ·
                      device {preview.inference.device}
                    </p>
                  </div>
                ) : null}
                <div className="sample-detail-card">
                  <span>Analyse des predictions</span>
                  <strong>
                    precision {formatPercent(predictionAnalysis.precision)} · recall {formatPercent(predictionAnalysis.recall)}
                  </strong>
                  <p>
                    {formatCount(predictionAnalysis.truePositives)} TP ·
                    {` ${formatCount(predictionAnalysis.falsePositives)} faux positifs · `}
                    {formatCount(predictionAnalysis.falseNegatives)} faux negatifs
                  </p>
                  <p>
                    confiance moyenne {formatPercent(predictionAnalysis.averageConfidence)} ·
                    IoU moyen {formatPercent(predictionAnalysis.averageIoU)}
                  </p>
                </div>
                <div className="class-chip-list">
                  {preview.class_names.map((className, index) => (
                    <span
                      key={className}
                      className="class-chip"
                      style={{ borderColor: classColor(index), color: classColor(index) }}
                    >
                      {className}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </section>
    </div>
  );
}

function SectionToast({ toast }) {
  if (!toast) {
    return null;
  }

  return (
    <div className={`app-toast app-toast-${toast.type}`}>
      <span className="app-toast-icon">{toast.icon}</span>
      <span>{toast.message}</span>
    </div>
  );
}

function Table({ headers, rows }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${rowIndex}-${row.join("-")}`}>
              {row.map((cell, cellIndex) => (
                <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricTrendChart({ details, metricKey, metricLabel }) {
  const palette = ["#c8742f", "#3f566b", "#8b4c18", "#5e7e95"];
  const series = details
    .map((detail, index) => {
      const points = (detail.epoch_rows ?? [])
        .filter((row) => row?.[metricKey] !== null && row?.[metricKey] !== undefined)
        .map((row) => ({ x: Number(row.epoch), y: Number(row[metricKey]) }));
      return {
        label: detail.summary?.run_name ?? `Run ${index + 1}`,
        color: palette[index % palette.length],
        points,
      };
    })
    .filter((item) => item.points.length >= 2);

  if (!series.length) {
    return <p className="dashboard-empty">Pas assez de donnees pour tracer cette metrique sur les runs selectionnes.</p>;
  }

  const width = 960;
  const height = 340;
  const padding = 32;
  const allPoints = series.flatMap((item) => item.points);
  const minX = Math.min(...allPoints.map((point) => point.x));
  const maxX = Math.max(...allPoints.map((point) => point.x));
  const minY = Math.min(...allPoints.map((point) => point.y));
  const maxY = Math.max(...allPoints.map((point) => point.y));
  const safeMaxY = minY === maxY ? maxY + 1 : maxY;

  function pointToCoords(point) {
    const x = padding + ((point.x - minX) / Math.max(1, maxX - minX)) * (width - padding * 2);
    const y = height - padding - ((point.y - minY) / Math.max(1e-9, safeMaxY - minY)) * (height - padding * 2);
    return `${x},${y}`;
  }

  return (
    <div className="evaluation-chart-shell">
      <div className="evaluation-chart-meta">
        <strong>{metricLabel}</strong>
        <span>
          epochs {formatCount(minX)} → {formatCount(maxX)} · plage {formatMetricValue(metricKey, minY)} → {formatMetricValue(metricKey, safeMaxY)}
        </span>
      </div>
      <svg className="evaluation-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Courbe ${metricLabel}`}>
        {[0, 1, 2, 3].map((index) => {
          const y = padding + ((height - padding * 2) / 3) * index;
          return <line key={index} x1={padding} y1={y} x2={width - padding} y2={y} className="evaluation-chart-grid" />;
        })}
        {series.map((item) => (
          <polyline
            key={item.label}
            points={item.points.map(pointToCoords).join(" ")}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
      </svg>
      <div className="evaluation-chart-legend">
        {series.map((item) => (
          <div key={item.label} className="evaluation-chart-legend-item">
            <span className="evaluation-chart-legend-swatch" style={{ backgroundColor: item.color }} />
            <strong>{item.label}</strong>
            <small>{formatCount(item.points.length)} epochs</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function BarList({ items }) {
  const maxValue = Math.max(...items.map((item) => item.value), 1);

  return (
    <div className="bar-list">
      {items.map((item) => (
        <div key={item.label} className="bar-row">
          <div className="bar-row-head">
            <strong>{item.label}</strong>
            <span>{item.value}%</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(item.value / maxValue) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function RichBarList({ items, labelKey, valueKey, valueFormatter }) {
  if (!items?.length) {
    return <p className="dashboard-empty">Aucune donnee disponible pour cette vue.</p>;
  }

  const maxValue = Math.max(...items.map((item) => Number(item[valueKey] ?? 0)), 1);

  return (
    <div className="bar-list">
      {items.map((item) => {
        const value = Number(item[valueKey] ?? 0);
        return (
          <div key={`${item[labelKey]}-${value}`} className="bar-row">
            <div className="bar-row-head">
              <strong>{item[labelKey]}</strong>
              <span>{valueFormatter ? valueFormatter(value, item) : formatCount(value)}</span>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(value / maxValue) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [page, setPage] = useState("connect");
  const [credentialInput, setCredentialInput] = useState("");
  const [passwordInput, setPasswordInput] = useState("");
  const [activeApiBase, setActiveApiBase] = useState(() => {
    const saved = typeof window !== "undefined" ? window.localStorage.getItem("detector2026-api-base") : null;
    return normalizeApiBase(saved || DEFAULT_API_BASE);
  });
  const [connectionInput, setConnectionInput] = useState(() => {
    const saved = typeof window !== "undefined" ? window.localStorage.getItem("detector2026-api-base") : null;
    return normalizeApiBase(saved || DEFAULT_API_BASE);
  });
  const [connectionState, setConnectionState] = useState({
    status: "idle",
    message: "Aucune verification lancee.",
    latencyMs: null
  });
  const [toast, setToast] = useState(null);
  const healthWatchRef = useRef(null);
  const toastTimerRef = useRef(null);

  const isConnected = connectionState.status === "connected";
  const connectionLabel = activeApiBase || DEFAULT_API_BASE;
  const apiFetch = (path, options) => fetch(resolveApiUrl(activeApiBase, path), options);

  useEffect(() => {
    return () => {
      if (healthWatchRef.current) {
        clearInterval(healthWatchRef.current);
      }
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isConnected && page !== "connect" && page !== "overview") {
      setPage("connect");
    }
  }, [isConnected, page]);

  useEffect(() => {
    if (!isConnected) {
      if (healthWatchRef.current) {
        clearInterval(healthWatchRef.current);
        healthWatchRef.current = null;
      }
      return;
    }

    let stopped = false;
    async function checkConnection() {
      try {
        const response = await fetch(resolveApiUrl(activeApiBase, "/health"));
        const data = await response.json();
        if (!response.ok || data.status !== "ok") {
          throw new Error("health check failed");
        }
      } catch (_error) {
        if (stopped) {
          return;
        }
        if (healthWatchRef.current) {
          clearInterval(healthWatchRef.current);
          healthWatchRef.current = null;
        }
        setConnectionState({
          status: "error",
          message: "Connexion perdue avec le backend.",
          latencyMs: null
        });
        setPage("connect");
        showToast("Connexion perdue", "warning");
      }
    }

    healthWatchRef.current = setInterval(() => {
      void checkConnection();
    }, 7000);

    return () => {
      stopped = true;
      if (healthWatchRef.current) {
        clearInterval(healthWatchRef.current);
        healthWatchRef.current = null;
      }
    };
  }, [isConnected, activeApiBase]);

  function showToast(message, type = "success") {
    if (toastTimerRef.current) {
      clearTimeout(toastTimerRef.current);
    }
    const icon = type === "success" ? "OK" : type === "warning" ? "!" : "i";
    setToast({ message, type, icon });
    toastTimerRef.current = setTimeout(() => setToast(null), 2200);
  }

  async function connectBackend() {
    if (credentialInput.trim() !== "admin" || passwordInput !== "admin") {
      setConnectionState({
        status: "error",
        message: "Identifiant ou mot de passe invalide.",
        latencyMs: null
      });
      return;
    }

    const base = normalizeApiBase(connectionInput);
    if (!base) {
      setConnectionState({
        status: "error",
        message: "Adresse backend invalide.",
        latencyMs: null
      });
      return;
    }

    setConnectionState({
      status: "checking",
      message: "Verification de /health en cours...",
      latencyMs: null
    });

    try {
      const startedAt = performance.now();
      const response = await fetch(resolveApiUrl(base, "/health"));
      const data = await response.json();
      const latencyMs = Math.round(performance.now() - startedAt);
      if (!response.ok || data.status !== "ok") {
        throw new Error(data.detail ?? "Reponse /health invalide.");
      }
      window.localStorage.setItem("detector2026-api-base", base);
      setActiveApiBase(base);
      setConnectionState({
        status: "connected",
        message: `Connexion reussie a ${base}`,
        latencyMs
      });
      showToast("Connexion reussie", "success");
    } catch (error) {
      setConnectionState({
        status: "error",
        message: error instanceof Error ? error.message : "Echec de connexion au backend.",
        latencyMs: null
      });
    }
  }

  function disconnectBackend() {
    if (healthWatchRef.current) {
      clearInterval(healthWatchRef.current);
      healthWatchRef.current = null;
    }
    setConnectionState({
      status: "idle",
      message: "Backend deconnecte. Reconnectez-vous pour continuer.",
      latencyMs: null
    });
    setPage("connect");
    showToast("Connexion perdue", "warning");
  }

  return (
    <div className="app-shell">
      <div className="background-orbit background-orbit-left" />
      <div className="background-orbit background-orbit-right" />

      <Navigation
        page={page}
        onNavigate={setPage}
        isConnected={isConnected}
        connectionLabel={connectionLabel}
        onDisconnect={disconnectBackend}
      />

      {page === "connect" ? (
        <ConnectionPage
          credentialInput={credentialInput}
          passwordInput={passwordInput}
          onCredentialInputChange={setCredentialInput}
          onPasswordInputChange={setPasswordInput}
          connectionInput={connectionInput}
          onConnectionInputChange={setConnectionInput}
          onConnect={connectBackend}
          onContinue={() => setPage("overview")}
          status={connectionState.status}
          message={connectionState.message}
          connectionLabel={connectionLabel}
          latencyMs={connectionState.latencyMs}
        />
      ) : null}

      {page === "overview" ? (
        <OverviewPage
          onOpenTraining={() => (isConnected ? setPage("training") : setPage("connect"))}
          onOpenDatasets={() => (isConnected ? setPage("datasets") : setPage("connect"))}
        />
      ) : null}

      {page === "datasets" && isConnected ? <DatasetsPage apiFetch={apiFetch} /> : null}
      {page === "training" && isConnected ? <TrainingPage apiFetch={apiFetch} /> : null}
      {page === "evaluation" && isConnected ? <EvaluationPage apiFetch={apiFetch} /> : null}
      {page === "artifacts" && isConnected ? <ArtifactsPage apiFetch={apiFetch} /> : null}

      <SectionToast toast={toast} />
    </div>
  );
}

export default App;
