import { useState, useCallback, useRef, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  AreaChart,
  Area,
} from "recharts";

function seededRng(seed) {
  let s = (seed ^ 0xdeadbeef) >>> 0;
  return () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 0xffffffff;
  };
}
const pick = (arr, rng) => arr[Math.floor(rng() * arr.length)];
const uni = (lo, hi, rng) => lo + rng() * (hi - lo);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const PROFS = {
  low: [
    "Retail Worker",
    "Food Service",
    "Security Guard",
    "Factory Worker",
    "Delivery Driver",
    "Janitor",
    "Cashier",
    "Warehouse Worker",
  ],
  middle: [
    "Teacher",
    "Nurse",
    "Accountant",
    "Engineer",
    "IT Specialist",
    "Project Manager",
    "Sales Manager",
    "Technician",
  ],
  high: [
    "Doctor",
    "Lawyer",
    "CEO",
    "Investment Banker",
    "Consultant",
    "Senior Engineer",
    "Architect",
    "Executive",
  ],
};
const EDUC = {
  low: ["High School", "Some College"],
  middle: ["Some College", "Bachelor's", "Master's"],
  high: ["Bachelor's", "Master's", "PhD"],
};
const ZONES = ["downtown", "industrial", "suburban", "rural"];
const POLVW = ["conservative", "neutral", "progressive"];
const GNDRS = ["Male", "Female", "Non-binary"];

function generatePopulation(size, seed = 42) {
  const rng = seededRng(seed);
  const nL = Math.round(size * 0.4),
    nM = Math.round(size * 0.4),
    nH = size - nL - nM;
  const levels = [
    ...Array(nL).fill("low"),
    ...Array(nM).fill("middle"),
    ...Array(nH).fill("high"),
  ];
  for (let i = levels.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [levels[i], levels[j]] = [levels[j], levels[i]];
  }
  return Array.from({ length: size }, (_, i) => {
    const il = levels[i];
    const base_income =
      il === "low"
        ? uni(300, 700, rng)
        : il === "middle"
        ? uni(900, 2200, rng)
        : uni(2500, 9000, rng);
    const base_happiness =
      il === "low"
        ? uni(0.25, 0.52, rng)
        : il === "middle"
        ? uni(0.4, 0.68, rng)
        : uni(0.55, 0.82, rng);
    return {
      id: i,
      age: Math.floor(uni(18, 72, rng)),
      gender: pick(GNDRS, rng),
      city_zone: pick(ZONES, rng),
      income_level: il,
      education: pick(EDUC[il], rng),
      profession: pick(PROFS[il], rng),
      family_size: Math.floor(uni(1, 7, rng)),
      political_view: pick(POLVW, rng),
      risk_tolerance: rng(),
      openness_to_change: rng(),
      base_happiness,
      base_income,
    };
  });
}

function oneHot(val, arr) {
  return arr.map((a) => (a === val ? 1 : 0));
}
function buildFeature(citizen, prevState, domain) {
  return [
    citizen.age / 100,
    ...oneHot(citizen.income_level, ["low", "middle", "high"]),
    ...oneHot(citizen.city_zone, ZONES),
    ...oneHot(citizen.political_view, POLVW),
    citizen.risk_tolerance,
    citizen.openness_to_change,
    citizen.family_size / 10,
    prevState.happiness,
    prevState.policy_support,
    Math.log1p(prevState.income) / 10,
    ...oneHot(domain, ["Economy", "Education", "Social", "Startup"]),
  ];
}

class MLP {
  constructor(inputDim = 21, hidden = [64, 32], outputDim = 3) {
    const dims = [inputDim, ...hidden, outputDim];
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const fan_in = dims[i],
        fan_out = dims[i + 1];
      const limit = Math.sqrt(6 / (fan_in + fan_out));
      this.layers.push({
        W: Array.from({ length: fan_out }, () =>
          Array.from({ length: fan_in }, () => (Math.random() * 2 - 1) * limit)
        ),
        b: Array(fan_out).fill(0),
        isLast: i === dims.length - 2,
      });
    }
    this.trained = false;
    this.trainingSamples = 0;
    this.lossHistory = [];
  }
  relu(x) {
    return x.map((v) => Math.max(0, v));
  }
  matVec(W, x, b) {
    return W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], b[i]));
  }
  forward(x) {
    let h = x;
    for (const layer of this.layers) {
      h = this.matVec(layer.W, h, layer.b);
      if (!layer.isLast) h = this.relu(h);
    }
    return h;
  }
  train(X, Y, epochs = 80, lr = 0.003, batchSize = 8) {
    const n = X.length;
    const mW = this.layers.map((l) => l.W.map((r) => r.map(() => 0)));
    const vW = this.layers.map((l) => l.W.map((r) => r.map(() => 0)));
    const mb = this.layers.map((l) => l.b.map(() => 0));
    const vb = this.layers.map((l) => l.b.map(() => 0));
    const beta1 = 0.9,
      beta2 = 0.999,
      eps = 1e-8;
    let t = 0;
    for (let ep = 0; ep < epochs; ep++) {
      let epochLoss = 0,
        batchCount = 0;
      const idx = Array.from({ length: n }, (_, i) => i).sort(
        () => Math.random() - 0.5
      );
      for (let start = 0; start < n; start += batchSize) {
        const batch = idx.slice(start, start + batchSize);
        t++;
        const gW = this.layers.map((l) => l.W.map((r) => r.map(() => 0)));
        const gb = this.layers.map((l) => l.b.map(() => 0));
        for (const bi of batch) {
          const x = X[bi],
            y = Y[bi];
          const acts = [x],
            pres = [];
          let h = x;
          for (const layer of this.layers) {
            const z = this.matVec(layer.W, h, layer.b);
            pres.push(z);
            h = layer.isLast ? z : this.relu(z);
            acts.push(h);
          }
          const pred = acts[acts.length - 1];
          epochLoss +=
            pred.reduce((s, p, i) => s + (p - y[i]) ** 2, 0) / y.length;
          let delta = pred.map(
            (p, i) => (2 * (p - y[i])) / (y.length * batch.length)
          );
          for (let li = this.layers.length - 1; li >= 0; li--) {
            const layer = this.layers[li];
            const aIn = acts[li];
            for (let j = 0; j < layer.W.length; j++) {
              gb[li][j] += delta[j];
              for (let k = 0; k < layer.W[j].length; k++)
                gW[li][j][k] += delta[j] * aIn[k];
            }
            if (li > 0) {
              delta = aIn.map((_, k) => {
                let s = 0;
                for (let j = 0; j < layer.W.length; j++)
                  s += layer.W[j][k] * delta[j];
                return s * (pres[li - 1][k] > 0 ? 1 : 0);
              });
            }
          }
        }
        for (let li = 0; li < this.layers.length; li++) {
          for (let j = 0; j < this.layers[li].W.length; j++) {
            mb[li][j] = beta1 * mb[li][j] + (1 - beta1) * gb[li][j];
            vb[li][j] = beta2 * vb[li][j] + (1 - beta2) * gb[li][j] ** 2;
            const mhb = mb[li][j] / (1 - beta1 ** t),
              vhb = vb[li][j] / (1 - beta2 ** t);
            this.layers[li].b[j] -= (lr * mhb) / (Math.sqrt(vhb) + eps);
            for (let k = 0; k < this.layers[li].W[j].length; k++) {
              mW[li][j][k] = beta1 * mW[li][j][k] + (1 - beta1) * gW[li][j][k];
              vW[li][j][k] =
                beta2 * vW[li][j][k] + (1 - beta2) * gW[li][j][k] ** 2;
              const mhw = mW[li][j][k] / (1 - beta1 ** t),
                vhw = vW[li][j][k] / (1 - beta2 ** t);
              this.layers[li].W[j][k] -= (lr * mhw) / (Math.sqrt(vhw) + eps);
            }
          }
        }
        batchCount++;
      }
      this.lossHistory.push(+(epochLoss / batchCount).toFixed(6));
    }
    this.trained = true;
    this.trainingSamples = n;
  }
  predict(x) {
    return this.forward(x);
  }
}

const POLICY_FX = {
  "Fuel Subsidy Removal": {
    id: { low: [-80, -20], middle: [-60, -5], high: [-30, 30] },
    hb: { low: -0.22, middle: -0.1, high: 0.02 },
    sb: { low: -0.45, middle: -0.18, high: 0.08 },
    zm: { downtown: 0.04, industrial: -0.08, suburban: -0.04, rural: -0.14 },
  },
  "Student Scholarship Program": {
    id: { low: [20, 80], middle: [-5, 20], high: [-20, 5] },
    hb: { low: 0.28, middle: 0.08, high: -0.05 },
    sb: { low: 0.55, middle: 0.2, high: -0.1 },
    zm: { downtown: 0.02, industrial: 0.08, suburban: 0.04, rural: 0.12 },
  },
  "Universal Basic Income": {
    id: { low: [80, 200], middle: [20, 60], high: [-60, -10] },
    hb: { low: 0.35, middle: 0.15, high: -0.14 },
    sb: { low: 0.65, middle: 0.3, high: -0.32 },
    zm: { downtown: -0.02, industrial: 0.1, suburban: 0.05, rural: 0.14 },
  },
  "AI Tutor Initiative": {
    id: { low: [5, 30], middle: [-5, 15], high: [-10, 5] },
    hb: { low: 0.18, middle: 0.06, high: -0.02 },
    sb: { low: 0.35, middle: 0.15, high: 0.04 },
    zm: { downtown: 0.05, industrial: 0.06, suburban: 0.04, rural: 0.1 },
  },
  "Startup Delivery Fee Hike": {
    id: { low: [-60, -10], middle: [-20, 10], high: [10, 60] },
    hb: { low: -0.28, middle: -0.08, high: 0.08 },
    sb: { low: -0.55, middle: -0.15, high: 0.22 },
    zm: { downtown: 0.05, industrial: -0.14, suburban: -0.03, rural: -0.08 },
  },
};
const DOMAIN_FX = {
  Economy: {
    id: { low: [-50, 10], middle: [-30, 20], high: [-20, 50] },
    hb: { low: -0.15, middle: -0.05, high: 0.08 },
    sb: { low: -0.25, middle: -0.05, high: 0.12 },
    zm: { downtown: 0.02, industrial: -0.05, suburban: -0.02, rural: -0.06 },
  },
  Education: {
    id: { low: [10, 50], middle: [-5, 15], high: [-10, 5] },
    hb: { low: 0.18, middle: 0.07, high: -0.03 },
    sb: { low: 0.3, middle: 0.15, high: 0.02 },
    zm: { downtown: 0.02, industrial: 0.05, suburban: 0.03, rural: 0.07 },
  },
  Social: {
    id: { low: [30, 100], middle: [5, 30], high: [-40, -5] },
    hb: { low: 0.25, middle: 0.1, high: -0.1 },
    sb: { low: 0.45, middle: 0.2, high: -0.22 },
    zm: { downtown: -0.02, industrial: 0.07, suburban: 0.04, rural: 0.1 },
  },
  Startup: {
    id: { low: [-40, 10], middle: [-10, 20], high: [20, 80] },
    hb: { low: -0.12, middle: 0.03, high: 0.15 },
    sb: { low: -0.2, middle: 0.05, high: 0.3 },
    zm: { downtown: 0.1, industrial: -0.05, suburban: 0.02, rural: -0.04 },
  },
};

function ruleUpdate(citizen, prev, policy, step, rng) {
  const fxKey = Object.keys(POLICY_FX).find((k) =>
    policy.title.includes(k.split(" ")[0])
  );
  const fx = fxKey
    ? POLICY_FX[fxKey]
    : DOMAIN_FX[policy.domain] || DOMAIN_FX.Economy;
  const il = citizen.income_level,
    zone = citizen.city_zone,
    pv = citizen.political_view;
  const [dMin, dMax] = fx.id[il];
  const income_delta = uni(dMin, dMax, rng);
  const decay = 1 / (1 + step * 0.2);
  let hd =
    (fx.hb[il] +
      fx.zm[zone] * 0.5 +
      (citizen.openness_to_change - 0.5) * 0.12 +
      (citizen.risk_tolerance - 0.5) * 0.08) *
      decay +
    uni(-0.04, 0.04, rng);
  let sd =
    (fx.sb[il] +
      (pv === "progressive" && fx.sb.low > 0 ? 0.08 : 0) +
      (pv === "conservative" && fx.sb.high > 0 ? 0.07 : 0)) *
      decay +
    uni(-0.06, 0.06, rng);
  if (
    citizen.family_size > 3 &&
    (policy.domain === "Social" || policy.domain === "Education")
  ) {
    hd += 0.04;
    sd += 0.05;
  }
  return {
    happiness: clamp(prev.happiness + hd, 0, 1),
    policy_support: clamp(prev.policy_support + sd, -1, 1),
    income: Math.max(0, prev.income + income_delta),
  };
}

async function callGemini(citizens10, policy, prevStates, apiKey) {
  const profiles = citizens10.map((c, i) => ({
    id: c.id,
    age: c.age,
    income_level: c.income_level,
    profession: c.profession,
    political_view: c.political_view,
    city_zone: c.city_zone,
    family_size: c.family_size,
    risk_tolerance: +c.risk_tolerance.toFixed(2),
    openness_to_change: +c.openness_to_change.toFixed(2),
    current_happiness: +prevStates[i].happiness.toFixed(3),
    current_support: +prevStates[i].policy_support.toFixed(3),
    current_income: +prevStates[i].income.toFixed(0),
  }));
  const prompt = `You are simulating how fictional synthetic citizens react to a policy.\n\nPOLICY:\nTitle: ${
    policy.title
  }\nDescription: ${policy.description}\nDomain: ${
    policy.domain
  }\n\nCITIZENS:\n${JSON.stringify(
    profiles,
    null,
    2
  )}\n\nFor each citizen simulate a REALISTIC reaction. Make results VARY DRAMATICALLY between income groups and political views.\n\nReturn ONLY a valid JSON array with exactly ${
    profiles.length
  } objects. Each object:\n- "id": citizen id\n- "new_happiness": float 0.0-1.0\n- "new_policy_support": float -1.0 to 1.0\n- "income_delta": realistic dollar change\n- "diary_entry": 2 sentences first-person reaction\n\nIMPORTANT: Return raw JSON array only. No markdown, no backticks.`;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.7, maxOutputTokens: 1200 },
    }),
  });
  if (!res.ok) throw new Error(`Gemini API ${res.status}`);
  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  return JSON.parse(text.replace(/```json|```/g, "").trim());
}

function computeStats(step, states, citizenMap) {
  const n = states.length;
  const avg_happiness = states.reduce((a, s) => a + s.happiness, 0) / n;
  const avg_support = states.reduce((a, s) => a + s.policy_support, 0) / n;
  const avg_income = states.reduce((a, s) => a + s.income, 0) / n;
  const byI = {},
    byZ = {};
  states.forEach((s) => {
    const c = citizenMap[s.citizen_id];
    (byI[c.income_level] = byI[c.income_level] || []).push(s);
    (byZ[c.city_zone] = byZ[c.city_zone] || []).push(s);
  });
  const by_income = Object.entries(byI).map(([k, v]) => ({
    income_level: k,
    avg_happiness: v.reduce((a, s) => a + s.happiness, 0) / v.length,
    avg_support: v.reduce((a, s) => a + s.policy_support, 0) / v.length,
    avg_income: v.reduce((a, s) => a + s.income, 0) / v.length,
    count: v.length,
  }));
  const by_zone = Object.entries(byZ).map(([k, v]) => ({
    city_zone: k,
    avg_happiness: v.reduce((a, s) => a + s.happiness, 0) / v.length,
    avg_support: v.reduce((a, s) => a + s.policy_support, 0) / v.length,
  }));
  const low = by_income.find((g) => g.income_level === "low"),
    high = by_income.find((g) => g.income_level === "high");
  return {
    step,
    avg_happiness,
    avg_support,
    avg_income,
    by_income,
    by_zone,
    inequality_gap: high && low ? high.avg_happiness - low.avg_happiness : 0,
  };
}

const POLICIES = [
  {
    title: "Fuel Subsidy Removal",
    description:
      "Remove govt fuel subsidies to cut budget deficit. Gas prices rise 30%. Low-income hit hardest.",
    domain: "Economy",
    icon: "⛽",
  },
  {
    title: "Student Scholarship Program",
    description:
      "Provide $5,000/year scholarships to low-income students pursuing higher education.",
    domain: "Education",
    icon: "🎓",
  },
  {
    title: "Universal Basic Income",
    description:
      "UBI pilot: $500/month to every citizen for 6 months, funded by progressive tax on high earners.",
    domain: "Social",
    icon: "💰",
  },
  {
    title: "AI Tutor Initiative",
    description:
      "Deploy free AI-powered tutoring to all students in low-income neighbourhoods.",
    domain: "Education",
    icon: "🤖",
  },
  {
    title: "Startup Delivery Fee Hike",
    description:
      "Food-delivery startup raises customer fees 25% and cuts driver commissions by 10%.",
    domain: "Startup",
    icon: "🚀",
  },
];

const INCOME_COLORS = { low: "#f87171", middle: "#fbbf24", high: "#34d399" };
const ZONE_COLORS = {
  downtown: "#818cf8",
  industrial: "#fb923c",
  suburban: "#38bdf8",
  rural: "#a3e635",
};
const DOMAIN_ACCENT = {
  Economy: "#f59e0b",
  Education: "#6366f1",
  Social: "#10b981",
  Startup: "#ec4899",
};
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const supLbl = (v) =>
  v > 0.15 ? "Supportive 👍" : v < -0.15 ? "Opposed 👎" : "Divided 🤷";

function StatCard({ label, value, sub, color }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,.04)",
        border: "1px solid rgba(255,255,255,.08)",
        borderRadius: 12,
        padding: "14px 18px",
        flex: 1,
        minWidth: 110,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: "#8b9ab0",
          textTransform: "uppercase",
          letterSpacing: 1.2,
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 22,
          fontWeight: 800,
          color: color || "#e2e8f0",
          fontFamily: "monospace",
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>
          {sub}
        </div>
      )}
    </div>
  );
}
function Tag({ children, color }) {
  return (
    <span
      style={{
        background: color + "22",
        color,
        border: `1px solid ${color}44`,
        borderRadius: 6,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {children}
    </span>
  );
}
function SupportBar({ value }) {
  const col = value > 0.1 ? "#10b981" : value < -0.1 ? "#f87171" : "#fbbf24";
  return (
    <div
      style={{
        background: "rgba(255,255,255,.06)",
        borderRadius: 4,
        height: 7,
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: "50%",
          width: 1,
          height: "100%",
          background: "rgba(255,255,255,.2)",
          zIndex: 1,
        }}
      />
      <div
        style={{
          position: "absolute",
          left: value >= 0 ? "50%" : `${((value + 1) / 2) * 100}%`,
          width: `${Math.abs(value) * 50}%`,
          height: "100%",
          background: col,
          borderRadius: 4,
        }}
      />
    </div>
  );
}
function ChartCard({ title, children, span }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,.03)",
        border: "1px solid rgba(255,255,255,.07)",
        borderRadius: 12,
        padding: "16px 20px",
        gridColumn: span === 2 ? "1 / -1" : undefined,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "#94a3b8",
          marginBottom: 14,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}
const ttStyle = {
  background: "#1e293b",
  border: "1px solid rgba(255,255,255,.1)",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};
const inp = {
  width: "100%",
  background: "rgba(255,255,255,.05)",
  border: "1px solid rgba(255,255,255,.1)",
  borderRadius: 8,
  padding: "10px 14px",
  color: "#e2e8f0",
  fontSize: 14,
  outline: "none",
  boxSizing: "border-box",
  fontFamily: "inherit",
};

export default function App() {
  const [screen, setScreen] = useState("apikey");
  const [apiKey, setApiKey] = useState("");
  const [selPol, setSelPol] = useState(null);
  const [useCustom, setUseCustom] = useState(false);
  const [customPol, setCustomPol] = useState({
    title: "",
    description: "",
    domain: "Economy",
  });
  const [popSize, setPopSize] = useState(2000);
  const [steps, setSteps] = useState(5);
  const [results, setResults] = useState(null);
  const [activeStep, setActiveStep] = useState(5);
  const [activeTab, setActiveTab] = useState("overview");
  const [log, setLog] = useState([]);
  const [progress, setProgress] = useState(0);
  const [progMsg, setProgMsg] = useState("");

  const addLog = (msg) =>
    setLog((p) => [...p, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const runSim = useCallback(async () => {
    const policy = useCustom ? customPol : selPol;
    if (!policy) return;
    setScreen("running");
    setLog([]);
    setProgress(0);
    try {
      setProgMsg("🧬 Generating population…");
      setProgress(5);
      addLog(`Generating ${popSize.toLocaleString()} citizens…`);
      await new Promise((r) => setTimeout(r, 40));
      const citizens = generatePopulation(popSize, 42);
      const citizenMap = Object.fromEntries(citizens.map((c) => [c.id, c]));
      addLog(
        `✅ ${citizens.filter((c) => c.income_level === "low").length} low / ${
          citizens.filter((c) => c.income_level === "middle").length
        } middle / ${
          citizens.filter((c) => c.income_level === "high").length
        } high`
      );
      setProgress(10);

      let stateMap = {};
      citizens.forEach((c) => {
        stateMap[c.id] = {
          citizen_id: c.id,
          step: 0,
          happiness: c.base_happiness,
          policy_support: 0,
          income: c.base_income,
        };
      });
      const allStepStats = [
        computeStats(0, Object.values(stateMap), citizenMap),
      ];

      const rng0 = seededRng(777);
      const s10Ids = [
        ...citizens
          .filter((c) => c.income_level === "low")
          .sort(() => rng0() - 0.5)
          .slice(0, 3)
          .map((c) => c.id),
        ...citizens
          .filter((c) => c.income_level === "middle")
          .sort(() => rng0() - 0.5)
          .slice(0, 3)
          .map((c) => c.id),
        ...citizens
          .filter((c) => c.income_level === "high")
          .sort(() => rng0() - 0.5)
          .slice(0, 4)
          .map((c) => c.id),
      ];
      const s10 = s10Ids.map((id) => citizenMap[id]);
      const s10prev = s10Ids.map((id) => stateMap[id]);

      let llmResults = null,
        trainingX = [],
        trainingY = [],
        llmDiaries = [];
      setProgMsg("🤖 Calling Gemini LLM…");
      setProgress(15);
      addLog("📡 Calling Google Gemini API with 10 representative citizens…");
      try {
        llmResults = await callGemini(s10, policy, s10prev, apiKey);
        addLog(`✅ Gemini returned ${llmResults.length} reactions`);
        llmDiaries = llmResults.map((r) => ({
          id: r.id,
          diary: r.diary_entry,
        }));
        setProgress(35);
        llmResults.forEach((res, i) => {
          const c = s10[i],
            prev = s10prev[i];
          trainingX.push(buildFeature(c, prev, policy.domain));
          trainingY.push([
            clamp(res.new_happiness, 0, 1) - prev.happiness,
            clamp(res.new_policy_support, -1, 1) - prev.policy_support,
            res.income_delta || 0,
          ]);
        });
        addLog(`🧠 Built ${trainingX.length} training samples`);
      } catch (e) {
        addLog(`⚠️ LLM failed (${e.message}) — using rule-based`);
      }

      let nn = null,
        nnStats = null;
      if (trainingX.length >= 5) {
        setProgMsg("🧠 Training Neural Network…");
        setProgress(40);
        addLog("Training MLP 21→64→32→3 with Adam SGD (80 epochs)…");
        await new Promise((r) => setTimeout(r, 20));
        nn = new MLP(21, [64, 32], 3);
        nn.train(trainingX, trainingY, 80, 0.003, 8);
        nnStats = {
          samples: trainingX.length,
          epochs: 80,
          finalLoss: nn.lossHistory[nn.lossHistory.length - 1],
          lossHistory: nn.lossHistory,
        };
        addLog(`✅ NN trained! Final MSE: ${nnStats.finalLoss.toFixed(6)}`);
        setProgress(55);
      }

      const simRng = seededRng(
        policy.title.split("").reduce((a, c) => a + c.charCodeAt(0), 0) * 7919
      );
      for (let step = 1; step <= steps; step++) {
        setProgMsg(`⚙️ Step ${step}/${steps}…`);
        const newSM = {};
        let nL = 0,
          nN = 0,
          nR = 0;
        citizens.forEach((c) => {
          const prev = stateMap[c.id];
          const hit =
            step === 1 && llmResults && llmResults.find((r) => r.id === c.id);
          if (hit) {
            newSM[c.id] = {
              citizen_id: c.id,
              step,
              happiness: clamp(hit.new_happiness, 0, 1),
              policy_support: clamp(hit.new_policy_support, -1, 1),
              income: Math.max(0, prev.income + (hit.income_delta || 0)),
            };
            nL++;
          } else if (nn && nn.trained) {
            const X = buildFeature(c, prev, policy.domain);
            const [dh, ds, di] = nn.predict(X);
            const d = 1 / (1 + step * 0.18);
            newSM[c.id] = {
              citizen_id: c.id,
              step,
              happiness: clamp(prev.happiness + dh * d, 0, 1),
              policy_support: clamp(prev.policy_support + ds * d, -1, 1),
              income: Math.max(0, prev.income + di),
            };
            nN++;
          } else {
            const u = ruleUpdate(c, prev, policy, step, simRng);
            newSM[c.id] = { citizen_id: c.id, step, ...u };
            nR++;
          }
        });
        stateMap = newSM;
        allStepStats.push(
          computeStats(step, Object.values(stateMap), citizenMap)
        );
        addLog(`Step ${step} — LLM:${nL} NN:${nN} Rule:${nR}`);
        setProgress(55 + (step / steps) * 38);
        await new Promise((r) => setTimeout(r, 15));
      }
      setProgress(98);
      setProgMsg("📊 Finalising…");
      await new Promise((r) => setTimeout(r, 80));
      setResults({
        stepStats: allStepStats,
        citizens,
        finalStates: Object.values(stateMap),
        policy,
        llmDiaries,
        nnStats,
      });
      setActiveStep(steps);
      setActiveTab("overview");
      setProgress(100);
      addLog(`✅ Done! ${popSize.toLocaleString()} citizens × ${steps} steps`);
      setTimeout(() => setScreen("results"), 300);
    } catch (err) {
      addLog(`❌ ${err.message}`);
      setProgMsg("❌ Error");
    }
  }, [useCustom, customPol, selPol, popSize, steps, apiKey]);

  const bg = {
    minHeight: "100vh",
    background: "#070d1a",
    fontFamily: "'Outfit','Segoe UI',sans-serif",
    color: "#e2e8f0",
    position: "relative",
  };
  return (
    <div style={bg}>
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 0,
          pointerEvents: "none",
          backgroundImage:
            "linear-gradient(rgba(99,102,241,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,.03) 1px,transparent 1px)",
          backgroundSize: "44px 44px",
        }}
      />
      <div style={{ position: "relative", zIndex: 1 }}>
        {screen === "apikey" && (
          <ApiKeyScreen
            onContinue={(k) => {
              setApiKey(k);
              setScreen("home");
            }}
          />
        )}
        {screen === "home" && (
          <HomeScreen onStart={() => setScreen("config")} />
        )}
        {screen === "config" && (
          <ConfigScreen
            policies={POLICIES}
            selected={selPol}
            onSelect={(p) => {
              setSelPol(p);
              setUseCustom(false);
            }}
            custom={customPol}
            setCustom={setCustomPol}
            useCustom={useCustom}
            setUseCustom={setUseCustom}
            popSize={popSize}
            setPopSize={setPopSize}
            steps={steps}
            setSteps={setSteps}
            onBack={() => setScreen("home")}
            onRun={runSim}
          />
        )}
        {screen === "running" && (
          <RunningScreen
            progress={progress}
            msg={progMsg}
            log={log}
            policy={useCustom ? customPol : selPol}
          />
        )}
        {screen === "results" && results && (
          <ResultsScreen
            results={results}
            activeStep={activeStep}
            setActiveStep={setActiveStep}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            onNew={() => {
              setScreen("config");
              setResults(null);
            }}
          />
        )}
      </div>
    </div>
  );
}

function ApiKeyScreen({ onContinue }) {
  const [key, setKey] = useState("");
  const [err, setErr] = useState("");
  const [testing, setTesting] = useState(false);
  const go = async () => {
    const k = key.trim();
    if (!k) {
      onContinue("");
      return;
    }
    setTesting(true);
    setErr("");
    try {
      const r = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${k}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: "hi" }] }],
            generationConfig: { maxOutputTokens: 5 },
          }),
        }
      );
      const t = await r.text();
      if (r.status === 400 || r.status === 403) {
        if (t.includes("API_KEY_INVALID") || t.includes("invalid")) {
          setErr("Invalid key — check and try again.");
          setTesting(false);
          return;
        }
      }
      onContinue(k);
    } catch (e) {
      setErr("Connection error: " + e.message);
      setTesting(false);
    }
  };
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: 32,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 52, marginBottom: 12 }}>🏙️</div>
      <h1
        style={{
          fontSize: "clamp(2rem,5vw,3rem)",
          fontWeight: 900,
          margin: "0 0 8px",
          letterSpacing: -2,
        }}
      >
        <span
          style={{
            background: "linear-gradient(135deg,#818cf8,#6366f1,#10b981)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          CIVI-GENESIS
        </span>
      </h1>
      <p
        style={{
          color: "#94a3b8",
          fontSize: 13,
          maxWidth: 480,
          lineHeight: 1.7,
          margin: "0 auto 10px",
        }}
      >
        AI Policy Simulator — Gemini LLM + Neural Network ML
      </p>
      <div
        style={{
          background: "rgba(16,185,129,.08)",
          border: "1px solid rgba(16,185,129,.25)",
          borderRadius: 10,
          padding: "8px 18px",
          fontSize: 12,
          color: "#6ee7b7",
          marginBottom: 24,
          maxWidth: 400,
        }}
      >
        ✅ <strong>100% FREE</strong> — Google Gemini: 1,500 requests/day, no
        credit card
      </div>
      <div
        style={{
          width: "100%",
          maxWidth: 440,
          background: "rgba(255,255,255,.04)",
          border: "1px solid rgba(255,255,255,.1)",
          borderRadius: 14,
          padding: 24,
          textAlign: "left",
          marginBottom: 18,
        }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: "#94a3b8",
            marginBottom: 12,
          }}
        >
          📋 Get FREE Gemini API key (2 min):
        </div>
        {[
          ["1", "Go to", "aistudio.google.com", "→ sign in with Google"],
          ["2", "Click", '"Get API Key"', "on the left sidebar"],
          ["3", "Click", '"Create API key"', "→ copy it"],
          ["4", "Paste below", "and click Launch", ""],
        ].map(([n, a, b, c]) => (
          <div
            key={n}
            style={{
              display: "flex",
              gap: 10,
              marginBottom: 8,
              fontSize: 12,
              color: "#64748b",
              alignItems: "flex-start",
            }}
          >
            <span
              style={{
                background: "rgba(16,185,129,.25)",
                color: "#6ee7b7",
                borderRadius: "50%",
                width: 20,
                height: 20,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 11,
                fontWeight: 800,
                flexShrink: 0,
                marginTop: 1,
              }}
            >
              {n}
            </span>
            <span>
              {a} <strong style={{ color: "#a5b4fc" }}>{b}</strong> {c}
            </span>
          </div>
        ))}
        <a
          href="https://aistudio.google.com/app/apikey"
          target="_blank"
          rel="noreferrer"
          style={{
            display: "block",
            marginTop: 12,
            background: "linear-gradient(135deg,#10b981,#059669)",
            color: "#fff",
            borderRadius: 8,
            padding: "9px 16px",
            fontSize: 13,
            fontWeight: 700,
            textDecoration: "none",
            textAlign: "center",
          }}
        >
          🔗 Open Google AI Studio →
        </a>
      </div>
      <div style={{ width: "100%", maxWidth: 440, marginBottom: 10 }}>
        <input
          value={key}
          onChange={(e) => setKey(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
          placeholder="Paste Gemini key here (AIza...)"
          type="password"
          style={{ ...inp, fontSize: 13 }}
        />
      </div>
      {err && (
        <div style={{ color: "#f87171", fontSize: 12, marginBottom: 10 }}>
          ⚠️ {err}
        </div>
      )}
      <button
        onClick={go}
        disabled={testing}
        style={{
          background: "linear-gradient(135deg,#10b981,#6366f1)",
          color: "#fff",
          border: "none",
          borderRadius: 10,
          padding: "12px 36px",
          fontSize: 14,
          fontWeight: 800,
          cursor: testing ? "wait" : "pointer",
          boxShadow: "0 0 24px rgba(16,185,129,.3)",
        }}
      >
        {testing
          ? "⏳ Testing…"
          : key.trim()
          ? "✅ Launch Simulator"
          : "▶ Skip — rule-based only"}
      </button>
      <p style={{ color: "#334155", fontSize: 11, marginTop: 12 }}>
        No key? Click Skip — app still works without LLM.
      </p>
    </div>
  );
}

function HomeScreen({ onStart }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: 32,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 60, marginBottom: 16 }}>🏙️</div>
      <h1
        style={{
          fontSize: "clamp(2.4rem,6vw,4rem)",
          fontWeight: 900,
          margin: "0 0 12px",
          letterSpacing: -2,
        }}
      >
        <span
          style={{
            background: "linear-gradient(135deg,#818cf8,#6366f1,#10b981)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          CIVI-GENESIS
        </span>
      </h1>
      <p
        style={{
          color: "#94a3b8",
          maxWidth: 520,
          lineHeight: 1.7,
          margin: "0 auto 16px",
          fontSize: 15,
        }}
      >
        <strong style={{ color: "#c7d2fe" }}>
          Gemini LLM + Neural Network ML
        </strong>{" "}
        — simulate how thousands of citizens react to real policies
      </p>
      <div
        style={{
          display: "flex",
          gap: 10,
          justifyContent: "center",
          flexWrap: "wrap",
          marginBottom: 32,
        }}
      >
        {[
          ["🤖", "Gemini LLM", "Free Google AI generates citizen reactions"],
          ["🧠", "Neural Net", "MLP 21→64→32→3, Adam SGD"],
          ["📊", "Analytics", "Happiness, income, inequality charts"],
          ["👥", "2,000 Citizens", "Diverse demographics & income levels"],
        ].map(([icon, t, d]) => (
          <div
            key={t}
            style={{
              background: "rgba(255,255,255,.04)",
              border: "1px solid rgba(255,255,255,.07)",
              borderRadius: 12,
              padding: "14px 16px",
              width: 148,
              textAlign: "left",
            }}
          >
            <div style={{ fontSize: 22, marginBottom: 6 }}>{icon}</div>
            <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 3 }}>
              {t}
            </div>
            <div style={{ fontSize: 11, color: "#64748b", lineHeight: 1.4 }}>
              {d}
            </div>
          </div>
        ))}
      </div>
      <button
        onClick={onStart}
        style={{
          background: "linear-gradient(135deg,#6366f1,#818cf8)",
          color: "#fff",
          border: "none",
          borderRadius: 12,
          padding: "14px 44px",
          fontSize: 16,
          fontWeight: 800,
          cursor: "pointer",
          boxShadow: "0 0 32px rgba(99,102,241,.4)",
        }}
      >
        🚀 Launch Simulator
      </button>
    </div>
  );
}

function ConfigScreen({
  policies,
  selected,
  onSelect,
  custom,
  setCustom,
  useCustom,
  setUseCustom,
  popSize,
  setPopSize,
  steps,
  setSteps,
  onBack,
  onRun,
}) {
  const active = useCustom ? custom : selected;
  const canRun =
    active && (useCustom ? custom.title && custom.description : true);
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 20px" }}>
      <button
        onClick={onBack}
        style={{
          background: "none",
          border: "1px solid rgba(255,255,255,.12)",
          color: "#94a3b8",
          borderRadius: 8,
          padding: "6px 14px",
          cursor: "pointer",
          fontSize: 13,
          marginBottom: 24,
        }}
      >
        ← Back
      </button>
      <h2 style={{ fontSize: 26, fontWeight: 800, marginBottom: 4 }}>
        Configure Simulation
      </h2>
      <p style={{ color: "#64748b", marginBottom: 24, fontSize: 13 }}>
        Pick a policy preset or write your own
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill,minmax(155px,1fr))",
          gap: 10,
          marginBottom: 14,
        }}
      >
        {policies.map((p) => {
          const acc = DOMAIN_ACCENT[p.domain],
            sel = !useCustom && selected?.title === p.title;
          return (
            <div
              key={p.title}
              onClick={() => onSelect(p)}
              style={{
                background: sel ? `${acc}18` : "rgba(255,255,255,.03)",
                border: `1px solid ${sel ? acc : "rgba(255,255,255,.07)"}`,
                borderRadius: 10,
                padding: "14px 16px",
                cursor: "pointer",
                position: "relative",
              }}
            >
              {sel && (
                <div
                  style={{
                    position: "absolute",
                    top: 8,
                    right: 8,
                    width: 7,
                    height: 7,
                    borderRadius: "50%",
                    background: acc,
                  }}
                />
              )}
              <div style={{ fontSize: 22, marginBottom: 8 }}>{p.icon}</div>
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  marginBottom: 5,
                  lineHeight: 1.3,
                }}
              >
                {p.title}
              </div>
              <Tag color={acc}>{p.domain}</Tag>
            </div>
          );
        })}
      </div>
      <button
        onClick={() => setUseCustom(!useCustom)}
        style={{
          background: useCustom
            ? "rgba(99,102,241,.15)"
            : "rgba(255,255,255,.04)",
          border: `1px solid ${useCustom ? "#6366f1" : "rgba(255,255,255,.1)"}`,
          borderRadius: 8,
          padding: "7px 16px",
          color: useCustom ? "#818cf8" : "#64748b",
          cursor: "pointer",
          fontSize: 12,
          marginBottom: 14,
          fontWeight: 700,
        }}
      >
        ✏️ {useCustom ? "Using Custom Policy" : "Create Custom Policy"}
      </button>
      {useCustom && (
        <div
          style={{
            background: "rgba(255,255,255,.03)",
            border: "1px solid rgba(255,255,255,.08)",
            borderRadius: 12,
            padding: 18,
            marginBottom: 14,
          }}
        >
          <input
            value={custom.title}
            onChange={(e) => setCustom({ ...custom, title: e.target.value })}
            placeholder="Policy title…"
            style={inp}
          />
          <textarea
            value={custom.description}
            onChange={(e) =>
              setCustom({ ...custom, description: e.target.value })
            }
            placeholder="Describe the policy…"
            rows={3}
            style={{ ...inp, resize: "vertical", marginTop: 10 }}
          />
          <select
            value={custom.domain}
            onChange={(e) => setCustom({ ...custom, domain: e.target.value })}
            style={{ ...inp, marginTop: 10 }}
          >
            {["Economy", "Education", "Social", "Startup"].map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </div>
      )}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 14,
          marginBottom: 22,
        }}
      >
        <div
          style={{
            background: "rgba(255,255,255,.03)",
            border: "1px solid rgba(255,255,255,.08)",
            borderRadius: 12,
            padding: 18,
          }}
        >
          <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 8 }}>
            👥 Population:{" "}
            <strong style={{ color: "#e2e8f0" }}>
              {popSize.toLocaleString()}
            </strong>
          </div>
          <input
            type="range"
            min={500}
            max={5000}
            step={500}
            value={popSize}
            onChange={(e) => setPopSize(+e.target.value)}
            style={{ width: "100%", accentColor: "#6366f1" }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 10,
              color: "#475569",
              marginTop: 4,
            }}
          >
            <span>500</span>
            <span>5,000</span>
          </div>
        </div>
        <div
          style={{
            background: "rgba(255,255,255,.03)",
            border: "1px solid rgba(255,255,255,.08)",
            borderRadius: 12,
            padding: 18,
          }}
        >
          <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 8 }}>
            ⏱ Steps: <strong style={{ color: "#e2e8f0" }}>{steps}</strong>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            value={steps}
            onChange={(e) => setSteps(+e.target.value)}
            style={{ width: "100%", accentColor: "#6366f1" }}
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 10,
              color: "#475569",
              marginTop: 4,
            }}
          >
            <span>1</span>
            <span>10</span>
          </div>
        </div>
      </div>
      {active && (
        <div
          style={{
            background: `${DOMAIN_ACCENT[active.domain]}0d`,
            border: `1px solid ${DOMAIN_ACCENT[active.domain]}33`,
            borderRadius: 12,
            padding: 14,
            marginBottom: 20,
            fontSize: 13,
            lineHeight: 1.7,
            color: "#cbd5e1",
          }}
        >
          <strong style={{ color: "#e2e8f0" }}>{active.title}</strong>
          <br />
          {active.description}
        </div>
      )}
      <button
        onClick={onRun}
        disabled={!canRun}
        style={{
          background: canRun
            ? "linear-gradient(135deg,#6366f1,#818cf8)"
            : "rgba(255,255,255,.05)",
          color: canRun ? "#fff" : "#475569",
          border: "none",
          borderRadius: 12,
          padding: "14px 40px",
          fontSize: 16,
          fontWeight: 800,
          cursor: canRun ? "pointer" : "not-allowed",
          width: "100%",
          boxShadow: canRun ? "0 0 28px rgba(99,102,241,.3)" : "none",
        }}
      >
        🚀 Run Simulation
      </button>
    </div>
  );
}

function RunningScreen({ progress, msg, log, policy }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [log]);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: 32,
      }}
    >
      <div style={{ fontSize: 44, marginBottom: 16 }}>⚙️</div>
      <h2 style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
        Running Simulation
      </h2>
      <p style={{ color: "#64748b", marginBottom: 24, fontSize: 13 }}>
        <em style={{ color: "#c7d2fe" }}>{policy?.title}</em>
      </p>
      <div
        style={{
          width: 360,
          height: 7,
          background: "rgba(255,255,255,.06)",
          borderRadius: 6,
          overflow: "hidden",
          marginBottom: 8,
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${progress}%`,
            background: "linear-gradient(90deg,#6366f1,#10b981)",
            borderRadius: 6,
            transition: "width .3s",
          }}
        />
      </div>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 13,
          color: "#6366f1",
          marginBottom: 6,
        }}
      >
        {Math.floor(progress)}%
      </div>
      <div
        style={{
          fontSize: 12,
          color: "#94a3b8",
          marginBottom: 18,
          minHeight: 18,
        }}
      >
        {msg}
      </div>
      <div
        ref={ref}
        style={{
          width: "100%",
          maxWidth: 500,
          height: 180,
          overflowY: "auto",
          background: "rgba(0,0,0,.4)",
          border: "1px solid rgba(255,255,255,.07)",
          borderRadius: 10,
          padding: "10px 14px",
          fontFamily: "monospace",
          fontSize: 11,
          color: "#64748b",
          lineHeight: 1.9,
        }}
      >
        {log.map((l, i) => (
          <div
            key={i}
            style={{
              color: l.includes("✅")
                ? "#34d399"
                : l.includes("❌")
                ? "#f87171"
                : l.includes("⚠️")
                ? "#fbbf24"
                : "#64748b",
            }}
          >
            {l}
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultsScreen({
  results,
  activeStep,
  setActiveStep,
  activeTab,
  setActiveTab,
  onNew,
}) {
  const { stepStats, citizens, finalStates, policy, llmDiaries, nnStats } =
    results;
  const cur = stepStats[activeStep];
  const acc = DOMAIN_ACCENT[policy.domain] || "#6366f1";
  const TABS = [
    ["overview", "📊 Overview"],
    ["groups", "👥 Groups"],
    ["citizens", "👤 Citizens"],
    ["ml", "🧠 ML"],
    ["experts", "🎓 Experts"],
  ];
  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: "24px 16px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 18,
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 4,
            }}
          >
            <h2 style={{ fontSize: 20, fontWeight: 800, margin: 0 }}>
              {policy.title}
            </h2>
            <Tag color={acc}>{policy.domain}</Tag>
            {nnStats && <Tag color="#818cf8">LLM+NN ✓</Tag>}
          </div>
          <p style={{ color: "#64748b", fontSize: 12, margin: 0 }}>
            {citizens.length.toLocaleString()} citizens · {stepStats.length - 1}{" "}
            steps
          </p>
        </div>
        <button
          onClick={onNew}
          style={{
            background: "rgba(255,255,255,.06)",
            border: "1px solid rgba(255,255,255,.12)",
            color: "#94a3b8",
            borderRadius: 8,
            padding: "7px 14px",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          + New
        </button>
      </div>
      <div
        style={{
          background: "rgba(255,255,255,.03)",
          border: "1px solid rgba(255,255,255,.07)",
          borderRadius: 12,
          padding: "10px 18px",
          marginBottom: 18,
          display: "flex",
          alignItems: "center",
          gap: 14,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 12, color: "#94a3b8" }}>
          Step {activeStep}
        </span>
        <input
          type="range"
          min={0}
          max={stepStats.length - 1}
          value={activeStep}
          onChange={(e) => setActiveStep(+e.target.value)}
          style={{ flex: 1, minWidth: 120, accentColor: acc }}
        />
        <span style={{ fontSize: 11, color: "#475569" }}>
          {activeStep === 0 ? "Baseline" : `Step ${activeStep}`}
        </span>
      </div>
      <div
        style={{ display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" }}
      >
        <StatCard
          label="Avg Happiness"
          value={pct(cur.avg_happiness)}
          color={
            cur.avg_happiness > 0.6
              ? "#34d399"
              : cur.avg_happiness < 0.4
              ? "#f87171"
              : "#fbbf24"
          }
        />
        <StatCard
          label="Policy Support"
          value={supLbl(cur.avg_support)}
          color={cur.avg_support > 0 ? "#34d399" : "#f87171"}
        />
        <StatCard
          label="Avg Income"
          value={`$${Math.round(cur.avg_income).toLocaleString()}`}
          color="#818cf8"
        />
        <StatCard
          label="Inequality Gap"
          value={`${(cur.inequality_gap * 100).toFixed(1)}pp`}
          color={Math.abs(cur.inequality_gap) > 0.15 ? "#f87171" : "#34d399"}
        />
      </div>
      <div
        style={{
          display: "flex",
          gap: 4,
          marginBottom: 16,
          background: "rgba(255,255,255,.03)",
          borderRadius: 10,
          padding: 4,
          width: "fit-content",
          flexWrap: "wrap",
        }}
      >
        {TABS.map(([t, l]) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            style={{
              background:
                activeTab === t ? "rgba(99,102,241,.3)" : "transparent",
              border:
                activeTab === t
                  ? "1px solid rgba(99,102,241,.5)"
                  : "1px solid transparent",
              color: activeTab === t ? "#c7d2fe" : "#64748b",
              borderRadius: 8,
              padding: "6px 14px",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            {l}
          </button>
        ))}
      </div>
      {activeTab === "overview" && (
        <OverviewTab stepStats={stepStats} accent={acc} />
      )}
      {activeTab === "groups" && <GroupsTab cur={cur} />}
      {activeTab === "citizens" && (
        <CitizensTab
          citizens={citizens}
          finalStates={finalStates}
          llmDiaries={llmDiaries}
        />
      )}
      {activeTab === "ml" && (
        <MLTab nnStats={nnStats} llmDiaries={llmDiaries} policy={policy} />
      )}
      {activeTab === "experts" && <ExpertsTab policy={policy} cur={cur} />}
    </div>
  );
}

function OverviewTab({ stepStats, accent }) {
  const data = stepStats.map((s) => ({
    step: `S${s.step}`,
    happiness: +(s.avg_happiness * 100).toFixed(1),
    support: +(s.avg_support * 100).toFixed(1),
    income: +s.avg_income.toFixed(0),
    gap: +(s.inequality_gap * 100).toFixed(1),
  }));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <ChartCard title="😊 Happiness Over Time">
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="gh" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="step"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Area
              type="monotone"
              dataKey="happiness"
              stroke="#34d399"
              strokeWidth={2.5}
              fill="url(#gh)"
              dot={{ r: 3, fill: "#34d399" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="🗳️ Policy Support Over Time">
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="gs" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={accent} stopOpacity={0.25} />
                <stop offset="95%" stopColor={accent} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="step"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[-100, 100]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Area
              type="monotone"
              dataKey="support"
              stroke={accent}
              strokeWidth={2.5}
              fill="url(#gs)"
              dot={{ r: 3, fill: accent }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="💵 Average Income" span={2}>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={data}>
            <XAxis
              dataKey="step"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `$${v}`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Line
              type="monotone"
              dataKey="income"
              stroke="#818cf8"
              strokeWidth={2.5}
              dot={{ r: 3, fill: "#818cf8" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="⚖️ Inequality Gap" span={2}>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={data} barSize={26}>
            <XAxis
              dataKey="step"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}pp`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Bar dataKey="gap" radius={[5, 5, 0, 0]}>
              {data.map((d, i) => (
                <Cell
                  key={i}
                  fill={
                    d.gap > 10 ? "#f87171" : d.gap < -5 ? "#34d399" : "#fbbf24"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function GroupsTab({ cur }) {
  const inc = ["low", "middle", "high"]
    .map((k) => {
      const g = cur.by_income.find((x) => x.income_level === k);
      return g
        ? {
            name: k[0].toUpperCase() + k.slice(1),
            happiness: +(g.avg_happiness * 100).toFixed(1),
            support: +(g.avg_support * 100).toFixed(1),
            income: +g.avg_income.toFixed(0),
            fill: INCOME_COLORS[k],
          }
        : null;
    })
    .filter(Boolean);
  const zn = cur.by_zone.map((g) => ({
    name: g.city_zone[0].toUpperCase() + g.city_zone.slice(1),
    happiness: +(g.avg_happiness * 100).toFixed(1),
    fill: ZONE_COLORS[g.city_zone],
  }));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <ChartCard title="💰 Happiness by Income">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={inc} barSize={40}>
            <XAxis
              dataKey="name"
              tick={{ fill: "#64748b", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Bar dataKey="happiness" radius={[6, 6, 0, 0]}>
              {inc.map((e, i) => (
                <Cell key={i} fill={e.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="📍 Happiness by Zone">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={zn} barSize={32}>
            <XAxis
              dataKey="name"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip contentStyle={ttStyle} />
            <Bar dataKey="happiness" radius={[6, 6, 0, 0]}>
              {zn.map((e, i) => (
                <Cell key={i} fill={e.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="🗳️ Support by Income Group" span={2}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3,1fr)",
            gap: 12,
          }}
        >
          {inc.map((g) => (
            <div
              key={g.name}
              style={{
                background: "rgba(255,255,255,.03)",
                borderRadius: 10,
                padding: 16,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 8,
                }}
              >
                <span style={{ fontSize: 14, fontWeight: 800, color: g.fill }}>
                  {g.name}
                </span>
                <span
                  style={{
                    fontSize: 14,
                    color: g.support > 0 ? "#34d399" : "#f87171",
                    fontWeight: 800,
                  }}
                >
                  {g.support > 0 ? "+" : ""}
                  {g.support.toFixed(1)}%
                </span>
              </div>
              <SupportBar value={g.support / 100} />
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 8 }}>
                Income:{" "}
                <span style={{ color: "#818cf8", fontWeight: 700 }}>
                  ${g.income.toLocaleString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}

function CitizensTab({ citizens, finalStates, llmDiaries }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const sm = Object.fromEntries(finalStates.map((s) => [s.citizen_id, s]));
  const dm = Object.fromEntries((llmDiaries || []).map((d) => [d.id, d.diary]));
  const list = citizens
    .filter((c) => filter === "all" || c.income_level === filter)
    .filter(
      (c) =>
        !search ||
        c.profession.toLowerCase().includes(search.toLowerCase()) ||
        c.city_zone.toLowerCase().includes(search.toLowerCase())
    )
    .slice(0, 36);
  return (
    <div>
      <div
        style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}
      >
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search profession / zone…"
          style={{ ...inp, width: "auto", flex: 1, minWidth: 140 }}
        />
        {["all", "low", "middle", "high"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              background:
                filter === f ? "rgba(99,102,241,.2)" : "rgba(255,255,255,.04)",
              border: `1px solid ${
                filter === f ? "#6366f1" : "rgba(255,255,255,.1)"
              }`,
              color: filter === f ? "#c7d2fe" : "#64748b",
              borderRadius: 8,
              padding: "6px 14px",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 700,
            }}
          >
            {f === "all" ? "All" : f[0].toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill,minmax(240px,1fr))",
          gap: 10,
        }}
      >
        {list.map((c) => {
          const s = sm[c.id];
          if (!s) return null;
          const hCol =
            s.happiness > 0.6
              ? "#34d399"
              : s.happiness < 0.4
              ? "#f87171"
              : "#fbbf24";
          const diary = dm[c.id];
          return (
            <div
              key={c.id}
              style={{
                background: "rgba(255,255,255,.03)",
                border: `1px solid ${
                  diary ? "rgba(99,102,241,.3)" : "rgba(255,255,255,.07)"
                }`,
                borderRadius: 10,
                padding: 14,
              }}
            >
              {diary && (
                <div
                  style={{
                    fontSize: 10,
                    color: "#818cf8",
                    marginBottom: 5,
                    fontWeight: 700,
                  }}
                >
                  🤖 LLM-Analysed
                </div>
              )}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 8,
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}>
                    {c.profession}
                  </div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>
                    {c.city_zone} · {c.political_view}
                  </div>
                </div>
                <Tag color={INCOME_COLORS[c.income_level]}>
                  {c.income_level}
                </Tag>
              </div>
              <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                <div
                  style={{
                    flex: 1,
                    background: "rgba(255,255,255,.04)",
                    borderRadius: 6,
                    padding: "6px 8px",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: 10, color: "#64748b" }}>
                    Happiness
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 800, color: hCol }}>
                    {pct(s.happiness)}
                  </div>
                </div>
                <div
                  style={{
                    flex: 1,
                    background: "rgba(255,255,255,.04)",
                    borderRadius: 6,
                    padding: "6px 8px",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: 10, color: "#64748b" }}>Income</div>
                  <div
                    style={{ fontSize: 15, fontWeight: 800, color: "#818cf8" }}
                  >
                    ${Math.round(s.income)}
                  </div>
                </div>
              </div>
              <SupportBar value={s.policy_support} />
              {diary && (
                <div
                  style={{
                    marginTop: 10,
                    fontSize: 11,
                    color: "#94a3b8",
                    lineHeight: 1.6,
                    fontStyle: "italic",
                    borderTop: "1px solid rgba(255,255,255,.06)",
                    paddingTop: 8,
                  }}
                >
                  "{diary}"
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MLTab({ nnStats, llmDiaries, policy }) {
  if (!nnStats)
    return (
      <div
        style={{
          background: "rgba(255,165,0,.07)",
          border: "1px solid rgba(255,165,0,.2)",
          borderRadius: 12,
          padding: 28,
          textAlign: "center",
          color: "#fbbf24",
        }}
      >
        <div style={{ fontSize: 36, marginBottom: 12 }}>⚠️</div>
        <div style={{ fontWeight: 800, marginBottom: 6 }}>
          LLM not used — rule-based fallback active
        </div>
        <div style={{ fontSize: 13, color: "#94a3b8" }}>
          Add a Gemini API key to enable LLM + Neural Network mode.
        </div>
      </div>
    );
  const lossData = nnStats.lossHistory.map((l, i) => ({
    epoch: i + 1,
    loss: l,
  }));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <ChartCard title="🧠 Neural Network Training Loss" span={2}>
        <div
          style={{
            display: "flex",
            gap: 12,
            marginBottom: 14,
            flexWrap: "wrap",
          }}
        >
          {[
            ["LLM Samples", nnStats.samples, "From Gemini API", "#818cf8"],
            ["Epochs", nnStats.epochs, "Adam SGD", "#34d399"],
            ["Final MSE", nnStats.finalLoss.toFixed(6), "Loss", "#fbbf24"],
            ["Architecture", "21→64→32→3", "MLP", "#f87171"],
          ].map(([l, v, s, c]) => (
            <div
              key={l}
              style={{
                background: "rgba(255,255,255,.04)",
                border: "1px solid rgba(255,255,255,.08)",
                borderRadius: 10,
                padding: "12px 16px",
                flex: 1,
                minWidth: 110,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: "#8b9ab0",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  marginBottom: 3,
                }}
              >
                {l}
              </div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 800,
                  color: c,
                  fontFamily: "monospace",
                }}
              >
                {v}
              </div>
              <div style={{ fontSize: 10, color: "#475569" }}>{s}</div>
            </div>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={170}>
          <LineChart data={lossData}>
            <XAxis
              dataKey="epoch"
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip contentStyle={ttStyle} />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="🔄 ML Pipeline" span={2}>
        <div style={{ display: "flex", flexWrap: "wrap" }}>
          {[
            {
              i: "📋",
              s: "1. Policy Input",
              d: `"${policy.title}"`,
              c: "#64748b",
            },
            {
              i: "👥",
              s: "2. Sample 10",
              d: "3 low + 3 mid + 4 high",
              c: "#818cf8",
            },
            {
              i: "🤖",
              s: "3. Gemini LLM",
              d: "Generates reactions",
              c: "#6366f1",
            },
            {
              i: "🧠",
              s: "4. Train MLP",
              d: `${nnStats.samples} samples, Adam SGD`,
              c: "#10b981",
            },
            {
              i: "⚡",
              s: "5. Scale Up",
              d: "NN predicts all citizens",
              c: "#fbbf24",
            },
            {
              i: "📊",
              s: "6. Aggregate",
              d: "By income, zone, views",
              c: "#f87171",
            },
          ].map((x, idx, arr) => (
            <div
              key={x.s}
              style={{
                flex: 1,
                minWidth: 110,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                textAlign: "center",
                padding: "12px 8px",
                position: "relative",
              }}
            >
              {idx < arr.length - 1 && (
                <div
                  style={{
                    position: "absolute",
                    right: 0,
                    top: "50%",
                    width: 1,
                    height: "50%",
                    transform: "translateY(-50%)",
                    background: "rgba(255,255,255,.07)",
                  }}
                />
              )}
              <div
                style={{
                  width: 38,
                  height: 38,
                  borderRadius: "50%",
                  background: x.c + "22",
                  border: `2px solid ${x.c}55`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 17,
                  marginBottom: 7,
                }}
              >
                {x.i}
              </div>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: x.c,
                  marginBottom: 3,
                }}
              >
                {x.s}
              </div>
              <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.4 }}>
                {x.d}
              </div>
            </div>
          ))}
        </div>
      </ChartCard>
      <ChartCard title="📖 Gemini Diary Entries" span={2}>
        {(llmDiaries || []).slice(0, 5).map((d, i) => (
          <div
            key={i}
            style={{
              background: "rgba(255,255,255,.03)",
              borderRadius: 8,
              padding: "10px 14px",
              fontSize: 12,
              color: "#94a3b8",
              lineHeight: 1.6,
              borderLeft: "3px solid #6366f1",
              fontStyle: "italic",
              marginBottom: 8,
            }}
          >
            <span
              style={{ color: "#818cf8", fontWeight: 700, fontStyle: "normal" }}
            >
              #{d.id}:{" "}
            </span>
            "{d.diary}"
          </div>
        ))}
      </ChartCard>
    </div>
  );
}

function ExpertsTab({ policy, cur }) {
  const hap = cur.avg_happiness,
    sup = cur.avg_support,
    inc = cur.avg_income;
  const hl = hap > 0.6 ? "positive" : hap < 0.4 ? "concerning" : "mixed";
  const sl =
    sup > 0.15
      ? "broadly supportive"
      : sup < -0.15
      ? "largely opposed"
      : "divided";
  const low = cur.by_income.find((g) => g.income_level === "low"),
    high = cur.by_income.find((g) => g.income_level === "high");
  const gap =
    high && low
      ? ((high.avg_happiness - low.avg_happiness) * 100).toFixed(1)
      : "N/A";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {[
        {
          role: "📈 Economist",
          color: "#fbbf24",
          view: `The ${
            policy.title
          } shows ${hl} outcomes. Avg income $${Math.round(
            inc
          ).toLocaleString()}, population ${sl}. The ${Math.abs(
            +gap
          )}pp happiness gap is ${
            Math.abs(+gap) > 15
              ? "alarming — compensatory mechanisms needed"
              : "within acceptable bounds"
          }.`,
        },
        {
          role: "✊ Social Activist",
          color: "#f87171",
          view: `This intervention ${
            hap < 0.5 ? "deepens fractures" : "creates opportunity"
          }. ${pct(hap)} avg happiness, opinion ${sl}. ${
            low
              ? `Low-income at ${pct(low.avg_happiness)} vs high at ${pct(
                  high?.avg_happiness ?? 0
                )}.`
              : ""
          } ${
            +gap > 10
              ? "Immediate action needed."
              : "Cautious support with oversight."
          }`,
        },
        {
          role: "🏢 Business Owner",
          color: "#34d399",
          view: `Market conditions are ${hl}. Consumer income $${Math.round(
            inc
          ).toLocaleString()}, sentiment ${sl}. ${
            sup > 0
              ? "Growth expected across key sectors."
              : "Brace for reduced discretionary spending."
          } Inequality gap: ${gap}pp.`,
        },
      ].map((e) => (
        <div
          key={e.role}
          style={{
            background: `${e.color}0a`,
            border: `1px solid ${e.color}30`,
            borderRadius: 12,
            padding: 20,
          }}
        >
          <div
            style={{
              fontWeight: 800,
              fontSize: 15,
              color: e.color,
              marginBottom: 10,
            }}
          >
            {e.role}
          </div>
          <p
            style={{
              color: "#cbd5e1",
              fontSize: 13,
              lineHeight: 1.75,
              margin: 0,
            }}
          >
            {e.view}
          </p>
        </div>
      ))}
      <div
        style={{
          background: "rgba(255,255,255,.03)",
          borderRadius: 10,
          padding: 12,
          fontSize: 11,
          color: "#475569",
        }}
      >
        ⚠️ Synthetic simulation for educational purposes only.
      </div>
    </div>
  );
}
