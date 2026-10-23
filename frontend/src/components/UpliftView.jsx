import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

const COLORS = {
  forest: "#0F6B6B", linear: "#4C9A9A", predictive: "#E0A100", random: "#8896A1", oracle: "#14212B",
};
const LABELS = {
  forest: "Causal forest (DML)", linear: "Linear DML", predictive: "Predictive baseline",
  random: "Random rollout", oracle: "Oracle (true effect)",
};
const FONT = { family: "Public Sans, system-ui, sans-serif", color: "#14212B" };
const BASE = { font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { t: 16, r: 16, b: 56, l: 64 } };
const CONFIG = { responsive: true, displaylogo: false };

function useChart(ref, traces, layout) {
  useEffect(() => {
    if (!ref.current || !traces) return;
    Plotly.react(ref.current, traces, layout, CONFIG);
  }, [ref, traces, layout]);
  useEffect(() => {
    const node = ref.current;
    return () => node && Plotly.purge(node);
  }, [ref]);
}

export default function UpliftView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [view, setView] = useState("ground_truth");
  const qiniRef = useRef(null);
  const decileRef = useRef(null);

  useEffect(() => {
    fetch("/week2_results.json")
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setData)
      .catch(() => setError("Results file not found. Run python backend/causal/uplift.py, then reload."));
  }, []);

  const curves = data?.curves[view];
  const qiniTraces = curves && Object.keys(LABELS).filter((k) => curves[k]).map((k) => ({
    x: curves.x, y: curves[k], type: "scatter", mode: "lines", name: LABELS[k],
    line: { color: COLORS[k], width: k === "forest" ? 3 : 2, dash: k === "random" || k === "oracle" ? "dash" : "solid" },
  }));
  const qiniLayout = {
    ...BASE,
    xaxis: { title: "Share of customers given a discount (best-ranked first)", tickformat: ".0%" },
    yaxis: { title: "Extra purchases caused by the discount" },
    legend: { orientation: "h", y: -0.28 },
  };
  useChart(qiniRef, qiniTraces, qiniLayout);

  const d = data?.deciles;
  const decileTraces = d && [
    { x: d.decile, y: d.estimated, type: "bar", name: "Model estimate", marker: { color: COLORS.forest } },
    { x: d.decile, y: d.true, type: "bar", name: "True effect (simulation)", marker: { color: COLORS.predictive } },
  ];
  const decileLayout = {
    ...BASE, barmode: "group",
    xaxis: { title: "Customer decile (1 = highest predicted effect)", dtick: 1 },
    yaxis: { title: "Change in purchase probability", tickformat: ".0%" },
    legend: { orientation: "h", y: -0.3 },
  };
  useChart(decileRef, decileTraces, decileLayout);

  if (error) return <section className="panel"><p className="errors" role="alert">{error}</p></section>;
  if (!data) return <section className="panel"><p className="hint">Loading results...</p></section>;

  const m = data.metrics;
  const top = m[view].extra_purchases_top30;
  const pct = (x) => `${(x * 100).toFixed(1)} pts`;

  return (
    <div className="results">
      <section className="panel">
        <h2>Does the causal model pick better customers?</h2>
        <p className="hint">
          Each line shows how many extra purchases the discount creates as you give it to more customers, starting with
          the ones the method ranks highest. A line above the grey "random" line is better than guessing.
        </p>

        <div className="seg" role="group" aria-label="Evaluation basis">
          <button className={view === "ground_truth" ? "on" : ""} onClick={() => setView("ground_truth")}>Ground truth</button>
          <button className={view === "observed" ? "on" : ""} onClick={() => setView("observed")}>Observed data</button>
        </div>
        <p className="note">
          {view === "ground_truth"
            ? "Uses the simulated true effect of each customer. This is only possible with synthetic data."
            : "Uses only discounts and purchases a real company would see, weighted to correct for biased targeting. It is noisier."}
        </p>

        <dl className="stats">
          <div><dt>Top 30% with causal forest</dt><dd>{top.forest.toFixed(0)}</dd></div>
          <div><dt>Top 30% with linear DML</dt><dd>{top.linear.toFixed(0)}</dd></div>
          <div><dt>Top 30% by purchase likelihood</dt><dd>{top.predictive.toFixed(0)}</dd></div>
          <div><dt>Top 30% at random</dt><dd>{top.random.toFixed(0)}</dd></div>
        </dl>
        <p className="note">Extra purchases from giving the discount to the best-ranked 30% of {m.n_test.toLocaleString()} test customers.</p>

        <div ref={qiniRef} className="chart" role="img" aria-label="Qini curves comparing targeting methods" />
      </section>

      <section className="panel">
        <h2>Estimated vs true effect</h2>
        <p className="hint">
          Customers are sorted by predicted effect and split into ten equal groups. The bars should fall from left to
          right, and the model estimate should track the true effect.
        </p>
        <dl className="stats">
          <div><dt>Average effect, estimated</dt><dd>{pct(m.linear_ate)}</dd></div>
          <div><dt>Average effect, true</dt><dd>{pct(m.true_ate)}</dd></div>
          <div><dt>Correlation with true effect</dt><dd>{m.linear_corr.toFixed(2)}</dd></div>
        </dl>
        <div ref={decileRef} className="chart" role="img" aria-label="Estimated versus true effect by decile" />
      </section>
    </div>
  );
}
