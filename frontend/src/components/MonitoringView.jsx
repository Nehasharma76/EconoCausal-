import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { PSI_ALERT, PSI_WARN, detectDrift } from "../psi.js";

const FONT = { family: "Public Sans, system-ui, sans-serif", color: "#14212B" };
const COLOR = { ok: "#0F6B6B", warning: "#E0A100", alert: "#A4262C" };
const HEADLINE = {
  ok: "Stable. Customers look like the data the model was trained on.",
  warning: "Warning. Some features have shifted, so effect estimates may be less reliable. Keep monitoring.",
  alert: "Alert. Customer behaviour has changed a lot. Retrain the model before trusting these estimates.",
};

export default function MonitoringView({ dataset }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  const [choice, setChoice] = useState("");
  const chartRef = useRef(null);

  useEffect(() => {
    fetch("/week4_drift.json")
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setInfo)
      .catch(() => setError("Drift file not found. Run python backend/monitoring/simulate_drift.py, then reload."));
  }, []);

  const uploaded = dataset && !dataset.errors?.length ? dataset : null;
  const options = useMemo(() => (info ? [...(uploaded ? ["Uploaded data"] : []), ...Object.keys(info.scenarios)] : []), [info, uploaded]);
  const selected = options.includes(choice) ? choice : options[0];

  const report = useMemo(() => {
    if (!info || !selected) return null;
    if (selected === "Uploaded data") return detectDrift(info.reference, info.features, uploaded.records);
    return info.scenarios[selected];
  }, [info, selected, uploaded]);

  useEffect(() => {
    if (!chartRef.current || !report || report.error) return;
    const feats = report.features;
    Plotly.react(
      chartRef.current,
      [{ x: feats.map((f) => f.feature), y: feats.map((f) => f.psi), type: "bar", marker: { color: feats.map((f) => COLOR[f.status]) } }],
      {
        font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { t: 10, r: 10, b: 60, l: 60 },
        yaxis: { title: "PSI (higher = more drift)", rangemode: "tozero" },
        shapes: [PSI_WARN, PSI_ALERT].map((y, i) => ({
          type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
          line: { color: i ? "#14212B" : "#8896A1", width: 1, dash: "dash" },
        })),
        annotations: [
          { xref: "paper", x: 1, y: PSI_WARN, text: "warning", showarrow: false, xanchor: "right", yshift: 8, font: { size: 10 } },
          { xref: "paper", x: 1, y: PSI_ALERT, text: "alert", showarrow: false, xanchor: "right", yshift: 8, font: { size: 10 } },
        ],
      },
      { responsive: true, displaylogo: false }
    );
  }, [report]);
  useEffect(() => { const n = chartRef.current; return () => n && Plotly.purge(n); }, []);

  if (error) return <section className="panel"><p className="errors" role="alert">{error}</p></section>;
  if (!info) return <section className="panel"><p className="hint">Loading monitoring data...</p></section>;

  return (
    <div className="results">
      <section className="panel">
        <h2>Is the data still like the training data?</h2>
        <p className="hint">
          A causal model is only reliable while new customers resemble the ones it learned from. The drift score (PSI)
          compares each customer feature with the training data. Below {PSI_WARN} is stable, {PSI_WARN} to {PSI_ALERT} is a
          warning, and above {PSI_ALERT} is an alert.
        </p>

        <label>
          Batch to check
          <select value={selected} onChange={(e) => setChoice(e.target.value)}>
            {options.map((o) => <option key={o}>{o}</option>)}
          </select>
        </label>
        {!uploaded && <p className="note">Upload a campaign CSV on the Setup tab to check your own data here.</p>}

        {report?.error && <p className="errors" role="alert">{report.error}</p>}
        {report && !report.error && (
          <>
            <p className={`banner ${report.overall}`} role="status">{HEADLINE[report.overall]}</p>
            {report.notes?.map((n) => <p className="note" key={n}>{n}</p>)}
            <div ref={chartRef} className="chart short" role="img" aria-label="Drift score for each customer feature" />
            <div className="scroll">
              <table>
                <thead><tr><th>Feature</th><th>PSI</th><th>Status</th><th>Average shift (std devs)</th></tr></thead>
                <tbody>
                  {report.features.map((f) => (
                    <tr key={f.feature}>
                      <td>{f.feature}</td>
                      <td>{f.psi.toFixed(3)}</td>
                      <td><span className={`badge ${f.status}`}>{f.status}</span></td>
                      <td>{f.mean_shift_std == null && f.shift == null ? "-" : `${(f.mean_shift_std ?? f.shift) >= 0 ? "+" : ""}${(f.mean_shift_std ?? f.shift).toFixed(2)}`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
