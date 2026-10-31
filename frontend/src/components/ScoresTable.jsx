import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

// Columns shown in the table: key, label, number of decimals (null = whole number)
const COLUMNS = [
  { key: "customer_id", label: "Customer", dp: null },
  { key: "past_spend", label: "Past spend", dp: 1 },
  { key: "tenure_months", label: "Tenure (months)", dp: null },
  { key: "visits_per_month", label: "Visits / month", dp: null },
  { key: "price_sensitivity", label: "Price sensitivity", dp: 2 },
  { key: "region", label: "Region", dp: null },
  { key: "ite_forest", label: "Effect (forest)", dp: 3 },
  { key: "ite_linear", label: "Effect (linear)", dp: 3 },
];
const PAGE_SIZE = 25;
const FONT = { family: "Public Sans, system-ui, sans-serif", color: "#14212B" };

export default function ScoresTable() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");
  const [sortKey, setSortKey] = useState("ite_forest");
  const [sortDir, setSortDir] = useState("desc");
  const [minEffect, setMinEffect] = useState(null); // null = no lower limit yet
  const [region, setRegion] = useState("all");
  const [page, setPage] = useState(0);
  const histRef = useRef(null);

  useEffect(() => {
    fetch("/ite_scores.json")
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setRows)
      .catch(() => setError("Scores file not found. Run python backend/causal/export_scores.py, then reload."));
  }, []);

  // Lowest and highest effect in the data, used for the slider range
  const bounds = useMemo(() => {
    if (!rows) return { lo: 0, hi: 1 };
    const values = rows.map((r) => r.ite_forest);
    return { lo: Math.floor(Math.min(...values) * 100) / 100, hi: Math.ceil(Math.max(...values) * 100) / 100 };
  }, [rows]);
  const threshold = minEffect ?? bounds.lo;

  // 1. filter  2. sort  (recomputed only when an input changes, so 1,500+ rows stay smooth)
  const filtered = useMemo(() => {
    if (!rows) return [];
    return rows.filter((r) => r.ite_forest >= threshold && (region === "all" || r.region === Number(region)));
  }, [rows, threshold, region]);

  const sorted = useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * dir);
  }, [filtered, sortKey, sortDir]);

  const average = useMemo(
    () => (filtered.length ? filtered.reduce((s, r) => s + r.ite_forest, 0) / filtered.length : 0),
    [filtered]
  );

  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visible = sorted.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  // Histogram of the filtered customers' effects
  useEffect(() => {
    if (!histRef.current || !rows) return;
    Plotly.react(
      histRef.current,
      [{ x: filtered.map((r) => r.ite_forest), type: "histogram", nbinsx: 30, marker: { color: "#0F6B6B" } }],
      {
        font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 10, r: 10, b: 50, l: 56 },
        xaxis: { title: "Estimated effect of the discount on purchase probability", tickformat: ".0%" },
        yaxis: { title: "Customers" },
        bargap: 0.05,
      },
      { responsive: true, displaylogo: false }
    );
  }, [filtered, rows]);
  useEffect(() => {
    const node = histRef.current;
    return () => node && Plotly.purge(node);
  }, []);

  const changeSort = (key) => {
    if (key === sortKey) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
    setPage(0);
  };

  if (error) return <section className="panel"><p className="errors" role="alert">{error}</p></section>;
  if (!rows) return <section className="panel"><p className="hint">Loading customer scores...</p></section>;

  const fmt = (value, dp) => (dp === null ? value : Number(value).toFixed(dp));

  return (
    <div className="results">
      <section className="panel">
        <h2>Customer scores</h2>
        <p className="hint">
          Each row is a customer with the model's estimate of how much a discount raises their chance of buying.
          Filter and sort to find the customers worth targeting.
        </p>

        <div className="filters">
          <label>
            Minimum effect: {(threshold * 100).toFixed(0)}%
            <input
              type="range" min={bounds.lo} max={bounds.hi} step="0.01" value={threshold}
              onChange={(e) => { setMinEffect(Number(e.target.value)); setPage(0); }}
            />
          </label>
          <label>
            Region
            <select value={region} onChange={(e) => { setRegion(e.target.value); setPage(0); }}>
              <option value="all">All regions</option>
              {[0, 1, 2, 3].map((r) => <option key={r} value={r}>Region {r}</option>)}
            </select>
          </label>
        </div>

        <dl className="stats">
          <div><dt>Customers loaded</dt><dd>{rows.length.toLocaleString()}</dd></div>
          <div><dt>Matching filters</dt><dd>{filtered.length.toLocaleString()}</dd></div>
          <div><dt>Average effect of matches</dt><dd>{(average * 100).toFixed(1)} pts</dd></div>
        </dl>

        <div ref={histRef} className="chart short" role="img" aria-label="Histogram of estimated discount effects" />
      </section>

      <section className="panel">
        <div className="scroll">
          <table>
            <thead>
              <tr>
                {COLUMNS.map((c) => (
                  <th key={c.key} aria-sort={sortKey === c.key ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
                    <button className="th" onClick={() => changeSort(c.key)}>
                      {c.label}{sortKey === c.key ? (sortDir === "asc" ? " \u25B2" : " \u25BC") : ""}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.customer_id}>{COLUMNS.map((c) => <td key={c.key}>{fmt(r[c.key], c.dp)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="pager">
          <button className="ghost" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>Previous</button>
          <span>Page {safePage + 1} of {pageCount}</span>
          <button className="ghost" disabled={safePage >= pageCount - 1} onClick={() => setPage(safePage + 1)}>Next</button>
        </div>
      </section>
    </div>
  );
}
