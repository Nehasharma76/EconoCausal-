import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

const FONT = { family: "Public Sans, system-ui, sans-serif", color: "#14212B" };
const money = (x) => `${x < 0 ? "-" : ""}$${Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const PAGE_SIZE = 25;

// Plain-English summary of a result (works for saved results and live API results)
export function describe(result) {
  const { optimizer: o, blanket: b } = result.strategies;
  const better = o.net_revenue - b.net_revenue;
  return (
    `With a ${money(result.budget)} budget, this strategy gives discounts to ${o.customers_discounted.toLocaleString()} ` +
    `customers, spends ${money(o.spend)}, and is expected to add ${money(o.net_revenue)} in net revenue. ` +
    (better >= 0
      ? `That is ${money(better)} better than giving a blanket $20 discount to the same number of dollars of customers.`
      : `A blanket $20 rollout would do ${money(-better)} better on these estimates.`)
  );
}

export default function AllocationView({ budget: setupBudget }) {
  const [saved, setSaved] = useState(null);
  const [live, setLive] = useState(null); // result from the API, if the user recalculated
  const [apiUp, setApiUp] = useState(false);
  const [budgetInput, setBudgetInput] = useState(5000);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [tierFilter, setTierFilter] = useState("all");
  const [sortKey, setSortKey] = useState("expected_net_gain");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);
  const sweepRef = useRef(null);

  useEffect(() => {
    fetch("/week3_prescription.json")
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((d) => { setSaved(d); setBudgetInput(d.budget); })
      .catch(() => setError("Results file not found. Run python backend/optimize/prescribe.py, then reload."));
    fetch("/api/health").then((r) => setApiUp(r.ok)).catch(() => setApiUp(false));
  }, []);

  const result = live ?? saved;

  const recalculate = async () => {
    setBusy(true); setMessage("");
    try {
      const customers = await (await fetch("/ite_scores.json")).json();
      const tiers = setupBudget.tiers.filter((t) => t <= setupBudget.maxPerCustomer);
      const res = await fetch("/api/allocate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          budget: Number(budgetInput), tiers: tiers.length ? tiers : undefined,
          customers: customers.map((c) => ({
            customer_id: c.customer_id, past_spend: c.past_spend, tenure_months: c.tenure_months,
            visits_per_month: c.visits_per_month, price_sensitivity: c.price_sensitivity, region: c.region,
          })),
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? "The API returned an error.");
      setLive(await res.json()); setPage(0);
    } catch (e) {
      setMessage(`Could not recalculate: ${e.message}`);
    } finally { setBusy(false); }
  };

  const rows = useMemo(() => {
    if (!result) return [];
    const filtered = result.rows.filter((r) => tierFilter === "all" || String(r.discount) === tierFilter);
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * dir);
  }, [result, tierFilter, sortKey, sortDir]);

  // Chart: net revenue against budget (saved results only, because it uses the simulated truth)
  useEffect(() => {
    if (!sweepRef.current || !saved) return;
    const s = saved.sweep;
    const line = (name, label, color, width) => ({ x: s.budgets, y: s[name], type: "scatter", mode: "lines+markers", name: label, line: { color, width } });
    Plotly.react(
      sweepRef.current,
      [line("optimizer", "Optimizer", "#0F6B6B", 3), line("blanket", "Blanket $20 rollout", "#8896A1", 2), line("likely_buyers", "Likely buyers", "#E0A100", 2)],
      {
        font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { t: 10, r: 10, b: 60, l: 70 },
        xaxis: { title: "Discount budget ($)" }, yaxis: { title: "Net revenue gain ($)", zeroline: true },
        legend: { orientation: "h", y: -0.3 },
      },
      { responsive: true, displaylogo: false }
    );
  }, [saved]);
  useEffect(() => { const n = sweepRef.current; return () => n && Plotly.purge(n); }, []);

  if (error) return <section className="panel"><p className="errors" role="alert">{error}</p></section>;
  if (!result) return <section className="panel"><p className="hint">Loading allocation...</p></section>;

  const o = result.strategies.optimizer, b = result.strategies.blanket, l = result.strategies.likely_buyers;
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const visible = rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);
  const changeSort = (key) => {
    if (key === sortKey) setSortDir(sortDir === "asc" ? "desc" : "asc"); else { setSortKey(key); setSortDir("desc"); }
    setPage(0);
  };
  const cols = [
    ["customer_id", "Customer"], ["discount", "Discount"], ["effect", "Effect of a full discount"],
    ["baseline", "Would buy anyway"], ["order_value", "Order value"], ["expected_net_gain", "Expected net gain"],
  ];
  const trueRes = !live && saved.strategies_true;

  return (
    <div className="results">
      <section className="panel">
        <h2>Who gets which discount</h2>
        <p className="summary">{describe(result)}</p>

        <div className="filters budgetbar">
          <label>
            Budget ($)
            <input type="number" min="100" step="100" value={budgetInput} onChange={(e) => setBudgetInput(e.target.value)} />
          </label>
          <div>
            <button onClick={recalculate} disabled={!apiUp || busy}>{busy ? "Calculating..." : "Recalculate"}</button>
            {live && <button className="ghost" onClick={() => { setLive(null); setBudgetInput(saved.budget); setPage(0); }}>Back to saved results</button>}
          </div>
        </div>
        <p className="note">
          {apiUp
            ? "The API is running, so you can try any budget."
            : "Showing saved results for a $5,000 budget. Start the API (uvicorn backend.api.main:app --port 8000) to try other budgets."}
        </p>
        {message && <p className="errors" role="alert">{message}</p>}

        <dl className="stats">
          <div><dt>Customers discounted</dt><dd>{o.customers_discounted.toLocaleString()}</dd></div>
          <div><dt>Budget spent</dt><dd>{money(o.spend)}</dd></div>
          <div><dt>Expected net revenue</dt><dd>{money(o.net_revenue)}</dd></div>
          <div><dt>Blanket $20 rollout</dt><dd>{money(b.net_revenue)}</dd></div>
          <div><dt>Likely-buyers targeting</dt><dd>{money(l.net_revenue)}</dd></div>
        </dl>
        <p className="note">
          Net revenue counts the extra sales a discount causes, minus the money given away to customers who would have
          bought anyway. The optimizer skips customers whose discount would cost more than it earns, so it can spend less than the full budget.
        </p>

        {trueRes && (
          <div className="callout">
            <strong>Check against the simulated truth.</strong> The estimates above are optimistic because the optimizer
            picks the customers the model rates highest. Using the true effects, the optimizer earns {money(trueRes.optimizer.net_revenue)},
            the blanket rollout {money(trueRes.blanket.net_revenue)}, and likely-buyers targeting {money(trueRes.likely_buyers.net_revenue)}.
          </div>
        )}
      </section>

      {saved && (
        <section className="panel">
          <h2>Net revenue against budget</h2>
          <p className="hint">Evaluated with the simulated true effects. The optimizer stops gaining once every worthwhile customer has a discount.</p>
          <div ref={sweepRef} className="chart" role="img" aria-label="Net revenue gain against budget for three strategies" />
        </section>
      )}

      <section className="panel">
        <h2>Allocation matrix</h2>
        <div className="filters">
          <label>
            Show
            <select value={tierFilter} onChange={(e) => { setTierFilter(e.target.value); setPage(0); }}>
              <option value="all">All customers</option>
              {result.tiers.map((t) => <option key={t} value={String(t)}>Discount ${t}</option>)}
              <option value="0">No discount</option>
            </select>
          </label>
          <p className="note">{rows.length.toLocaleString()} customers shown</p>
        </div>

        <div className="scroll">
          <table>
            <thead>
              <tr>{cols.map(([k, label]) => (
                <th key={k} aria-sort={sortKey === k ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
                  <button className="th" onClick={() => changeSort(k)}>{label}{sortKey === k ? (sortDir === "asc" ? " \u25B2" : " \u25BC") : ""}</button>
                </th>
              ))}</tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.customer_id}>
                  <td>{r.customer_id}</td>
                  <td>{r.discount > 0 ? `$${r.discount}` : "None"}</td>
                  <td>{(r.effect * 100).toFixed(1)} pts</td>
                  <td>{(r.baseline * 100).toFixed(0)}%</td>
                  <td>{money(r.order_value)}</td>
                  <td>{r.discount > 0 ? money(r.expected_net_gain) : "-"}</td>
                </tr>
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
