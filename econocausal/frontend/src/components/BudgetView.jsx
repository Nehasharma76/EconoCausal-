const fmt = (n) => n.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export default function BudgetView({ budget, onChange, dataset }) {
  const set = (patch) => onChange({ ...budget, ...patch });

  const setTierText = (text) => {
    const tiers = text
      .split(",")
      .map((t) => Number(t.trim()))
      .filter((n) => Number.isFinite(n) && n > 0);
    set({ tiers });
  };

  const customers = dataset?.stats.rows ?? 0;
  const invalidCap = budget.tiers.length && budget.maxPerCustomer < Math.min(...budget.tiers);
  // Upper bound on customers reachable if everyone received the smallest tier
  const reachable = budget.tiers.length ? Math.floor(budget.total / Math.min(...budget.tiers)) : 0;

  return (
    <section className="panel" aria-labelledby="budget-title">
      <h2 id="budget-title">Budget</h2>
      <p className="hint">The optimizer in Week 3 will spend within these limits.</p>

      <label>
        Total discount budget (USD)
        <input
          type="number" min="0" step="100" value={budget.total}
          onChange={(e) => set({ total: Math.max(0, Number(e.target.value)) })}
        />
      </label>

      <label>
        Discount options in USD, separated by commas
        <input type="text" defaultValue={budget.tiers.join(", ")} onChange={(e) => setTierText(e.target.value)} />
      </label>

      <label>
        Highest discount for one customer (USD)
        <input
          type="number" min="0" value={budget.maxPerCustomer}
          onChange={(e) => set({ maxPerCustomer: Math.max(0, Number(e.target.value)) })}
        />
      </label>

      {invalidCap && <p className="errors" role="alert">The highest discount is below your smallest discount option.</p>}
      {!budget.tiers.length && <p className="errors" role="alert">Enter at least one discount option, for example 10, 20.</p>}

      <dl className="stats">
        <div><dt>Budget</dt><dd>{fmt(budget.total)}</dd></div>
        <div><dt>Smallest discount reaches up to</dt><dd>{reachable.toLocaleString()} customers</dd></div>
        {dataset && (
          <div><dt>Share of loaded customers</dt><dd>{customers ? `${Math.min(100, (reachable / customers) * 100).toFixed(0)}%` : "-"}</dd></div>
        )}
      </dl>
    </section>
  );
}
