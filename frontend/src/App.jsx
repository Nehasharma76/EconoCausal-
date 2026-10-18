import { useState } from "react";
import UploadView from "./components/UploadView.jsx";
import BudgetView from "./components/BudgetView.jsx";

const DEFAULT_BUDGET = { total: 5000, tiers: [10, 20], maxPerCustomer: 20 };

export default function App() {
  const [dataset, setDataset] = useState(null); // { fileName, columns, records, stats }
  const [budget, setBudget] = useState(DEFAULT_BUDGET);

  const ready = Boolean(dataset && !dataset.errors?.length);

  return (
    <div className="shell">
      <header className="top">
        <h1>EconoCausal</h1>
        <p>Decide who gets a discount by measuring who it actually changes.</p>
      </header>

      <main className="grid">
        <UploadView dataset={dataset} onLoaded={setDataset} />
        <BudgetView budget={budget} onChange={setBudget} dataset={ready ? dataset : null} />
      </main>

      <footer className="bar">
        <span className={ready ? "state ok" : "state"}>
          {ready ? "Data loaded and budget set." : "Upload campaign data to continue."}
        </span>
        <button className="primary" disabled={!ready} title="Available in Week 2">
          Estimate discount impact
        </button>
      </footer>
    </div>
  );
}
