import { useState } from "react";
import UploadView from "./components/UploadView.jsx";
import BudgetView from "./components/BudgetView.jsx";
import UpliftView from "./components/UpliftView.jsx";

const DEFAULT_BUDGET = { total: 5000, tiers: [10, 20], maxPerCustomer: 20 };

export default function App() {
  const [tab, setTab] = useState("setup"); // "setup" (Week 1) | "results" (Week 2)
  const [dataset, setDataset] = useState(null); // { fileName, columns, records, stats }
  const [budget, setBudget] = useState(DEFAULT_BUDGET);

  const ready = Boolean(dataset && !dataset.errors?.length);

  return (
    <div className="shell">
      <header className="top">
        <h1>EconoCausal</h1>
        <p>Decide who gets a discount by measuring who it actually changes.</p>
        <nav className="tabs" aria-label="Sections">
          <button className={tab === "setup" ? "on" : ""} onClick={() => setTab("setup")}>Setup</button>
          <button className={tab === "results" ? "on" : ""} onClick={() => setTab("results")}>Uplift results</button>
        </nav>
      </header>

      {tab === "setup" ? (
        <main className="grid">
          <UploadView dataset={dataset} onLoaded={setDataset} />
          <BudgetView budget={budget} onChange={setBudget} dataset={ready ? dataset : null} />
        </main>
      ) : (
        <main>
          <UpliftView />
        </main>
      )}

      {tab === "setup" && (
        <footer className="bar">
          <span className={ready ? "state ok" : "state"}>
            {ready ? "Data loaded and budget set." : "Upload campaign data to continue."}
          </span>
          <button className="primary" disabled={!ready} onClick={() => setTab("results")}>
            View discount impact
          </button>
        </footer>
      )}
    </div>
  );
}
