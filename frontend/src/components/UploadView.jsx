import { useRef, useState } from "react";
import { parseCSV, validateCampaign, REQUIRED_COLUMNS } from "../csv.js";

const pct = (x) => `${(x * 100).toFixed(1)}%`;

export default function UploadView({ dataset, onLoaded }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleText = (text, fileName) => {
    const { columns, records } = parseCSV(text);
    const { errors, stats } = validateCampaign(columns, records);
    onLoaded({ fileName, columns, records, errors, stats });
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      onLoaded({ fileName: file.name, columns: [], records: [], errors: ["Upload a .csv file."], stats: null });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => handleText(String(reader.result), file.name);
    reader.readAsText(file);
  };

  const loadSample = async () => {
    const res = await fetch("/sample_campaign.csv");
    handleText(await res.text(), "sample_campaign.csv");
  };

  const stats = dataset?.stats;

  return (
    <section className="panel" aria-labelledby="upload-title">
      <h2 id="upload-title">Campaign history</h2>
      <p className="hint">
        One row per customer. Required columns: <code>{REQUIRED_COLUMNS.join(", ")}</code>, plus customer features.
      </p>

      <div
        className={`drop ${dragging ? "over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
      >
        <p>Drop a CSV here</p>
        <div className="row">
          <button onClick={() => inputRef.current?.click()}>Choose file</button>
          <button className="ghost" onClick={loadSample}>Use sample data</button>
        </div>
        <input ref={inputRef} type="file" accept=".csv" hidden onChange={(e) => handleFile(e.target.files[0])} />
      </div>

      {dataset?.errors?.length > 0 && (
        <ul className="errors" role="alert">
          {dataset.errors.map((e) => <li key={e}>{e}</li>)}
        </ul>
      )}

      {stats && (
        <>
          <p className="file">{dataset.fileName}</p>
          <dl className="stats">
            <div><dt>Customers</dt><dd>{stats.rows.toLocaleString()}</dd></div>
            <div><dt>Features</dt><dd>{stats.features}</dd></div>
            <div><dt>Got discount</dt><dd>{stats.treated.toLocaleString()}</dd></div>
            <div><dt>No discount</dt><dd>{stats.control.toLocaleString()}</dd></div>
          </dl>
          <p className="note">
            Purchase rate: {pct(stats.treatedRate)} with discount vs {pct(stats.controlRate)} without. This gap mixes
            the discount's effect with who was targeted, so it is not the true impact.
          </p>

          <div className="scroll">
            <table>
              <thead><tr>{dataset.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {dataset.records.slice(0, 5).map((r, i) => (
                  <tr key={i}>{dataset.columns.map((c) => <td key={c}>{r[c]}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
