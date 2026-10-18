// Minimal CSV parser (handles quoted fields). Enough for campaign uploads.
export function parseCSV(text) {
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') inQuotes = false;
      else field += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((c) => c !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field !== "" || row.length) { row.push(field); if (row.some((c) => c !== "")) rows.push(row); }
  if (!rows.length) return { columns: [], records: [] };
  const [columns, ...body] = rows;
  const records = body.map((r) => Object.fromEntries(columns.map((c, i) => [c.trim(), r[i] ?? ""])));
  return { columns: columns.map((c) => c.trim()), records };
}

export const REQUIRED_COLUMNS = ["customer_id", "discount_given", "purchased"];

// Returns { errors: string[], stats: {...} | null }
export function validateCampaign(columns, records) {
  const errors = [];
  const missing = REQUIRED_COLUMNS.filter((c) => !columns.includes(c));
  if (missing.length) errors.push(`Missing required column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}.`);
  if (columns.length < 4) errors.push("Add at least one customer feature column (for example past_spend).");
  if (!records.length) errors.push("The file has no data rows.");
  if (errors.length) return { errors, stats: null };

  const bad = records.filter(
    (r) => !["0", "1"].includes(String(r.discount_given).trim()) || !["0", "1"].includes(String(r.purchased).trim())
  ).length;
  if (bad) errors.push(`${bad} row${bad > 1 ? "s have" : " has"} a discount_given or purchased value that is not 0 or 1.`);

  const treated = records.filter((r) => r.discount_given.trim() === "1");
  const control = records.filter((r) => r.discount_given.trim() === "0");
  const rate = (rs) => (rs.length ? rs.filter((r) => r.purchased.trim() === "1").length / rs.length : 0);
  return {
    errors,
    stats: {
      rows: records.length,
      features: columns.length - REQUIRED_COLUMNS.length,
      treated: treated.length,
      control: control.length,
      treatedRate: rate(treated),
      controlRate: rate(control),
    },
  };
}
