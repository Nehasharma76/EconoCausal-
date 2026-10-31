// Drift detection in the browser. Mirrors backend/monitoring/drift.py so both give the same numbers.
// Population Stability Index (PSI): sum over bins of (actual - expected) * ln(actual / expected).

export const PSI_WARN = 0.1;
export const PSI_ALERT = 0.25;
export const MIN_ROWS = 500;
const EPS = 1e-4;

const clip = (p) => Math.max(p, EPS);
const psiValue = (expected, actual) =>
  expected.reduce((sum, e, i) => sum + (clip(actual[i]) - clip(e)) * Math.log(clip(actual[i]) / clip(e)), 0);

export const statusFor = (v) => (v >= PSI_ALERT ? "alert" : v >= PSI_WARN ? "warning" : "ok");

// Same as numpy.digitize(x, edges): number of edges that are <= x
const binIndex = (x, edges) => edges.filter((e) => e <= x).length;

function featureReport(ref, values) {
  const n = values.length;
  let expected, actual, shift = null;
  if (ref.type === "categorical") {
    actual = ref.values.map((v) => values.filter((x) => x === v).length / n);
    actual.push(values.filter((x) => !ref.values.includes(x)).length / n); // categories never seen in training
    expected = [...ref.proportions, 0];
  } else {
    const counts = new Array(ref.edges.length + 1).fill(0);
    values.forEach((x) => { counts[binIndex(x, ref.edges)] += 1; });
    actual = counts.map((c) => c / n);
    expected = ref.proportions;
    const mean = values.reduce((s, x) => s + x, 0) / n;
    shift = ref.std ? (mean - ref.mean) / ref.std : 0;
  }
  const psi = psiValue(expected, actual);
  return { psi, status: statusFor(psi), shift };
}

// records: array of objects (values may be strings, as read from a CSV)
export function detectDrift(reference, features, records) {
  const missing = features.filter((f) => !records.length || !(f in records[0]));
  if (missing.length) return { error: `The data is missing column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}.` };

  const rows = features.map((f) => {
    const values = records.map((r) => Number(r[f])).filter((x) => Number.isFinite(x));
    return { feature: f, ...featureReport(reference[f], values) };
  });
  const rank = { ok: 0, warning: 1, alert: 2 };
  const overall = rows.reduce((worst, r) => (rank[r.status] > rank[worst] ? r.status : worst), "ok");
  const notes = records.length < MIN_ROWS
    ? [`Only ${records.length} rows: PSI is noisy below ${MIN_ROWS} rows, so treat this as a rough guide.`] : [];
  return { n_rows: records.length, overall, features: rows, notes };
}
