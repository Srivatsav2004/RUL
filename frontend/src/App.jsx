// frontend/src/App.jsx
import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { Line } from "react-chartjs-2";
import "tailwindcss/tailwind.css";

const FEATURES = [
  "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi",
  "NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32",
  "T48","SmFan","SmLPC","SmHPC"
];

// default placeholders (sensible engineering-ish defaults)
const DEFAULTS = {
  T2: 518, T24: 642, T30: 1590, T50: 1400,
  P2: 14.6, P15: 21.6, P30: 553.0, Nf: 8000, Nc: 10000,
  epr: 8.4, Ps30: 2388, phi: 47.5,
  NRf: 2000, NRc: 6000, BPR: 5.0, farB: 0.01, htBleed: 392,
  Nf_dmd: 8000, PCNfR_dmd: 2000, W31: 100.0, W32: 0.0,
  T48: 2388, SmFan: 0.1, SmLPC: 0.1, SmHPC: 0.1
};

export default function App(){
  const [inputs, setInputs] = useState(() => {
    const init = {};
    FEATURES.forEach(f => init[f] = DEFAULTS[f] ?? 0);
    return init;
  });
  const [Xcycles, setXcycles] = useState(30);
  const [sigma, setSigma] = useState(20);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const onChange = (k, v) => setInputs(prev => ({...prev, [k]: parseFloat(v)}));

  const submit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const payload = { features: inputs, X_cycles: Number(Xcycles), sigma: Number(sigma) };
      const API = import.meta.env.VITE_API_URL || "";
      const resp = await axios.post(`${API}/predict`, payload);
      // const resp = await axios.post("https://rul-3tso.onrender.com/predict", payload);
      setResult(resp.data);
    } catch(err){
      console.error(err);
      alert("Prediction failed. Is backend running at /api/predict ?");
    } finally {
      setLoading(false);
    }
  };

  // Prepare small degradation-like visualization using predicted value
  const chartData = () => {
    if (!result) return null;
    const pred = result.predicted_RUL;
    // build a synthetic degradation curve: sensor value decays over cycles to failure
    const cycles = Array.from({length: 50}, (_,i) => i);
    const series = cycles.map(c => Math.max(0, 100 - (c / Math.max(pred / 25,1)) * 100));
    return {
      labels: cycles,
      datasets: [{
        label: "Synthetic Health Index",
        data: series,
        fill: true,
        tension: 0.3,
      }]
    };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-sky-900 text-gray-100 p-6">
      <div className="max-w-6xl mx-auto">
        <motion.header initial={{y:-20, opacity:0}} animate={{y:0, opacity:1}} className="mb-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-extrabold">RUL Predictor — Demo</h1>
            <div className="text-sm text-gray-300">Interactive engine RUL predictions</div>
          </div>
          <p className="text-gray-400 mt-2">Enter sensor values (Kaggle FD-style sensors). Powered by a FastAPI backend.</p>
        </motion.header>

        <main className="grid grid-cols-12 gap-6">
          <section className="col-span-5 bg-white/5 rounded-2xl p-5 shadow-lg">
            <h2 className="text-xl font-semibold mb-3">Input features</h2>
            <div className="space-y-4 max-h-[60vh] overflow-auto pr-2">
              {FEATURES.map(f => (
                <div key={f} className="flex items-center justify-between gap-4">
                  <label className="w-36 text-sm text-gray-200">{f}</label>
                  <input type="number" step="any" value={inputs[f]} onChange={(e)=>onChange(f, e.target.value)}
                         className="flex-1 bg-white/5 border border-white/10 rounded px-3 py-2"/>
                </div>
              ))}
            </div>

            <div className="mt-4 space-y-2">
              <div className="flex items-center gap-3">
                <label className="text-sm text-gray-200 w-36">X cycles</label>
                <input type="number" value={Xcycles} onChange={(e)=>setXcycles(e.target.value)} className="w-28 bg-white/5 rounded px-3 py-2"/>
                <label className="text-sm text-gray-200">σ (sigma)</label>
                <input type="number" value={sigma} onChange={(e)=>setSigma(e.target.value)} className="w-28 bg-white/5 rounded px-3 py-2"/>
              </div>
              <div className="flex gap-3 mt-2">
                <button onClick={submit} className="px-4 py-2 rounded bg-gradient-to-r from-indigo-500 to-sky-500 hover:scale-105 transition">
                  {loading ? "Predicting..." : "Predict RUL"}
                </button>
                <button onClick={()=>{ setInputs(Object.fromEntries(Object.entries(DEFAULTS).map(([k,v])=>[k,v])) ) }} className="px-3 py-2 rounded border border-white/10 text-sm">Reset to defaults</button>
              </div>
            </div>

            <footer className="mt-6 text-xs text-gray-400">
              Website built by "Srivatsav"
            </footer>
          </section>

          <section className="col-span-7 space-y-4">
            <motion.div initial={{opacity:0}} animate={{opacity:1}} className="bg-white/5 rounded-2xl p-5 shadow-lg">
              <h2 className="text-xl font-semibold">Prediction</h2>
              {!result && <div className="mt-4 text-gray-300">No prediction yet. Fill inputs and click <b>Predict RUL</b>.</div>}
              {result && (
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="p-4 bg-white/6 rounded">
                    <div className="text-sm text-gray-300">Predicted Remaining Useful Life</div>
                    <div className="text-4xl font-bold mt-2">{result.predicted_RUL.toFixed(2)} cycles</div>
                  </div>
                  <div className="p-4 bg-white/6 rounded">
                    <div className="text-sm text-gray-300">Probability of failure within {Xcycles} cycles</div>
                    <div className="text-4xl font-bold mt-2">{(result.probability_failure_within_X*100).toFixed(2)}%</div>
                  </div>
                  <div className="col-span-2 mt-2 p-4 bg-white/6 rounded">
                    <div className="text-sm text-gray-300">Used features (first 8 shown)</div>
                    <div className="flex gap-2 mt-2 flex-wrap">
                      {FEATURES.slice(0,8).map(f => <span key={f} className="bg-white/8 px-3 py-1 rounded text-sm">{f}: {inputs[f]}</span>)}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>

            <motion.div initial={{y:10, opacity:0}} animate={{y:0, opacity:1}} className="bg-white/5 rounded-2xl p-5 shadow-lg">
              <h2 className="text-xl font-semibold">Degradation preview</h2>
              <div className="mt-3">
                {result ? (
                  <Line data={chartData()} options={{responsive:true, plugins:{legend:{display:false}}}} />
                ) : (
                  <div className="text-gray-300">Prediction needed to show degradation preview.</div>
                )}
              </div>
            </motion.div>

            <motion.div initial={{y:10, opacity:0}} animate={{y:0, opacity:1}} className="bg-white/4 rounded-2xl p-5 shadow-lg">
              <h3 className="text-lg font-semibold">Notes</h3>
              <ul className="list-disc ml-5 mt-2 text-gray-300">
                <li>Model served from backend; ensure the FastAPI server is running at <code>/api/predict</code>.</li>
                <li>SHAP explainability will be added as a separate page when requested.</li>
                <li>This frontend is responsive and can be deployed to Vercel / Netlify.</li>
              </ul>
            </motion.div>

          </section>
        </main>

      </div>
    </div>
  );
}
