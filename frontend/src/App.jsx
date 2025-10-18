import { useState } from "react";
import axios from "axios";

export default function App() {
  const [inputs, setInputs] = useState({});
  const [Xcycles, setXcycles] = useState(30);
  const [sigma, setSigma] = useState(20);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // handle input for features
  const handleChange = (e) => {
    const { name, value } = e.target;
    setInputs((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // call backend
  const submit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const payload = { features: inputs, X_cycles: Number(Xcycles), sigma: Number(sigma) };
      const API = import.meta.env.VITE_API_URL || "";
      const resp = await axios.post(`${API}/predict`, payload);
      setResult(resp.data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Check if backend is running or URL is correct.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center py-8 px-4">
      <h1 className="text-3xl font-bold mb-4 text-center">Turbofan RUL Prediction</h1>

      {/* Input fields */}
      <div className="w-full max-w-3xl bg-gray-800 p-6 rounded-lg shadow-lg">
        <h2 className="text-lg font-semibold mb-2">Enter Engine Parameters</h2>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[
            "T2","T24","T30","T50","P2","P15","P30","Nf","Nc","epr","Ps30","phi",
            "NRf","NRc","BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32",
            "T48","SmFan","SmLPC","SmHPC"
          ].map((f) => (
            <input
              key={f}
              name={f}
              type="number"
              step="any"
              placeholder={f}
              value={inputs[f] || ""}
              onChange={handleChange}
              className="bg-gray-700 rounded px-3 py-2 focus:outline-none text-sm"
            />
          ))}
        </div>

        <div className="flex gap-4 mt-4">
          <input
            type="number"
            value={Xcycles}
            onChange={(e) => setXcycles(e.target.value)}
            placeholder="X cycles (default 30)"
            className="bg-gray-700 rounded px-3 py-2 w-1/2"
          />
          <input
            type="number"
            value={sigma}
            onChange={(e) => setSigma(e.target.value)}
            placeholder="Sigma (default 20)"
            className="bg-gray-700 rounded px-3 py-2 w-1/2"
          />
        </div>

        <button
          onClick={submit}
          disabled={loading}
          className="w-full mt-6 bg-blue-500 hover:bg-blue-600 transition rounded py-2 font-semibold"
        >
          {loading ? "Predicting..." : "Predict RUL"}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="w-full max-w-4xl mt-8 space-y-4">

          {/* Model Predictions Table */}
          <div className="bg-gray-800 p-4 rounded-lg">
            <h3 className="text-xl font-semibold mb-2">Model Predictions</h3>
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="py-2">Model</th>
                  <th>Predicted RUL</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.model_predictions).map(([name, val]) => (
                  <tr key={name} className="border-b border-gray-700">
                    <td className="py-1">{name}</td>
                    <td>{val?.toFixed ? val.toFixed(2) : val}</td>
                    <td>
                      {result.highest_accuracy_model?.name === name
                        ? `${result.highest_accuracy_model.accuracy} ⭐`
                        : result.lowest_accuracy_model?.name === name
                        ? `${result.lowest_accuracy_model.accuracy} ⚠️`
                        : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Weighted Accuracy Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-gray-400 text-sm">Final Weighted Accuracy</div>
              <div className="text-2xl font-bold text-blue-400 mt-1">
                {(result.final_weighted_accuracy * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-gray-400 text-sm">Highest Accuracy Model</div>
              <div className="text-green-400 font-semibold mt-1">
                {result.highest_accuracy_model.name} ({result.highest_accuracy_model.accuracy})
              </div>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-gray-400 text-sm">Lowest Accuracy Model</div>
              <div className="text-red-400 font-semibold mt-1">
                {result.lowest_accuracy_model.name} ({result.lowest_accuracy_model.accuracy})
              </div>
            </div>
          </div>

          {/* Failure Probability */}
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-sm text-gray-300">
              Probability of failure within {Xcycles} cycles:
            </div>
            <div className="text-3xl font-bold mt-2 text-yellow-400">
              {(result.probability_failure_within_X * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
