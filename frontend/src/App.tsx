import { useState } from 'react';
import './index.css';

interface ModelResult {
  prediction: number;
  probability: number;
  accuracy: number;
}

interface PredictionResponse {
  xgboost: ModelResult;
  ft_transformer: ModelResult;
}

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    CreditScore: 650,
    Gender: 1,
    Age: 35,
    Tenure: 5,
    Balance: 45000.0,
    NumOfProducts: 1,
    HasCrCard: 1,
    IsActiveMember: 1,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: Number(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Assuming FastAPI is running locally on port 8000
      // In production, this should point to your Render backend URL
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Error connecting to the API. Make sure the FastAPI server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Customer Insight</h1>
        <p className="subtitle">AI-Powered Churn Prediction (XGBoost vs FT-Transformer)</p>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="row">
          <div className="form-group">
            <label>Credit Score</label>
            <input type="number" name="CreditScore" value={formData.CreditScore} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label>Age</label>
            <input type="number" name="Age" value={formData.Age} onChange={handleChange} required />
          </div>
        </div>

        <div className="row">
          <div className="form-group">
            <label>Balance ($)</label>
            <input type="number" step="0.01" name="Balance" value={formData.Balance} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label>Tenure (Years)</label>
            <input type="number" name="Tenure" value={formData.Tenure} onChange={handleChange} required />
          </div>
          <div className="form-group">
            <label>Products</label>
            <input type="number" name="NumOfProducts" value={formData.NumOfProducts} onChange={handleChange} required />
          </div>
        </div>

        <div className="row">
          <div className="form-group">
            <label>Gender</label>
            <select name="Gender" value={formData.Gender} onChange={handleChange}>
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
          <div className="form-group">
            <label>Credit Card?</label>
            <select name="HasCrCard" value={formData.HasCrCard} onChange={handleChange}>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>
          <div className="form-group">
            <label>Active Member?</label>
            <select name="IsActiveMember" value={formData.IsActiveMember} onChange={handleChange}>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>
        </div>

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Analyzing Profile...' : 'Predict Churn Risk'}
        </button>
      </form>

      {error && (
        <div style={{ color: '#ef4444', marginTop: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      {result && (
        <div className="results-container">
          <div className={`result-card ${result.xgboost.prediction === 1 ? 'risk' : 'safe'}`}>
            <h4>XGBoost</h4>
            <div className="result-title">
              {result.xgboost.prediction === 1 ? 'High Risk' : 'Low Risk'}
            </div>
            <div className="result-prob">
              {(result.xgboost.probability * 100).toFixed(1)}%
            </div>
            <div className="result-desc">
              Model Accuracy: {(result.xgboost.accuracy * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className={`result-card ${result.ft_transformer.prediction === 1 ? 'risk' : 'safe'}`}>
            <h4>FT-Transformer</h4>
            <div className="result-title">
              {result.ft_transformer.prediction === 1 ? 'High Risk' : 'Low Risk'}
            </div>
            <div className="result-prob">
              {(result.ft_transformer.probability * 100).toFixed(1)}%
            </div>
            <div className="result-desc">
              Model Accuracy: {(result.ft_transformer.accuracy * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
