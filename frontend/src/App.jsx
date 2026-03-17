import { useState } from 'react'
import './App.css'

function App() {
  const [abstract, setAbstract] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    
    try {
      // URL-encode the abstract to handle spaces/special characters
      const url = `http://localhost:8000/predict?text=${encodeURIComponent(abstract)}`;
      
      const response = await fetch(url, {
        method: 'POST', // Keep as POST as per your FastAPI route
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error("Fetch error:", error);
      alert("Error: Could not connect to the model. Check the console for details.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>SkimLit Classifier</h1>
      <form onSubmit={handlePredict}>
        <textarea
          className="abstract-input"
          placeholder="Paste your research abstract here..."
          value={abstract}
          onChange={(e) => setAbstract(e.target.value)}
          rows={12}
          required
        />
        <br />
        <button type="submit" disabled={loading} className='button-container'>
          {loading ? "Processing..." : "Summarize!"}
        </button>
      </form>

      {prediction && (
        <div className="result-area">
          <h3>Prediction Result:</h3>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export default App