document.getElementById('record-btn').addEventListener('click', async () => {
  const name = document.getElementById('name').value.trim();
  const age = document.getElementById('age').value.trim();
  const status = document.getElementById('status');
  const resultsSection = document.getElementById('results');
  const outputList = document.getElementById('output-list');

  if (!name || !age) {
    alert("Please enter both name and age.");
    return;
  }

  status.textContent = "⏳ Collecting data (audio & video)...";

  // Call the backend API (Flask server) that executes your combined.py logic
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: JSON.stringify({ name, age }),
      headers: { 'Content-Type': 'application/json' }
    });

    const result = await response.json();
    status.textContent = "✅ Prediction complete!";
    resultsSection.hidden = false;

    outputList.innerHTML = `
      <li><strong>Fused Probability:</strong> ${result.fused_proba.toFixed(4)}</li>
      <li><strong>Prediction:</strong> ${result.prediction == 1 ? "Parkinson's Detected" : "No Parkinson's"}</li>
      <li><strong>Blink Rate:</strong> ${result.blink_rate} blinks/min</li>
      <li><strong>Audio Probability:</strong> ${result.audio_proba.toFixed(4)}</li>
      <li><strong>Age Probability:</strong> ${result.age_proba.toFixed(4)}</li>
    `;
  } catch (error) {
    console.error(error);
    status.textContent = "❌ Failed to get prediction. Is the backend running?";
  }
});
