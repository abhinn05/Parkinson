const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultDiv.innerHTML = 'Processing...';

  const formData = new FormData(form); // includes age as string

  try {
    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    resultDiv.innerHTML = data.prediction || `Error: ${data.error}`;
  } catch (err) {
    resultDiv.innerHTML = 'Something went wrong!';
  }
});
