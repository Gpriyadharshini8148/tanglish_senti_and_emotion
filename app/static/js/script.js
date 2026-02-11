document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const trainBtn = document.getElementById('train-btn');
    const textInput = document.getElementById('text-input');
    const statusBox = document.getElementById('status-display');
    const statusText = document.getElementById('status-text');

    const sentimentValue = document.getElementById('sentiment-value');
    const sentimentBar = document.getElementById('sentiment-conf-bar');
    const sentimentConfText = document.getElementById('sentiment-conf-text');

    const emotionValue = document.getElementById('emotion-value');
    const emotionBar = document.getElementById('emotion-conf-bar');
    const emotionConfText = document.getElementById('emotion-conf-text');

    let isTraining = false;

    // Analyze text function
    const analyzeText = async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (data.error) {
                statusBox.classList.remove('hidden');
                statusText.innerText = data.error;
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-bolt"></i> Analyze Text';
                return;
            }

            // Update Sentiment
            sentimentValue.innerText = data.sentiment;
            sentimentBar.style.width = `${data.sentiment_confidence}%`;
            sentimentConfText.innerText = `Confidence: ${data.sentiment_confidence}%`;

            // Update Emotion
            emotionValue.innerText = data.emotion;
            emotionBar.style.width = `${data.emotion_confidence}%`;
            emotionConfText.innerText = `Confidence: ${data.emotion_confidence}%`;

        } catch (error) {
            console.error('Error:', error);
            statusBox.classList.remove('hidden');
            statusText.innerText = 'Failed to connect to the server.';
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-bolt"></i> Analyze Text';
        }
    };

    // Train model function
    const startTraining = async () => {
        if (isTraining) return;

        try {
            const response = await fetch('/train', { method: 'POST' });
            const data = await response.json();

            statusBox.classList.remove('hidden');
            statusText.innerText = data.status;

            if (data.status.includes('started')) {
                isTraining = true;
                trainBtn.disabled = true;
                pollStatus();
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    // Poll training status
    const pollStatus = async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            statusText.innerText = data.status;

            if (data.status.includes('Complete') || data.status.includes('Failed')) {
                isTraining = false;
                trainBtn.disabled = false;
                if (data.status.includes('Failed')) {
                    statusBox.style.background = 'rgba(239, 68, 68, 0.1)';
                    statusBox.style.borderColor = 'rgba(239, 68, 68, 0.2)';
                }
                return;
            }

            setTimeout(pollStatus, 3000);
        } catch (error) {
            console.error('Error polling status:', error);
            isTraining = false;
            trainBtn.disabled = false;
        }
    };

    analyzeBtn.addEventListener('click', analyzeText);
    trainBtn.addEventListener('click', startTraining);

    // Initial status check
    pollStatus();
});
