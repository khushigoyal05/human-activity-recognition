// UI Elements
const btnSimulate = document.getElementById('btn-simulate');
const btnStop = document.getElementById('btn-stop');
const txtPrediction = document.getElementById('main-prediction');
const txtConfidence = document.getElementById('main-confidence');
const iconContainer = document.getElementById('activity-icon');
const iconElement = iconContainer.querySelector('i');

// State
let simulationInterval;
let isSimulating = false;
let currentSampleIndex = 0;

// API Config
const API_URL = "http://127.0.0.1:8000";

// Chart.js Setup: Probabilities Chart
const ctxProb = document.getElementById('probChart').getContext('2d');
const probChart = new Chart(ctxProb, {
    type: 'bar',
    data: {
        labels: ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'],
        datasets: [{
            label: 'Confidence',
            data: [0, 0, 0, 0, 0, 0],
            backgroundColor: [
                '#34d399', '#38bdf8', '#818cf8', '#fbbf24', '#fb923c', '#f87171'
            ],
            borderRadius: 6,
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y', // horizontal bar chart
        scales: {
            x: {
                min: 0,
                max: 1,
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
            },
            y: {
                grid: { display: false },
                ticks: { color: '#94a3b8', font: { family: 'Outfit' } }
            }
        },
        plugins: {
            legend: { display: false }
        },
        animation: { duration: 300 }
    }
});

// Chart.js Setup: Live Sensor Chart
const ctxSensor = document.getElementById('sensorChart').getContext('2d');
const MAX_DATA_POINTS = 50;

// Initialize empty arrays for moving chart
const timeLabels = Array(MAX_DATA_POINTS).fill('');
const accX = Array(MAX_DATA_POINTS).fill(0);
const accY = Array(MAX_DATA_POINTS).fill(0);
const accZ = Array(MAX_DATA_POINTS).fill(0);

const sensorChart = new Chart(ctxSensor, {
    type: 'line',
    data: {
        labels: timeLabels,
        datasets: [
            {
                label: 'Acc X',
                data: accX,
                borderColor: '#f43f5e',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'Acc Y',
                data: accY,
                borderColor: '#10b981',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'Acc Z',
                data: accZ,
                borderColor: '#3b82f6',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { display: false },
            y: {
                min: -2,
                max: 2,
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: {
                position: 'top',
                align: 'end',
                labels: { color: '#94a3b8', boxWidth: 12, usePointStyle: true }
            }
        },
        animation: { duration: 0 } // No animation for real-time scrolling
    }
});

// Helper: Map Activity to Icons and CSS classes
const activityStyles = {
    "WALKING": { icon: "fa-person-walking", class: "walking" },
    "WALKING UPSTAIRS": { icon: "fa-stairs", class: "upstairs" },
    "WALKING DOWNSTAIRS": { icon: "fa-arrow-down-short-wide", class: "downstairs" },
    "SITTING": { icon: "fa-chair", class: "sitting" },
    "STANDING": { icon: "fa-person", class: "standing" },
    "LAYING": { icon: "fa-bed", class: "laying" },
    "WAITING...": { icon: "fa-user", class: "" }
};

function updateUI(prediction, confidence, probs, features) {
    // 1. Update Main Text
    txtPrediction.innerText = prediction;
    txtConfidence.innerText = (confidence * 100).toFixed(1) + "%";
    
    // 2. Update Styles
    const style = activityStyles[prediction];
    
    // Remove old classes
    txtPrediction.className = "";
    txtConfidence.className = "confidence-value";
    iconContainer.className = "activity-icon";
    
    if (style.class) {
        txtPrediction.classList.add(style.class);
        txtConfidence.classList.add(style.class);
        iconContainer.classList.add("icon-" + style.class);
    }
    
    iconElement.className = `fa-solid ${style.icon}`;

    // 3. Update Probabilities Chart
    probChart.data.datasets[0].data = [
        probs["WALKING"],
        probs["WALKING UPSTAIRS"],
        probs["WALKING DOWNSTAIRS"],
        probs["SITTING"],
        probs["STANDING"],
        probs["LAYING"]
    ];
    probChart.update();

    // 4. Update Sensor Chart (Simulating streaming)
    // The dataset is 561 features. The first 3 are tBodyAcc-mean()-X,Y,Z
    // In a real app, we'd plot raw signal, but here we plot the extracted features to simulate movement
    
    // Shift data
    accX.shift(); accY.shift(); accZ.shift();
    
    // Add new data (Features 0, 1, 2)
    accX.push(features[0]);
    accY.push(features[1]);
    accZ.push(features[2]);
    
    sensorChart.update();
}

async function runSimulationStep() {
    try {
        // 1. Fetch a sequential sample from the test set
        const sampleRes = await fetch(`${API_URL}/sample?index=${currentSampleIndex}`);
        const sampleData = await sampleRes.json();
        
        // 2. Send it to the model for prediction
        const predictRes = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: sampleData.features })
        });
        const predictData = await predictRes.json();
        
        // 3. Update UI
        updateUI(predictData.prediction, predictData.confidence, predictData.probabilities, sampleData.features);
        
        // Advance index
        currentSampleIndex++;
        if(currentSampleIndex > 2900) currentSampleIndex = 0;

    } catch (err) {
        console.error("Simulation Error:", err);
        stopSimulation();
        txtPrediction.innerText = "API ERROR";
        txtPrediction.style.color = "#f87171";
    }
}

function startSimulation() {
    if (isSimulating) return;
    isSimulating = true;
    btnSimulate.disabled = true;
    btnStop.disabled = false;
    
    // Reset index on start for consistent demo
    currentSampleIndex = 0;
    
    // Run immediately, then interval
    runSimulationStep();
    simulationInterval = setInterval(runSimulationStep, 800); // Poll every 800ms
}

function stopSimulation() {
    isSimulating = false;
    btnSimulate.disabled = false;
    btnStop.disabled = true;
    clearInterval(simulationInterval);
    
    txtPrediction.innerText = "STOPPED";
    txtPrediction.className = "";
    txtConfidence.innerText = "--";
    txtConfidence.className = "confidence-value";
    iconContainer.className = "activity-icon";
    iconElement.className = "fa-solid fa-power-off";
}

// Event Listeners
btnSimulate.addEventListener('click', startSimulation);
btnStop.addEventListener('click', stopSimulation);
