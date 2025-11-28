/**
 * Mobile Money Fraud Detection System - Frontend Application
 * Handles file upload, API communication, and data visualization
 */

// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const alertBanner = document.getElementById('alertBanner');
const resultsSection = document.getElementById('resultsSection');

// Chart instances
let confusionChart = null;
let performanceChart = null;
let timeSeriesChart = null;

// Selected file
let selectedFile = null;

// ===========================
// Event Listeners
// ===========================

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Analyze button click handler
analyzeBtn.addEventListener('click', analyzeFile);

// Clear button click handler
clearBtn.addEventListener('click', clearSelection);

// ===========================
// File Handling Functions
// ===========================

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a valid CSV file.', 'error');
        return;
    }

    selectedFile = file;
    
    // Show file info
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    fileInfo.innerHTML = `
        <strong>Selected:</strong> ${file.name}<br>
        <strong>Size:</strong> ${sizeMB} MB
    `;
    fileInfo.classList.add('active');
    
    // Enable analyze button
    analyzeBtn.disabled = false;
    
    // Hide previous results
    hideResults();
    hideAlert();
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.remove('active');
    fileInfo.innerHTML = '';
    analyzeBtn.disabled = true;
    hideResults();
    hideAlert();
}

// ===========================
// API Communication
// ===========================

async function analyzeFile() {
    if (!selectedFile) {
        showAlert('Please select a file first.', 'error');
        return;
    }

    // Show loading
    showLoading();
    hideAlert();
    hideResults();

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        if (data.success) {
            showAlert('Analysis completed successfully!', 'success');
            displayResults(data);
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        showAlert(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ===========================
// Results Display Functions
// ===========================

function displayResults(data) {
    // Display dataset summary
    displaySummary(data.dataset_summary);
    
    // Display performance metrics
    displayMetrics(data.performance);
    
    // Display charts
    displayConfusionChart(data.confusion_matrices);
    displayPerformanceChart(data.performance);
    displayTimeSeriesChart(data.anomaly_scores_over_time);
    
    // Display flagged transactions
    displayFlaggedTransactions(data.flagged_transactions);
    
    // Show results section
    showResults();
}

function displaySummary(summary) {
    document.getElementById('totalTransactions').textContent = formatNumber(summary.total_transactions);
    document.getElementById('fraudTransactions').textContent = formatNumber(summary.fraudulent_transactions);
    document.getElementById('validTransactions').textContent = formatNumber(summary.valid_transactions);
    document.getElementById('fraudRate').textContent = summary.fraud_rate;
}

function displayMetrics(performance) {
    // Isolation Forest metrics
    document.getElementById('ifAccuracy').textContent = formatPercentage(performance.isolation_forest.accuracy);
    document.getElementById('ifPrecision').textContent = formatPercentage(performance.isolation_forest.precision);
    document.getElementById('ifRecall').textContent = formatPercentage(performance.isolation_forest.recall);
    document.getElementById('ifF1').textContent = formatPercentage(performance.isolation_forest.f1);
    
    // LOF metrics
    document.getElementById('lofAccuracy').textContent = formatPercentage(performance.lof.accuracy);
    document.getElementById('lofPrecision').textContent = formatPercentage(performance.lof.precision);
    document.getElementById('lofRecall').textContent = formatPercentage(performance.lof.recall);
    document.getElementById('lofF1').textContent = formatPercentage(performance.lof.f1);
    
    // Autoencoder metrics
    document.getElementById('aeReconError').textContent = performance.autoencoder.reconstruction_error.toFixed(4);
    document.getElementById('aeThreshold').textContent = performance.autoencoder.threshold.toFixed(4);
    document.getElementById('aeLoss').textContent = performance.autoencoder.loss.toFixed(4);
    
    // Hybrid ensemble metrics
    document.getElementById('hybridScore').textContent = performance.hybrid.score.toFixed(4);
    document.getElementById('hybridAccuracy').textContent = formatPercentage(performance.hybrid.accuracy);
    document.getElementById('hybridPrecision').textContent = formatPercentage(performance.hybrid.precision);
    document.getElementById('hybridF1').textContent = formatPercentage(performance.hybrid.f1);
}

function displayConfusionChart(confusionMatrices) {
    const ctx = document.getElementById('confusionChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (confusionChart) {
        confusionChart.destroy();
    }
    
    const labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'];
    
    confusionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Isolation Forest',
                    data: [
                        confusionMatrices.IF.TN,
                        confusionMatrices.IF.FP,
                        confusionMatrices.IF.FN,
                        confusionMatrices.IF.TP
                    ],
                    backgroundColor: 'rgba(124, 58, 237, 0.7)',
                    borderColor: '#7C3AED',
                    borderWidth: 1
                },
                {
                    label: 'LOF',
                    data: [
                        confusionMatrices.LOF.TN,
                        confusionMatrices.LOF.FP,
                        confusionMatrices.LOF.FN,
                        confusionMatrices.LOF.TP
                    ],
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: '#6366F1',
                    borderWidth: 1
                },
                {
                    label: 'Autoencoder',
                    data: [
                        confusionMatrices.AE.TN,
                        confusionMatrices.AE.FP,
                        confusionMatrices.AE.FN,
                        confusionMatrices.AE.TP
                    ],
                    backgroundColor: 'rgba(6, 182, 212, 0.7)',
                    borderColor: '#06B6D4',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#A0A0B2'
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#A0A0B2'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    ticks: {
                        color: '#A0A0B2'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            }
        }
    });
}

function displayPerformanceChart(performance) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    performanceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Isolation Forest', 'LOF', 'Hybrid Ensemble'],
            datasets: [{
                data: [
                    (performance.isolation_forest.f1 * 100).toFixed(1),
                    (performance.lof.f1 * 100).toFixed(1),
                    (performance.hybrid.f1 * 100).toFixed(1)
                ],
                backgroundColor: [
                    'rgba(124, 58, 237, 0.8)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(6, 182, 212, 0.8)'
                ],
                borderColor: [
                    '#7C3AED',
                    '#6366F1',
                    '#06B6D4'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#A0A0B2',
                        padding: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}% F1 Score`;
                        }
                    }
                }
            }
        }
    });
}

function displayTimeSeriesChart(timeSeriesData) {
    const ctx = document.getElementById('timeSeriesChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (timeSeriesChart) {
        timeSeriesChart.destroy();
    }
    
    // Extract data for chart
    const labels = timeSeriesData.map(d => `Step ${d.step}`);
    const ifData = timeSeriesData.map(d => d.if_avg);
    const lofData = timeSeriesData.map(d => d.lof_avg);
    const aeData = timeSeriesData.map(d => d.ae_avg);
    
    timeSeriesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Isolation Forest',
                    data: ifData,
                    borderColor: '#7C3AED',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'LOF',
                    data: lofData,
                    borderColor: '#6366F1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Autoencoder',
                    data: aeData,
                    borderColor: '#06B6D4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#A0A0B2'
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#A0A0B2',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                y: {
                    ticks: {
                        color: '#A0A0B2'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    title: {
                        display: true,
                        text: 'Anomaly Score',
                        color: '#A0A0B2'
                    }
                }
            }
        }
    });
}

function displayFlaggedTransactions(transactions) {
    const tbody = document.getElementById('flaggedTableBody');
    
    if (!transactions || transactions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="no-data">No flagged transactions found</td></tr>';
        return;
    }
    
    let html = '';
    transactions.forEach(tx => {
        const badgeClass = tx.final_result === 'Fraud' ? 'badge-fraud' : 'badge-valid';
        html += `
            <tr>
                <td>${tx.id}</td>
                <td>$${formatNumber(tx.amount)}</td>
                <td>${tx.if_score.toFixed(4)}</td>
                <td>${tx.lof_score.toFixed(4)}</td>
                <td>${tx.ae_score.toFixed(4)}</td>
                <td><span class="badge ${badgeClass}">${tx.final_result}</span></td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// ===========================
// UI Helper Functions
// ===========================

function showLoading() {
    loading.classList.add('active');
}

function hideLoading() {
    loading.classList.remove('active');
}

function showAlert(message, type) {
    alertBanner.textContent = message;
    alertBanner.className = `alert ${type}`;
}

function hideAlert() {
    alertBanner.className = 'alert';
}

function showResults() {
    resultsSection.classList.add('active');
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    resultsSection.classList.remove('active');
}

// ===========================
// Formatting Functions
// ===========================

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

// ===========================
// Initialization
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Mobile Money Fraud Detection System initialized');
});
