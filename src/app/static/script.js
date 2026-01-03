document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const welcomeMessage = document.getElementById('welcomeMessage');
    const resultsContent = document.getElementById('resultsContent');
    const resultImage = document.getElementById('resultImage');
    const currentStatus = document.getElementById('currentStatus');
    const taskList = document.getElementById('taskList');
    const issueCount = document.getElementById('issueCount');
    const maxSeverity = document.getElementById('maxSeverity');
    const detectedIssuesList = document.getElementById('detectedIssuesList');
    const solutionsList = document.getElementById('solutionsList');

    let currentFile = null;
    let activePolling = null;

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            if (currentFile.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    dropZone.classList.add('hidden');
                    previewContainer.classList.remove('hidden');
                };
                reader.readAsDataURL(currentFile);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    // Buttons
    resetBtn.addEventListener('click', () => {
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerText = 'Uploading...';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            addTaskToHistory(data.request_id, currentFile.name);
            startPolling(data.request_id);

            // UI state after upload
            currentStatus.classList.remove('hidden');
            currentStatus.innerText = 'Processing...';
            welcomeMessage.classList.add('hidden');
            resultsContent.classList.add('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert('Error starting analysis: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = 'Start Analysis';
        }
    });

    function addTaskToHistory(requestId, filename) {
        const time = new Date().toLocaleTimeString();
        const taskItem = document.createElement('div');
        taskItem.className = 'task-item';
        taskItem.id = `task-${requestId}`;
        taskItem.innerHTML = `
            <div class="task-info">
                <span class="task-name">${filename}</span>
                <span class="task-time">${time}</span>
            </div>
            <div class="task-status pending">Pending</div>
        `;
        taskList.prepend(taskItem);
    }

    function startPolling(requestId) {
        if (activePolling) clearInterval(activePolling);

        activePolling = setInterval(async () => {
            try {
                const response = await fetch(`/result/${requestId}`);
                if (!response.ok) return;

                const result = await response.json();

                if (result.status === 'completed') {
                    clearInterval(activePolling);
                    activePolling = null;
                    displayResults(result);
                    updateTaskStatus(requestId, 'completed');
                    currentStatus.innerText = 'Success';
                    currentStatus.classList.remove('pulse');
                } else if (result.status === 'failed') {
                    clearInterval(activePolling);
                    activePolling = null;
                    alert('Detection failed: ' + (result.error || 'Unknown error'));
                    updateTaskStatus(requestId, 'failed');
                    currentStatus.innerText = 'Failed';
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 2000);
    }

    function updateTaskStatus(requestId, status) {
        const taskStatus = document.querySelector(`#task-${requestId} .task-status`);
        if (taskStatus) {
            taskStatus.innerText = status.charAt(0) + status.slice(1);
            taskStatus.className = `task-status ${status}`;
        }
    }

    async function displayResults(result) {
        resultsContent.classList.remove('hidden');
        currentStatus.classList.add('hidden');

        // Count issues
        const issues = result.detected_issues || [];
        issueCount.innerText = issues.length;

        // Find max severity
        let maxSev = 'low';
        issues.forEach(issue => {
            if (issue.severity === 'high') maxSev = 'high';
            else if (issue.severity === 'medium' && maxSev !== 'high') maxSev = 'medium';
        });

        maxSeverity.innerText = issues.length > 0 ? maxSev : '-';
        maxSeverity.className = `stat-value severity-${maxSev}`;

        // Set result image (annotated)
        // Extract filename from issue or use a placeholder if none
        if (issues.length > 0 && issues[0].annotated_image_url) {
            resultImage.src = issues[0].annotated_image_url;
        } else {
            // Fallback: If no objects detected, the worker might not have generated an annotated image
            // We might want to show the original or a notice
            resultImage.src = imagePreview.src;
        }

        // List detected issues
        detectedIssuesList.innerHTML = '';
        if (issues.length === 0) {
            detectedIssuesList.innerHTML = '<p class="empty-info">No structural issues detected.</p>';
        } else {
            issues.forEach(issue => {
                const card = document.createElement('div');
                card.className = 'issue-card';
                card.innerHTML = `
                    <div class="issue-header">
                        <span class="issue-name">${issue.issue_name}</span>
                        <span class="severity-pill severity-${issue.severity || 'low'}">${issue.severity || 'low'}</span>
                    </div>
                <!-- <div class="issue-conf">Confidence: ${Math.round(issue.confidence * 100)}%</div> -->
                `;
                detectedIssuesList.appendChild(card);
            });
        }

        // Fetch recommendations from /issues endpoint for more details if needed
        // For simplicity, we can use the solutions retrieved in the result if they are present
        // Looking at the worker code, it adds 'mitigation' and 'severity' to each detected issue
        solutionsList.innerHTML = '';
        const uniqueIssues = [...new Set(issues.map(i => i.issue_name))];

        if (uniqueIssues.length === 0) {
            solutionsList.innerHTML = '<p class="empty-info">No maintenance required at this time.</p>';
        } else {
            uniqueIssues.forEach(issueName => {
                const issueData = issues.find(i => i.issue_name === issueName);
                const solutionItem = document.createElement('div');
                solutionItem.className = 'solution-item';
                solutionItem.innerHTML = `
                    <h4>${issueName} Management</h4>
                    <p>${issueData.mitigation || "Refer to industrial standard protocols for structural integrity."}</p>
                `;
                solutionsList.appendChild(solutionItem);
            });
        }
    }
});
