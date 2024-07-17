document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startButton');
    const endButton = document.getElementById('endButton');
    const severityForm = document.getElementById('severityForm');
    const contractionList = document.getElementById('contractionList');
    let startTime;

    startButton.addEventListener('click', function() {
        fetch('/start_timer', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                startTime = new Date(data.start_time);
                startButton.disabled = true;
                endButton.disabled = false;
                severityForm.style.display = 'block';
            })
            .catch(error => console.error('Error starting timer:', error));
    });

    severityForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const severity = document.getElementById('severity').value;
        fetch('/end_timer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ severity: severity })
        })
        .then(response => response.json())
        .then(data => {
            const endTime = new Date(data.end_time);
            const duration = data.duration;
            const listItem = document.createElement('li');
            listItem.textContent = `Start: ${startTime} - End: ${endTime} - Duration: ${duration} seconds - Severity: ${data.severity}`;
            contractionList.appendChild(listItem);
            startButton.disabled = false;
            endButton.disabled = true;
            severityForm.style.display = 'none';
            document.getElementById('severity').value = '';
            location.reload();  // Reload the page to update the plot
        })
        .catch(error => console.error('Error stopping timer:', error));
    });

    setInterval(function() {
        location.reload();  // Reload the page every 10 minutes to update the plot
    }, 600000);
});
