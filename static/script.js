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
            });
    });

    endButton.addEventListener('click', function() {
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
            updateChart();
        });
    });

    function updateChart() {
        fetch('/plot_data')
            .then(response => response.json())
            .then(data => {
                const graphData = JSON.parse(data.data);
                const layout = JSON.parse(data.layout);
                Plotly.newPlot('contractionChart', graphData, layout);
            });
    }

    updateChart();
    setInterval(updateChart, 600000); // Update the chart every 10 minutes
});
