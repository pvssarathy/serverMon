# VM Resource Utilization Monitor with LLM Recommendations

A production-ready system for monitoring CPU, memory, and storage across multiple Linux and Windows VMs, with AI-powered optimization recommendations using open-source LLM models.

## Features

- **Comprehensive Resource Monitoring**
  - CPU utilization and load averages
  - Memory and swap usage
  - Disk space across all partitions
  - Network connections
  - System information

- **Intelligent Analysis**
  - Configurable thresholds for warnings and critical alerts
  - Severity-based issue detection
  - Historical trend analysis

- **AI-Powered Recommendations**
  - Uses open-source LLMs (via Ollama) for intelligent suggestions
  - Context-aware recommendations based on system metrics
  - Fallback to rule-based recommendations if LLM unavailable

- **Production Features**
  - Comprehensive logging
  - JSON report generation
  - Continuous monitoring with scheduling
  - Automatic report cleanup
  - Graceful shutdown handling
  - Cross-platform support (Linux/Windows)

## Architecture

```
vm_monitor.py              # Core monitoring and analysis
├── ResourceMonitor        # Collects system metrics
├── LLMRecommendationEngine # Generates AI recommendations
└── VMMonitorOrchestrator  # Orchestrates the workflow

monitor_scheduler.py       # Continuous monitoring
└── MonitorScheduler       # Periodic execution and trend analysis
```

## Prerequisites

- Python 3.8+
- Ollama (for LLM recommendations)
- Linux or Windows operating system

## Installation

### Quick Setup (Linux)

```bash
# Clone or download the files
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama**

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

**macOS:**
```bash
brew install ollama
```

4. **Pull LLM Model**
```bash
ollama pull llama3.2
# Other options: mistral, phi, codellama, etc.
```

## Usage

### Single Run

Run a one-time analysis:

```bash
python vm_monitor.py
```

### Continuous Monitoring

Monitor every 5 minutes (300 seconds):

```bash
python monitor_scheduler.py --interval 300
```

Monitor once and exit:

```bash
python monitor_scheduler.py --once
```

Check monitoring status:

```bash
python monitor_scheduler.py --status
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Threshold Configuration
thresholds:
  cpu:
    warning: 70.0    # CPU warning threshold (%)
    critical: 90.0   # CPU critical threshold (%)
  memory:
    warning: 75.0
    critical: 90.0
  disk:
    warning: 80.0
    critical: 95.0

# LLM Configuration
llm:
  host: "http://localhost:11434"
  model: "llama3.2"  # Change to preferred model
  timeout: 60

# Monitoring Configuration
monitoring:
  interval: 300      # Check every 5 minutes
  history_size: 100  # Keep 100 samples in memory
```

## Output

### Console Output

```
================================================================================
VM RESOURCE ANALYSIS - hostname
Timestamp: 2025-11-02T10:30:00
================================================================================

📊 RESOURCE UTILIZATION:
  CPU: 45.2%
  Memory: 62.5% (5.00GB / 8.00GB)
  Swap: 10.2%

💾 DISK USAGE:
  /: 65.3% (45.67GB / 70.00GB)
  /home: 42.1% (210.50GB / 500.00GB)

✅ No issues detected - system is healthy

🤖 RECOMMENDATIONS (LLM):
--------------------------------------------------------------------------------
1. SHORT-TERM OPTIMIZATIONS:
   - System is operating efficiently within normal parameters
   - Continue current monitoring practices
   
2. MONITORING SUGGESTIONS:
   - Set up alerts for when CPU exceeds 75%
   - Track memory usage trends weekly
--------------------------------------------------------------------------------
```

### JSON Reports

Reports are saved to `reports/` directory:

```json
{
  "metrics": {
    "timestamp": "2025-11-02T10:30:00",
    "hostname": "prod-server-01",
    "cpu_percent": 45.2,
    "memory_percent": 62.5,
    ...
  },
  "analysis": {
    "issues": [],
    "has_critical": false
  },
  "recommendations": {
    "source": "llm",
    "model": "llama3.2",
    "recommendations": "..."
  }
}
```

## LLM Models

Supported open-source models via Ollama:

- **llama3.2** (Recommended) - Best balance of quality and speed
- **mistral** - Fast and efficient
- **phi** - Lightweight option
- **codellama** - Good for technical recommendations
- **mixtral** - High quality, larger model

Pull any model:
```bash
ollama pull <model-name>
```

## Monitoring Multiple VMs

### Option 1: Agent-Based (Recommended)

Deploy the monitor on each VM:

```bash
# On each VM
./setup.sh
python monitor_scheduler.py --interval 300
```

### Option 2: SSH-Based Collection

Use the included remote collection script:

```bash
python remote_collector.py --hosts hosts.txt
```

### Option 3: Central Dashboard

Aggregate reports from multiple VMs:

```bash
python dashboard.py --reports-dir /path/to/shared/reports
```

## Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/vm-monitor.service`:

```ini
[Unit]
Description=VM Resource Monitor
After=network.target

[Service]
Type=simple
User=monitor
WorkingDirectory=/opt/vm-monitor
ExecStart=/opt/vm-monitor/venv/bin/python monitor_scheduler.py --interval 300
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable vm-monitor
sudo systemctl start vm-monitor
```

### Windows Service

Use NSSM (Non-Sucking Service Manager):

```powershell
nssm install VMMonitor "C:\vm-monitor\venv\Scripts\python.exe"
nssm set VMMonitor AppDirectory "C:\vm-monitor"
nssm set VMMonitor AppParameters "monitor_scheduler.py --interval 300"
nssm start VMMonitor
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "monitor_scheduler.py", "--interval", "300"]
```

## Customization

### Adjust Thresholds

```python
monitor = ResourceMonitor(
    cpu_warning=80.0,     # Custom warning level
    cpu_critical=95.0,    # Custom critical level
    memory_warning=80.0,
    memory_critical=95.0,
    disk_warning=85.0,
    disk_critical=98.0
)
```

### Use Different LLM

```python
llm_engine = LLMRecommendationEngine(
    ollama_host="http://localhost:11434",
    model="mistral",  # Change model
    timeout=90        # Adjust timeout
)
```

### Custom Prompts

Edit the `generate_prompt()` method in `LLMRecommendationEngine` class to customize the AI prompts.

## Troubleshooting

### Ollama Connection Issues

1. Ensure Ollama is running:
```bash
ollama serve
```

2. Check Ollama status:
```bash
curl http://localhost:11434/api/tags
```

3. Verify model is pulled:
```bash
ollama list
```

### Permission Issues (Linux)

If you can't access certain metrics:

```bash
# Run with elevated privileges
sudo python vm_monitor.py
```

### High Memory Usage

Reduce history size in configuration:

```yaml
monitoring:
  history_size: 50  # Reduce from default 100
```

## Performance

- **CPU Impact**: <1% during collection
- **Memory Usage**: ~50-100MB base + history
- **Disk I/O**: Minimal (reports only)
- **Network**: None (unless using remote collection)

## Best Practices

1. **Set Appropriate Thresholds**: Adjust based on your workload
2. **Monitor Trends**: Look at historical data, not just snapshots
3. **Regular Reviews**: Review LLM recommendations weekly
4. **Test Alerts**: Verify alert mechanisms work
5. **Backup Reports**: Keep historical reports for capacity planning
6. **Update Models**: Regularly update Ollama models

## Security Considerations

- Run with minimum required privileges
- Secure API endpoints if exposing remotely
- Protect configuration files (may contain credentials)
- Use SSH keys for remote collection
- Restrict report directory access
- Review LLM recommendations before acting

## Contributing

Contributions welcome! Please:

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Use type hints

## License

MIT License - See LICENSE file

## Support

For issues or questions:
- Check logs: `vm_monitor.log`, `monitor_scheduler.log`
- Review troubleshooting section
- Open an issue on GitHub

## Roadmap

- [ ] Web dashboard UI
- [ ] Database storage for metrics
- [ ] Email/Slack alerts
- [ ] Process-level analysis
- [ ] Container monitoring support
- [ ] Predictive analytics
- [ ] Custom plugin system
- [ ] Multi-cloud support (AWS, Azure, GCP)

## Acknowledgments

- Built with [psutil](https://github.com/giampaolo/psutil)
- LLM integration via [Ollama](https://ollama.com/)
- Inspired by modern DevOps practices
