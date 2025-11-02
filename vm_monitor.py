"""
VM Resource Utilization Monitor with LLM-based Recommendations
A production-ready system for monitoring CPU, memory, and storage across multiple VMs
and generating optimization recommendations using open-source LLM models.
"""

import psutil
import platform
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from pathlib import Path
import subprocess
import sys

# For LLM integration - using Ollama (open-source LLM runtime)
try:
    import requests
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vm_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Resource utilization severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    NORMAL = "normal"
    OPTIMAL = "optimal"


@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics"""
    timestamp: str
    hostname: str
    os_type: str
    os_version: str
    
    # CPU metrics
    cpu_percent: float
    cpu_count: int
    cpu_freq_current: float
    cpu_freq_max: float
    load_avg_1min: Optional[float]
    load_avg_5min: Optional[float]
    load_avg_15min: Optional[float]
    
    # Memory metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    
    # Disk metrics
    disk_partitions: List[Dict]
    
    # Network metrics
    network_connections: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class ResourceMonitor:
    """Monitor system resources with configurable thresholds"""
    
    def __init__(self, 
                 cpu_warning: float = 70.0,
                 cpu_critical: float = 90.0,
                 memory_warning: float = 75.0,
                 memory_critical: float = 90.0,
                 disk_warning: float = 80.0,
                 disk_critical: float = 95.0):
        """
        Initialize resource monitor with thresholds
        
        Args:
            cpu_warning: CPU usage warning threshold (%)
            cpu_critical: CPU usage critical threshold (%)
            memory_warning: Memory usage warning threshold (%)
            memory_critical: Memory usage critical threshold (%)
            disk_warning: Disk usage warning threshold (%)
            disk_critical: Disk usage critical threshold (%)
        """
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.disk_warning = disk_warning
        self.disk_critical = disk_critical
        
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        
    def get_system_info(self) -> Dict:
        """Get basic system information"""
        return {
            "hostname": platform.node(),
            "os_type": platform.system(),
            "os_version": platform.version(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        }
    
    def get_cpu_metrics(self) -> Dict:
        """Collect CPU utilization metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "cpu_per_core": cpu_per_core,
                "cpu_count": cpu_count,
                "cpu_freq_current": cpu_freq.current if cpu_freq else 0,
                "cpu_freq_max": cpu_freq.max if cpu_freq else 0
            }
            
            # Load average (Linux/Unix only)
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()
                metrics["load_avg_1min"] = load_avg[0]
                metrics["load_avg_5min"] = load_avg[1]
                metrics["load_avg_15min"] = load_avg[2]
            else:
                metrics["load_avg_1min"] = None
                metrics["load_avg_5min"] = None
                metrics["load_avg_15min"] = None
                
            return metrics
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            raise
    
    def get_memory_metrics(self) -> Dict:
        """Collect memory utilization metrics"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "memory_total_gb": mem.total / (1024**3),
                "memory_used_gb": mem.used / (1024**3),
                "memory_available_gb": mem.available / (1024**3),
                "memory_percent": mem.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            raise
    
    def get_disk_metrics(self) -> List[Dict]:
        """Collect disk utilization metrics"""
        try:
            disk_info = []
            partitions = psutil.disk_partitions(all=False)
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    })
                except PermissionError:
                    # Skip partitions we don't have access to
                    continue
                    
            return disk_info
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            raise
    
    def get_network_metrics(self) -> Dict:
        """Collect network metrics"""
        try:
            connections = len(psutil.net_connections())
            net_io = psutil.net_io_counters()
            
            return {
                "connections": connections,
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {"connections": 0}
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect all resource metrics"""
        try:
            logger.info("Collecting system metrics...")
            
            sys_info = self.get_system_info()
            cpu_metrics = self.get_cpu_metrics()
            mem_metrics = self.get_memory_metrics()
            disk_metrics = self.get_disk_metrics()
            net_metrics = self.get_network_metrics()
            
            metrics = ResourceMetrics(
                timestamp=datetime.now().isoformat(),
                hostname=sys_info["hostname"],
                os_type=sys_info["os_type"],
                os_version=sys_info["os_version"],
                cpu_percent=cpu_metrics["cpu_percent"],
                cpu_count=cpu_metrics["cpu_count"],
                cpu_freq_current=cpu_metrics["cpu_freq_current"],
                cpu_freq_max=cpu_metrics["cpu_freq_max"],
                load_avg_1min=cpu_metrics["load_avg_1min"],
                load_avg_5min=cpu_metrics["load_avg_5min"],
                load_avg_15min=cpu_metrics["load_avg_15min"],
                memory_total_gb=mem_metrics["memory_total_gb"],
                memory_used_gb=mem_metrics["memory_used_gb"],
                memory_available_gb=mem_metrics["memory_available_gb"],
                memory_percent=mem_metrics["memory_percent"],
                swap_total_gb=mem_metrics["swap_total_gb"],
                swap_used_gb=mem_metrics["swap_used_gb"],
                swap_percent=mem_metrics["swap_percent"],
                disk_partitions=disk_metrics,
                network_connections=net_metrics["connections"]
            )
            
            logger.info("Metrics collection completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise
    
    def analyze_severity(self, metrics: ResourceMetrics) -> Dict:
        """Analyze metrics and determine severity levels"""
        issues = []
        
        # Analyze CPU
        if metrics.cpu_percent >= self.cpu_critical:
            issues.append({
                "resource": "CPU",
                "severity": Severity.CRITICAL.value,
                "value": metrics.cpu_percent,
                "threshold": self.cpu_critical,
                "message": f"CPU usage is critically high at {metrics.cpu_percent:.1f}%"
            })
        elif metrics.cpu_percent >= self.cpu_warning:
            issues.append({
                "resource": "CPU",
                "severity": Severity.WARNING.value,
                "value": metrics.cpu_percent,
                "threshold": self.cpu_warning,
                "message": f"CPU usage is elevated at {metrics.cpu_percent:.1f}%"
            })
        
        # Analyze Memory
        if metrics.memory_percent >= self.memory_critical:
            issues.append({
                "resource": "Memory",
                "severity": Severity.CRITICAL.value,
                "value": metrics.memory_percent,
                "threshold": self.memory_critical,
                "message": f"Memory usage is critically high at {metrics.memory_percent:.1f}%"
            })
        elif metrics.memory_percent >= self.memory_warning:
            issues.append({
                "resource": "Memory",
                "severity": Severity.WARNING.value,
                "value": metrics.memory_percent,
                "threshold": self.memory_warning,
                "message": f"Memory usage is elevated at {metrics.memory_percent:.1f}%"
            })
        
        # Analyze Disk
        for disk in metrics.disk_partitions:
            if disk["percent"] >= self.disk_critical:
                issues.append({
                    "resource": f"Disk ({disk['mountpoint']})",
                    "severity": Severity.CRITICAL.value,
                    "value": disk["percent"],
                    "threshold": self.disk_critical,
                    "message": f"Disk {disk['mountpoint']} is critically full at {disk['percent']:.1f}%"
                })
            elif disk["percent"] >= self.disk_warning:
                issues.append({
                    "resource": f"Disk ({disk['mountpoint']})",
                    "severity": Severity.WARNING.value,
                    "value": disk["percent"],
                    "threshold": self.disk_warning,
                    "message": f"Disk {disk['mountpoint']} usage is elevated at {disk['percent']:.1f}%"
                })
        
        # Analyze Swap
        if metrics.swap_percent > 50:
            issues.append({
                "resource": "Swap",
                "severity": Severity.WARNING.value,
                "value": metrics.swap_percent,
                "threshold": 50,
                "message": f"High swap usage at {metrics.swap_percent:.1f}% indicates memory pressure"
            })
        
        return {
            "issues": issues,
            "has_critical": any(i["severity"] == Severity.CRITICAL.value for i in issues),
            "has_warning": any(i["severity"] == Severity.WARNING.value for i in issues)
        }


class LLMRecommendationEngine:
    """Generate recommendations using open-source LLM via Ollama"""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 model: str = "llama3.2",
                 timeout: int = 60):
        """
        Initialize LLM recommendation engine
        
        Args:
            ollama_host: Ollama API endpoint
            model: Model name (llama3.2, mistral, phi, etc.)
            timeout: Request timeout in seconds
        """
        self.ollama_host = ollama_host
        self.model = model
        self.timeout = timeout
        
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False
    
    def generate_prompt(self, metrics: ResourceMetrics, analysis: Dict) -> str:
        """Generate a comprehensive prompt for the LLM"""
        
        prompt = f"""You are an expert DevOps engineer analyzing VM resource utilization. Provide specific, actionable recommendations.

SYSTEM INFORMATION:
- Hostname: {metrics.hostname}
- OS: {metrics.os_type} {metrics.os_version}
- CPU Cores: {metrics.cpu_count}

CURRENT RESOURCE UTILIZATION:
- CPU Usage: {metrics.cpu_percent:.1f}%
- Memory Usage: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.2f}GB / {metrics.memory_total_gb:.2f}GB)
- Swap Usage: {metrics.swap_percent:.1f}% ({metrics.swap_used_gb:.2f}GB / {metrics.swap_total_gb:.2f}GB)
"""
        
        if metrics.load_avg_1min is not None:
            prompt += f"- Load Average: {metrics.load_avg_1min:.2f}, {metrics.load_avg_5min:.2f}, {metrics.load_avg_15min:.2f}\n"
        
        prompt += "\nDISK USAGE:\n"
        for disk in metrics.disk_partitions:
            prompt += f"- {disk['mountpoint']}: {disk['percent']:.1f}% ({disk['used_gb']:.2f}GB / {disk['total_gb']:.2f}GB)\n"
        
        if analysis["issues"]:
            prompt += "\nIDENTIFIED ISSUES:\n"
            for issue in analysis["issues"]:
                prompt += f"- [{issue['severity'].upper()}] {issue['message']}\n"
        else:
            prompt += "\nNo critical issues identified. System is running within normal parameters.\n"
        
        prompt += """
PROVIDE RECOMMENDATIONS IN THE FOLLOWING FORMAT:

1. IMMEDIATE ACTIONS (if critical issues exist):
   - List urgent steps to resolve critical resource constraints

2. SHORT-TERM OPTIMIZATIONS:
   - Specific configurations, cleanup tasks, or process optimizations

3. LONG-TERM IMPROVEMENTS:
   - Infrastructure scaling, capacity planning, or architectural changes

4. MONITORING SUGGESTIONS:
   - Additional metrics to track or alerts to configure

Keep recommendations specific, actionable, and prioritized by impact."""
        
        return prompt
    
    def generate_recommendations(self, 
                                metrics: ResourceMetrics, 
                                analysis: Dict) -> Dict:
        """Generate recommendations using LLM"""
        
        # Check Ollama connection
        if not self.check_ollama_connection():
            logger.warning("Ollama not available, returning rule-based recommendations")
            return self._generate_fallback_recommendations(metrics, analysis)
        
        try:
            prompt = self.generate_prompt(metrics, analysis)
            
            logger.info(f"Generating recommendations using {self.model}...")
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                recommendations = result.get("response", "")
                
                return {
                    "source": "llm",
                    "model": self.model,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return self._generate_fallback_recommendations(metrics, analysis)
                
        except Exception as e:
            logger.error(f"Failed to generate LLM recommendations: {e}")
            return self._generate_fallback_recommendations(metrics, analysis)
    
    def _generate_fallback_recommendations(self, 
                                          metrics: ResourceMetrics, 
                                          analysis: Dict) -> Dict:
        """Generate rule-based recommendations when LLM is unavailable"""
        
        recommendations = []
        
        # CPU recommendations
        if metrics.cpu_percent >= 90:
            recommendations.append(
                "CRITICAL - CPU: Identify and terminate resource-intensive processes. "
                "Consider vertical scaling (add more CPU cores) or horizontal scaling."
            )
        elif metrics.cpu_percent >= 70:
            recommendations.append(
                "WARNING - CPU: Review process priorities and consider load balancing. "
                "Analyze CPU-intensive applications for optimization opportunities."
            )
        
        # Memory recommendations
        if metrics.memory_percent >= 90:
            recommendations.append(
                "CRITICAL - Memory: Restart memory-leaking applications. "
                "Increase RAM or implement memory caching strategies."
            )
        elif metrics.memory_percent >= 75:
            recommendations.append(
                "WARNING - Memory: Monitor for memory leaks. "
                "Consider application tuning or adding more RAM."
            )
        
        # Swap recommendations
        if metrics.swap_percent > 50:
            recommendations.append(
                "WARNING - Swap: High swap usage indicates memory pressure. "
                "Add more physical RAM to reduce swap dependency."
            )
        
        # Disk recommendations
        for disk in metrics.disk_partitions:
            if disk["percent"] >= 95:
                recommendations.append(
                    f"CRITICAL - Disk {disk['mountpoint']}: Immediate cleanup required. "
                    f"Remove unnecessary files or expand storage capacity."
                )
            elif disk["percent"] >= 80:
                recommendations.append(
                    f"WARNING - Disk {disk['mountpoint']}: Plan for storage expansion. "
                    f"Implement log rotation and cleanup policies."
                )
        
        if not recommendations:
            recommendations.append(
                "System is operating within normal parameters. "
                "Continue regular monitoring and maintain current configurations."
            )
        
        return {
            "source": "rule-based",
            "model": "fallback",
            "recommendations": "\n\n".join(recommendations),
            "timestamp": datetime.now().isoformat()
        }


class VMMonitorOrchestrator:
    """Main orchestrator for VM monitoring and recommendations"""
    
    def __init__(self, 
                 monitor: ResourceMonitor,
                 llm_engine: LLMRecommendationEngine,
                 output_dir: str = "reports"):
        """
        Initialize orchestrator
        
        Args:
            monitor: ResourceMonitor instance
            llm_engine: LLMRecommendationEngine instance
            output_dir: Directory for saving reports
        """
        self.monitor = monitor
        self.llm_engine = llm_engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run_analysis(self, save_report: bool = True) -> Dict:
        """
        Run complete analysis and generate recommendations
        
        Args:
            save_report: Whether to save the report to file
            
        Returns:
            Complete analysis report
        """
        try:
            # Collect metrics
            metrics = self.monitor.collect_metrics()
            
            # Analyze severity
            analysis = self.monitor.analyze_severity(metrics)
            
            # Generate recommendations
            recommendations = self.llm_engine.generate_recommendations(
                metrics, analysis
            )
            
            # Compile report
            report = {
                "metrics": metrics.to_dict(),
                "analysis": analysis,
                "recommendations": recommendations
            }
            
            # Save report
            if save_report:
                self._save_report(report)
            
            # Print summary
            self._print_summary(metrics, analysis, recommendations)
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _save_report(self, report: Dict):
        """Save report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = report["metrics"]["hostname"]
        filename = self.output_dir / f"vm_report_{hostname}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filename}")
    
    def _print_summary(self, 
                      metrics: ResourceMetrics, 
                      analysis: Dict, 
                      recommendations: Dict):
        """Print analysis summary to console"""
        
        print("\n" + "="*80)
        print(f"VM RESOURCE ANALYSIS - {metrics.hostname}")
        print(f"Timestamp: {metrics.timestamp}")
        print("="*80)
        
        print(f"\n📊 RESOURCE UTILIZATION:")
        print(f"  CPU: {metrics.cpu_percent:.1f}%")
        print(f"  Memory: {metrics.memory_percent:.1f}% "
              f"({metrics.memory_used_gb:.2f}GB / {metrics.memory_total_gb:.2f}GB)")
        print(f"  Swap: {metrics.swap_percent:.1f}%")
        
        print(f"\n💾 DISK USAGE:")
        for disk in metrics.disk_partitions:
            print(f"  {disk['mountpoint']}: {disk['percent']:.1f}% "
                  f"({disk['used_gb']:.2f}GB / {disk['total_gb']:.2f}GB)")
        
        if analysis["issues"]:
            print(f"\n⚠️  ISSUES DETECTED ({len(analysis['issues'])}):")
            for issue in analysis["issues"]:
                icon = "🔴" if issue["severity"] == "critical" else "🟡"
                print(f"  {icon} {issue['message']}")
        else:
            print(f"\n✅ No issues detected - system is healthy")
        
        print(f"\n🤖 RECOMMENDATIONS ({recommendations['source'].upper()}):")
        print("-" * 80)
        print(recommendations["recommendations"])
        print("-" * 80)
        print()


def main():
    """Main entry point"""
    
    print("🚀 VM Resource Utilization Monitor")
    print("="*80)
    
    # Initialize components
    monitor = ResourceMonitor(
        cpu_warning=70.0,
        cpu_critical=90.0,
        memory_warning=75.0,
        memory_critical=90.0,
        disk_warning=80.0,
        disk_critical=95.0
    )
    
    llm_engine = LLMRecommendationEngine(
        ollama_host="http://localhost:11434",
        model="llama3.2",  # Change to your preferred model
        timeout=60
    )
    
    orchestrator = VMMonitorOrchestrator(
        monitor=monitor,
        llm_engine=llm_engine,
        output_dir="reports"
    )
    
    # Run analysis
    try:
        report = orchestrator.run_analysis(save_report=True)
        print("✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
