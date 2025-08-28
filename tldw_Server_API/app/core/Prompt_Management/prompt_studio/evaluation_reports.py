# evaluation_reports.py
# Generate evaluation reports for Prompt Studio

import json
import csv
from io import StringIO
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from loguru import logger

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available for advanced report features")

########################################################################################################################
# Report Generator

class EvaluationReportGenerator:
    """Generates various report formats for evaluation results."""
    
    def __init__(self):
        """Initialize report generator."""
        pass
    
    ####################################################################################################################
    # Text Reports
    
    def generate_text_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable text report.
        
        Args:
            evaluation_results: Evaluation results from TestRunner
            
        Returns:
            Formatted text report
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("PROMPT STUDIO EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Evaluation ID: {evaluation_results.get('evaluation_id', 'N/A')}")
        report.append(f"Status: {evaluation_results.get('status', 'Unknown')}")
        report.append(f"Total Tests: {evaluation_results.get('total_tests', 0)}")
        report.append(f"Successful Tests: {evaluation_results.get('successful_tests', 0)}")
        report.append(f"Success Rate: {evaluation_results.get('success_rate', 0):.1f}%")
        report.append(f"Execution Time: {evaluation_results.get('execution_time_seconds', 0):.2f} seconds")
        report.append(f"Completed At: {evaluation_results.get('completed_at', 'N/A')}")
        report.append("")
        
        # Aggregated Metrics
        if "aggregated_metrics" in evaluation_results:
            report.append("AGGREGATED METRICS")
            report.append("-" * 40)
            
            metrics = evaluation_results["aggregated_metrics"]
            
            # Overall score
            if "overall_score" in metrics:
                report.append(f"Overall Score: {metrics['overall_score']:.3f}")
            
            # Individual metrics
            if "metrics" in metrics:
                report.append("\nMetric Scores:")
                for metric_name, stats in metrics["metrics"].items():
                    report.append(f"  {metric_name}:")
                    report.append(f"    Mean: {stats.get('mean', 0):.3f}")
                    report.append(f"    Std Dev: {stats.get('std', 0):.3f}")
                    report.append(f"    Min: {stats.get('min', 0):.3f}")
                    report.append(f"    Max: {stats.get('max', 0):.3f}")
            
            # Performance metrics
            if "execution_time" in metrics:
                report.append("\nExecution Time:")
                exec_time = metrics["execution_time"]
                report.append(f"  Mean: {exec_time.get('mean_ms', 0):.1f} ms")
                report.append(f"  Total: {exec_time.get('total_ms', 0):.1f} ms")
            
            if "tokens" in metrics:
                report.append("\nToken Usage:")
                tokens = metrics["tokens"]
                report.append(f"  Mean: {tokens.get('mean', 0):.0f}")
                report.append(f"  Total: {tokens.get('total', 0):.0f}")
            
            if "cost" in metrics:
                report.append("\nEstimated Cost:")
                cost = metrics["cost"]
                report.append(f"  Mean: ${cost.get('mean', 0):.4f}")
                report.append(f"  Total: ${cost.get('total', 0):.4f}")
            
            report.append("")
        
        # Model Comparison
        if "model_comparison" in evaluation_results:
            report.append("MODEL COMPARISON")
            report.append("-" * 40)
            
            comparison = evaluation_results["model_comparison"]
            
            if "rankings" in comparison:
                rankings = comparison["rankings"]
                
                if "by_score" in rankings:
                    report.append("\nRanking by Score:")
                    for i, (model, score) in enumerate(rankings["by_score"], 1):
                        report.append(f"  {i}. {model}: {score:.3f}")
                
                if "by_speed" in rankings:
                    report.append("\nRanking by Speed:")
                    for i, (model, speed) in enumerate(rankings["by_speed"], 1):
                        report.append(f"  {i}. {model}: {speed:.1f} ms")
                
                if "by_cost" in rankings:
                    report.append("\nRanking by Cost:")
                    for i, (model, cost) in enumerate(rankings["by_cost"], 1):
                        report.append(f"  {i}. {model}: ${cost:.4f}")
                
                if "by_value" in rankings:
                    report.append("\nRanking by Value (Score/Cost):")
                    for i, (model, value) in enumerate(rankings["by_value"], 1):
                        report.append(f"  {i}. {model}: {value:.2f}")
            
            report.append("")
        
        # Failed Tests
        if "test_runs" in evaluation_results:
            failed_tests = [
                run for run in evaluation_results["test_runs"]
                if not run.get("success", False)
            ]
            
            if failed_tests:
                report.append("FAILED TESTS")
                report.append("-" * 40)
                
                for test in failed_tests[:10]:  # Show first 10
                    report.append(f"Test Case: {test.get('test_case_name', 'Unknown')}")
                    report.append(f"  Error: {test.get('error', 'Unknown error')}")
                    report.append("")
                
                if len(failed_tests) > 10:
                    report.append(f"... and {len(failed_tests) - 10} more failed tests")
                    report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    ####################################################################################################################
    # CSV Reports
    
    def generate_csv_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate CSV report of test runs.
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            CSV string
        """
        output = StringIO()
        
        if "test_runs" not in evaluation_results:
            return ""
        
        test_runs = evaluation_results["test_runs"]
        if not test_runs:
            return ""
        
        # Determine all fields
        fieldnames = [
            "test_case_id", "test_case_name", "model", "provider",
            "success", "error", "execution_time_ms", "tokens_used",
            "cost_estimate"
        ]
        
        # Add score fields
        all_score_fields = set()
        for run in test_runs:
            if "scores" in run:
                all_score_fields.update(run["scores"].keys())
        
        for field in sorted(all_score_fields):
            fieldnames.append(f"score_{field}")
        
        # Write CSV
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for run in test_runs:
            row = {
                "test_case_id": run.get("test_case_id"),
                "test_case_name": run.get("test_case_name"),
                "model": run.get("model"),
                "provider": run.get("provider"),
                "success": "Yes" if run.get("success") else "No",
                "error": run.get("error", ""),
                "execution_time_ms": run.get("execution_time_ms", 0),
                "tokens_used": run.get("tokens_used", 0),
                "cost_estimate": run.get("cost_estimate", 0)
            }
            
            # Add scores
            for field in all_score_fields:
                row[f"score_{field}"] = run.get("scores", {}).get(field, "")
            
            writer.writerow(row)
        
        return output.getvalue()
    
    ####################################################################################################################
    # JSON Reports
    
    def generate_json_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate detailed JSON report.
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            JSON string
        """
        # Create structured report
        report = {
            "metadata": {
                "evaluation_id": evaluation_results.get("evaluation_id"),
                "generated_at": datetime.utcnow().isoformat(),
                "status": evaluation_results.get("status"),
                "execution_time_seconds": evaluation_results.get("execution_time_seconds")
            },
            "summary": {
                "total_tests": evaluation_results.get("total_tests", 0),
                "successful_tests": evaluation_results.get("successful_tests", 0),
                "success_rate": evaluation_results.get("success_rate", 0),
                "overall_score": evaluation_results.get("aggregated_metrics", {}).get("overall_score", 0)
            },
            "metrics": evaluation_results.get("aggregated_metrics", {}),
            "model_comparison": evaluation_results.get("model_comparison", {}),
            "test_runs": []
        }
        
        # Add test run details (simplified)
        if "test_runs" in evaluation_results:
            for run in evaluation_results["test_runs"]:
                report["test_runs"].append({
                    "test_case": run.get("test_case_name"),
                    "model": run.get("model"),
                    "provider": run.get("provider"),
                    "success": run.get("success"),
                    "scores": run.get("scores", {}),
                    "execution_time_ms": run.get("execution_time_ms"),
                    "error": run.get("error")
                })
        
        return json.dumps(report, indent=2)
    
    ####################################################################################################################
    # Visualization Reports
    
    def generate_visual_report(self, evaluation_results: Dict[str, Any],
                              output_path: str = "evaluation_report.pdf"):
        """
        Generate visual report with charts.
        
        Args:
            evaluation_results: Evaluation results
            output_path: Path to save PDF report
        """
        try:
            with PdfPages(output_path) as pdf:
                # Page 1: Summary
                self._create_summary_page(evaluation_results, pdf)
                
                # Page 2: Score Distribution
                if "test_runs" in evaluation_results:
                    self._create_score_distribution_page(evaluation_results, pdf)
                
                # Page 3: Model Comparison
                if "model_comparison" in evaluation_results:
                    self._create_model_comparison_page(evaluation_results, pdf)
                
                # Page 4: Performance Metrics
                self._create_performance_page(evaluation_results, pdf)
                
                # Metadata
                d = pdf.infodict()
                d['Title'] = 'Prompt Studio Evaluation Report'
                d['Author'] = 'Prompt Studio'
                d['Subject'] = f"Evaluation {evaluation_results.get('evaluation_id', 'N/A')}"
                d['Keywords'] = 'Prompt Engineering, Evaluation, LLM'
                d['CreationDate'] = datetime.utcnow()
            
            logger.info(f"Visual report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate visual report: {e}")
    
    def _create_summary_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create summary page for visual report."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Title
        fig.text(0.5, 0.95, 'Evaluation Report', 
                ha='center', size=20, weight='bold')
        
        # Summary stats
        summary_text = [
            f"Evaluation ID: {results.get('evaluation_id', 'N/A')}",
            f"Status: {results.get('status', 'Unknown')}",
            f"Total Tests: {results.get('total_tests', 0)}",
            f"Success Rate: {results.get('success_rate', 0):.1f}%",
            f"Overall Score: {results.get('aggregated_metrics', {}).get('overall_score', 0):.3f}",
            f"Execution Time: {results.get('execution_time_seconds', 0):.2f} seconds"
        ]
        
        y_pos = 0.85
        for line in summary_text:
            fig.text(0.1, y_pos, line, size=12)
            y_pos -= 0.05
        
        # Success rate pie chart
        ax = fig.add_subplot(2, 1, 2)
        successful = results.get('successful_tests', 0)
        failed = results.get('total_tests', 0) - successful
        
        if successful + failed > 0:
            sizes = [successful, failed]
            labels = ['Successful', 'Failed']
            colors = ['#4CAF50', '#F44336']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90)
            ax.set_title('Test Success Rate')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_score_distribution_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create score distribution visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Score Distributions', size=16, weight='bold')
        
        test_runs = results.get("test_runs", [])
        
        # Collect all scores
        all_scores = {}
        for run in test_runs:
            if "scores" in run:
                for metric, score in run["scores"].items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)
        
        # Plot distributions
        for idx, (metric, scores) in enumerate(list(all_scores.items())[:4]):
            ax = axes[idx // 2, idx % 2]
            
            if scores:
                ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(np.mean(scores), color='red', linestyle='dashed',
                          linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                ax.set_title(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(all_scores), 4):
            axes[idx // 2, idx % 2].set_visible(False)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_model_comparison_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create model comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Model Comparison', size=16, weight='bold')
        
        comparison = results.get("model_comparison", {})
        
        if "rankings" in comparison:
            rankings = comparison["rankings"]
            
            # Score comparison
            if "by_score" in rankings:
                ax = axes[0, 0]
                models = [m for m, _ in rankings["by_score"]]
                scores = [s for _, s in rankings["by_score"]]
                
                ax.barh(models, scores, color='#2196F3')
                ax.set_xlabel('Score')
                ax.set_title('Model Scores')
                ax.grid(True, alpha=0.3)
            
            # Speed comparison
            if "by_speed" in rankings:
                ax = axes[0, 1]
                models = [m for m, _ in rankings["by_speed"]]
                speeds = [s for _, s in rankings["by_speed"]]
                
                ax.barh(models, speeds, color='#FF9800')
                ax.set_xlabel('Execution Time (ms)')
                ax.set_title('Model Speed')
                ax.grid(True, alpha=0.3)
            
            # Cost comparison
            if "by_cost" in rankings:
                ax = axes[1, 0]
                models = [m for m, _ in rankings["by_cost"]]
                costs = [c for _, c in rankings["by_cost"]]
                
                ax.barh(models, costs, color='#4CAF50')
                ax.set_xlabel('Cost ($)')
                ax.set_title('Model Cost')
                ax.grid(True, alpha=0.3)
            
            # Value comparison
            if "by_value" in rankings:
                ax = axes[1, 1]
                models = [m for m, _ in rankings["by_value"]]
                values = [v for _, v in rankings["by_value"]]
                
                ax.barh(models, values, color='#9C27B0')
                ax.set_xlabel('Value (Score/Cost)')
                ax.set_title('Model Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_performance_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create performance metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Performance Metrics', size=16, weight='bold')
        
        metrics = results.get("aggregated_metrics", {})
        
        # Execution time distribution
        if "execution_time" in metrics:
            ax = axes[0, 0]
            exec_stats = metrics["execution_time"]
            
            categories = ['Mean', 'Min', 'Max']
            values = [
                exec_stats.get('mean_ms', 0),
                exec_stats.get('min_ms', 0),
                exec_stats.get('max_ms', 0)
            ]
            
            ax.bar(categories, values, color=['#2196F3', '#4CAF50', '#F44336'])
            ax.set_ylabel('Time (ms)')
            ax.set_title('Execution Time Statistics')
            ax.grid(True, alpha=0.3)
        
        # Token usage
        if "tokens" in metrics:
            ax = axes[0, 1]
            token_stats = metrics["tokens"]
            
            ax.text(0.5, 0.5, f"Total Tokens: {token_stats.get('total', 0):.0f}\n"
                           f"Mean Tokens: {token_stats.get('mean', 0):.0f}",
                   ha='center', va='center', size=14)
            ax.set_title('Token Usage')
            ax.axis('off')
        
        # Cost breakdown
        if "cost" in metrics:
            ax = axes[1, 0]
            cost_stats = metrics["cost"]
            
            ax.text(0.5, 0.5, f"Total Cost: ${cost_stats.get('total', 0):.4f}\n"
                           f"Mean Cost: ${cost_stats.get('mean', 0):.4f}",
                   ha='center', va='center', size=14)
            ax.set_title('Cost Analysis')
            ax.axis('off')
        
        # Success/Failure breakdown
        ax = axes[1, 1]
        test_runs = results.get("test_runs", [])
        if test_runs:
            by_model = {}
            for run in test_runs:
                model = run.get("model", "Unknown")
                if model not in by_model:
                    by_model[model] = {"success": 0, "failure": 0}
                
                if run.get("success"):
                    by_model[model]["success"] += 1
                else:
                    by_model[model]["failure"] += 1
            
            models = list(by_model.keys())
            success_counts = [by_model[m]["success"] for m in models]
            failure_counts = [by_model[m]["failure"] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, success_counts, width, label='Success', color='#4CAF50')
            ax.bar(x + width/2, failure_counts, width, label='Failure', color='#F44336')
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Count')
            ax.set_title('Success/Failure by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

########################################################################################################################
# Report Manager

class ReportManager:
    """Manages report generation and storage."""
    
    def __init__(self, db):
        """
        Initialize ReportManager.
        
        Args:
            db: Database instance
        """
        self.db = db
        self.generator = EvaluationReportGenerator()
    
    def generate_report(self, evaluation_id: int, format: str = "text") -> str:
        """
        Generate a report for an evaluation.
        
        Args:
            evaluation_id: Evaluation ID
            format: Report format (text, csv, json, pdf)
            
        Returns:
            Report content or path for PDF
        """
        # Get evaluation results
        results = self._get_evaluation_results(evaluation_id)
        if not results:
            raise ValueError(f"Evaluation {evaluation_id} not found")
        
        # Generate report based on format
        if format == "text":
            return self.generator.generate_text_report(results)
        elif format == "csv":
            return self.generator.generate_csv_report(results)
        elif format == "json":
            return self.generator.generate_json_report(results)
        elif format == "pdf":
            output_path = f"evaluation_{evaluation_id}_report.pdf"
            self.generator.generate_visual_report(results, output_path)
            return output_path
        else:
            raise ValueError(f"Unknown report format: {format}")
    
    def _get_evaluation_results(self, evaluation_id: int) -> Optional[Dict[str, Any]]:
        """Get evaluation results from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get evaluation
        cursor.execute("""
            SELECT * FROM prompt_studio_evaluations
            WHERE id = ?
        """, (evaluation_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        evaluation = self.db._row_to_dict(cursor, row)
        
        # Parse JSON fields
        test_run_ids = json.loads(evaluation.get("test_run_ids", "[]"))
        aggregate_metrics = json.loads(evaluation.get("aggregate_metrics", "{}"))
        
        # Get test runs
        test_runs = []
        if test_run_ids:
            cursor.execute(f"""
                SELECT * FROM prompt_studio_test_runs
                WHERE id IN ({','.join('?' * len(test_run_ids))})
            """, test_run_ids)
            
            for row in cursor.fetchall():
                run = self.db._row_to_dict(cursor, row)
                # Parse JSON fields
                run["inputs"] = json.loads(run.get("inputs", "{}"))
                run["outputs"] = json.loads(run.get("outputs", "{}"))
                run["expected_outputs"] = json.loads(run.get("expected_outputs", "{}"))
                run["scores"] = json.loads(run.get("scores", "{}"))
                test_runs.append(run)
        
        # Build results
        results = {
            "evaluation_id": evaluation_id,
            "status": evaluation.get("status"),
            "total_tests": len(test_runs),
            "successful_tests": sum(1 for run in test_runs if not run.get("error_message")),
            "success_rate": (sum(1 for run in test_runs if not run.get("error_message")) / len(test_runs) * 100) if test_runs else 0,
            "test_runs": test_runs,
            "aggregated_metrics": aggregate_metrics,
            "completed_at": evaluation.get("completed_at")
        }
        
        return results