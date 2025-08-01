#!/usr/bin/env python3
"""
RAG Test Runner for NewsHead

This script runs all RAG tests in sequence and provides a comprehensive
summary report to help decide whether to integrate RAG into production.

Usage:
    python3 tests/run_all_tests.py --full-suite
    python3 tests/run_all_tests.py --quick-test --sample-size 20
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGTestRunner:
    """Comprehensive test runner for RAG system validation"""
    
    def __init__(self, sample_size: int = 50):
        self.sample_size = sample_size
        self.test_results = {}
        self.start_time = datetime.now()
        
    async def run_embedding_tests(self) -> Dict[str, Any]:
        """Run embedding generation and similarity tests"""
        logger.info("🔬 Running Embedding Tests...")
        
        try:
            # Run embedding tests
            cmd = [
                "python3", "tests/rag_embedding_test.py",
                "--sample-size", str(self.sample_size),
                "--test-similarity",
                "--save-results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info("✅ Embedding tests completed successfully")
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"❌ Embedding tests failed: {result.stderr}")
                return {
                    "status": "failed", 
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Error running embedding tests: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_similarity_tests(self) -> Dict[str, Any]:
        """Run similarity search validation tests"""
        logger.info("🔍 Running Similarity Tests...")
        
        try:
            cmd = [
                "python3", "tests/rag_similarity_test.py",
                "--test-outcome-correlation",
                "--validate-precision",
                "--save-results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info("✅ Similarity tests completed successfully")
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"❌ Similarity tests failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Error running similarity tests: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_baseline_comparison(self) -> Dict[str, Any]:
        """Run baseline traditional sentiment analysis test"""
        logger.info("📊 Running Baseline Comparison Test...")
        
        try:
            cmd = [
                "python3", "tests/rag_comparison_test.py",
                "--test-mode", "traditional",
                "--sample-size", str(self.sample_size),
                "--save-results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info("✅ Baseline comparison completed successfully")
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"❌ Baseline comparison failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Error running baseline comparison: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_rag_comparison(self) -> Dict[str, Any]:
        """Run RAG vs traditional comparison test"""
        logger.info("🤖 Running RAG Comparison Test...")
        
        try:
            cmd = [
                "python3", "tests/rag_comparison_test.py",
                "--test-mode", "parallel",
                "--sample-size", str(self.sample_size),
                "--save-results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info("✅ RAG comparison completed successfully")
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logger.error(f"❌ RAG comparison failed: {result.stderr}")
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            logger.error(f"Error running RAG comparison: {e}")
            return {"status": "error", "error": str(e)}
    
    def parse_test_results(self) -> Dict[str, Any]:
        """Parse and analyze all test results"""
        logger.info("📈 Analyzing test results...")
        
        # Look for the most recent test result files
        results_dir = "tests/results"
        if not os.path.exists(results_dir):
            logger.warning("No results directory found")
            return {"error": "No test results found"}
        
        # Find most recent result files
        result_files = os.listdir(results_dir)
        
        # Parse comparison results
        comparison_files = [f for f in result_files if f.startswith("comparison_results_")]
        if comparison_files:
            latest_comparison = max(comparison_files)
            with open(os.path.join(results_dir, latest_comparison), 'r') as f:
                comparison_data = json.load(f)
        else:
            comparison_data = None
        
        # Parse performance metrics
        performance_files = [f for f in result_files if f.startswith("performance_metrics_")]
        if performance_files:
            latest_performance = max(performance_files)
            with open(os.path.join(results_dir, latest_performance), 'r') as f:
                performance_data = json.load(f)
        else:
            performance_data = None
        
        # Parse similarity results
        similarity_files = [f for f in result_files if f.startswith("similarity_test_results_")]
        if similarity_files:
            latest_similarity = max(similarity_files) 
            with open(os.path.join(results_dir, latest_similarity), 'r') as f:
                similarity_data = json.load(f)
        else:
            similarity_data = None
        
        # Parse embedding results
        embedding_files = [f for f in result_files if f.startswith("embedding_test_results_")]
        if embedding_files:
            latest_embedding = max(embedding_files)
            with open(os.path.join(results_dir, latest_embedding), 'r') as f:
                embedding_data = json.load(f)
        else:
            embedding_data = None
        
        return {
            "comparison_results": comparison_data,
            "performance_metrics": performance_data,
            "similarity_results": similarity_data,
            "embedding_results": embedding_data
        }
    
    def generate_integration_recommendation(self, parsed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation on whether to integrate RAG"""
        
        recommendation = {
            "integrate_rag": False,
            "confidence": "low",
            "reasons": [],
            "concerns": [],
            "next_steps": []
        }
        
        # Check performance metrics
        performance = parsed_results.get("performance_metrics", {})
        if performance:
            accuracy_metrics = performance.get("accuracy_metrics", {})
            trading_metrics = performance.get("trading_metrics", {})
            performance_metrics = performance.get("performance_metrics", {})
            
            # Check accuracy improvement
            accuracy_improvement = accuracy_metrics.get("accuracy_improvement", 0)
            if accuracy_improvement > 0.10:  # 10% improvement
                recommendation["reasons"].append(f"Strong accuracy improvement: {accuracy_improvement:.1%}")
            elif accuracy_improvement > 0.05:  # 5% improvement
                recommendation["reasons"].append(f"Moderate accuracy improvement: {accuracy_improvement:.1%}")
            else:
                recommendation["concerns"].append(f"Minimal accuracy improvement: {accuracy_improvement:.1%}")
            
            # Check BUY+high precision improvement
            precision_improvement = trading_metrics.get("buy_high_precision_improvement", 0)
            if precision_improvement > 0.15:  # 15% improvement
                recommendation["reasons"].append(f"Excellent BUY+high precision improvement: {precision_improvement:.1%}")
            elif precision_improvement > 0.05:  # 5% improvement
                recommendation["reasons"].append(f"Good BUY+high precision improvement: {precision_improvement:.1%}")
            else:
                recommendation["concerns"].append(f"Poor BUY+high precision improvement: {precision_improvement:.1%}")
            
            # Check performance impact
            time_overhead = performance_metrics.get("analysis_time_overhead", 0)
            if time_overhead < 0.3:  # Less than 300ms
                recommendation["reasons"].append(f"Acceptable performance overhead: {time_overhead:.2f}s")
            elif time_overhead < 0.5:  # Less than 500ms
                recommendation["concerns"].append(f"High but acceptable performance overhead: {time_overhead:.2f}s")
            else:
                recommendation["concerns"].append(f"Unacceptable performance overhead: {time_overhead:.2f}s")
        
        # Check similarity results
        similarity = parsed_results.get("similarity_results", {})
        if similarity:
            rag_validation = similarity.get("rag_precision_validation", {})
            if rag_validation:
                analysis = rag_validation.get("cross_contamination_analysis", {})
                separation_quality = analysis.get("overall_separation_quality", 0)
                contamination_risk = analysis.get("overall_contamination_risk", 0)
                
                if separation_quality > 0.7:  # Good separation
                    recommendation["reasons"].append(f"Good outcome separation quality: {separation_quality:.2f}")
                else:
                    recommendation["concerns"].append(f"Poor outcome separation quality: {separation_quality:.2f}")
                
                if contamination_risk < 0.3:  # Low contamination
                    recommendation["reasons"].append(f"Low contamination risk: {contamination_risk:.2f}")
                else:
                    recommendation["concerns"].append(f"High contamination risk: {contamination_risk:.2f}")
        
        # Make final recommendation
        positive_signals = len(recommendation["reasons"])
        negative_signals = len(recommendation["concerns"])
        
        if positive_signals >= 3 and negative_signals <= 1:
            recommendation["integrate_rag"] = True
            recommendation["confidence"] = "high"
            recommendation["next_steps"].append("Proceed with RAG integration into sentiment_service.py")
            recommendation["next_steps"].append("Deploy with gradual rollout (10% → 50% → 100%)")
            recommendation["next_steps"].append("Monitor production performance closely")
        elif positive_signals >= 2:
            recommendation["integrate_rag"] = True
            recommendation["confidence"] = "medium"
            recommendation["next_steps"].append("Proceed with cautious RAG integration")
            recommendation["next_steps"].append("Start with 10% traffic and monitor carefully")
            recommendation["next_steps"].append("Consider tuning similarity thresholds")
        else:
            recommendation["integrate_rag"] = False
            recommendation["confidence"] = "high"
            recommendation["next_steps"].append("Do not integrate RAG at this time")
            recommendation["next_steps"].append("Investigate embedding quality and similarity algorithms")
            recommendation["next_steps"].append("Consider collecting more labeled training data")
        
        return recommendation
    
    async def run_all_tests(self, test_suite: str = "full") -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🚀 Starting RAG Test Suite...")
        logger.info(f"📊 Sample size: {self.sample_size}")
        logger.info(f"🎯 Test suite: {test_suite}")
        
        all_results = {
            "start_time": self.start_time.isoformat(),
            "sample_size": self.sample_size,
            "test_suite": test_suite
        }
        
        if test_suite in ["full", "embedding"]:
            # Run embedding tests
            all_results["embedding_tests"] = await self.run_embedding_tests()
            
        if test_suite in ["full", "similarity"]:
            # Run similarity tests
            all_results["similarity_tests"] = await self.run_similarity_tests()
        
        if test_suite in ["full", "comparison"]:
            # Run baseline comparison
            all_results["baseline_tests"] = await self.run_baseline_comparison()
            
            # Run RAG comparison
            all_results["rag_comparison_tests"] = await self.run_rag_comparison()
        
        # Parse and analyze results
        parsed_results = self.parse_test_results()
        all_results["parsed_results"] = parsed_results
        
        # Generate integration recommendation
        recommendation = self.generate_integration_recommendation(parsed_results)
        all_results["integration_recommendation"] = recommendation
        
        # Calculate total test time
        end_time = datetime.now()
        all_results["end_time"] = end_time.isoformat()
        all_results["total_duration"] = (end_time - self.start_time).total_seconds()
        
        return all_results
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print comprehensive summary report"""
        
        print("\n" + "="*100)
        print("🎯 RAG TEST SUITE SUMMARY REPORT")
        print("="*100)
        
        # Test execution summary
        print(f"📅 Test Date: {results['start_time']}")
        print(f"⏱️ Total Duration: {results['total_duration']:.1f} seconds")
        print(f"📊 Sample Size: {results['sample_size']}")
        print(f"🧪 Test Suite: {results['test_suite']}")
        
        # Test results summary
        print(f"\n📋 TEST EXECUTION STATUS:")
        for test_type, test_result in results.items():
            if test_type.endswith("_tests") and isinstance(test_result, dict):
                status = test_result.get("status", "unknown")
                status_emoji = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
                print(f"  {status_emoji} {test_type.replace('_', ' ').title()}: {status}")
        
        # Performance metrics
        parsed = results.get("parsed_results", {})
        performance = parsed.get("performance_metrics", {})
        
        if performance:
            print(f"\n📈 PERFORMANCE METRICS:")
            
            accuracy_metrics = performance.get("accuracy_metrics", {})
            if accuracy_metrics:
                print(f"  🔍 Traditional Accuracy: {accuracy_metrics.get('traditional_overall_accuracy', 0):.3f}")
                print(f"  🤖 RAG Accuracy: {accuracy_metrics.get('rag_overall_accuracy', 0):.3f}")
                print(f"  📊 Accuracy Improvement: {accuracy_metrics.get('accuracy_improvement', 0):.3f}")
            
            trading_metrics = performance.get("trading_metrics", {})
            if trading_metrics:
                print(f"  🎯 Traditional BUY+high Precision: {trading_metrics.get('traditional_buy_high_precision', 0):.3f}")
                print(f"  🎯 RAG BUY+high Precision: {trading_metrics.get('rag_buy_high_precision', 0):.3f}")
                print(f"  📈 Precision Improvement: {trading_metrics.get('buy_high_precision_improvement', 0):.3f}")
            
            performance_metrics = performance.get("performance_metrics", {})
            if performance_metrics:
                print(f"  ⚡ Traditional Analysis Time: {performance_metrics.get('traditional_avg_analysis_time', 0):.2f}s")
                print(f"  🚀 RAG Analysis Time: {performance_metrics.get('rag_avg_analysis_time', 0):.2f}s")
                print(f"  ⏱️ Time Overhead: {performance_metrics.get('analysis_time_overhead', 0):.2f}s")
        
        # Integration recommendation
        recommendation = results.get("integration_recommendation", {})
        if recommendation:
            print(f"\n🎯 INTEGRATION RECOMMENDATION:")
            
            integrate = recommendation.get("integrate_rag", False)
            confidence = recommendation.get("confidence", "unknown")
            
            recommendation_emoji = "✅" if integrate else "❌"
            confidence_emoji = "🔥" if confidence == "high" else "⚡" if confidence == "medium" else "⚠️"
            
            print(f"  {recommendation_emoji} Integrate RAG: {integrate}")
            print(f"  {confidence_emoji} Confidence: {confidence}")
            
            reasons = recommendation.get("reasons", [])
            if reasons:
                print(f"  ✅ Positive Factors:")
                for reason in reasons:
                    print(f"    • {reason}")
            
            concerns = recommendation.get("concerns", [])
            if concerns:
                print(f"  ⚠️ Concerns:")
                for concern in concerns:
                    print(f"    • {concern}")
            
            next_steps = recommendation.get("next_steps", [])
            if next_steps:
                print(f"  📋 Next Steps:")
                for step in next_steps:
                    print(f"    • {step}")
        
        print("="*100)
        
        # Save summary report
        os.makedirs("tests/results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f"tests/results/test_suite_summary_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Full results saved to: tests/results/test_suite_summary_{timestamp}.json")

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='RAG Test Suite Runner')
    parser.add_argument('--sample-size', type=int, default=50, help='Number of articles to test')
    parser.add_argument('--test-suite', choices=['full', 'quick', 'embedding', 'similarity', 'comparison'], 
                       default='full', help='Test suite to run')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with reduced sample size')
    parser.add_argument('--full-suite', action='store_true', help='Run complete test suite')
    
    args = parser.parse_args()
    
    # Adjust parameters based on flags
    if args.quick_test:
        sample_size = min(args.sample_size, 20)
        test_suite = 'quick'
    elif args.full_suite:
        sample_size = max(args.sample_size, 100)
        test_suite = 'full'
    else:
        sample_size = args.sample_size
        test_suite = args.test_suite
    
    # Create test runner
    runner = RAGTestRunner(sample_size=sample_size)
    
    try:
        # Run all tests
        results = await runner.run_all_tests(test_suite=test_suite)
        
        # Print summary report
        runner.print_summary_report(results)
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 