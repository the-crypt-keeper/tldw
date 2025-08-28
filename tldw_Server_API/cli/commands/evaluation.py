"""
Evaluation execution commands for tldw Evaluations CLI.

Provides command-line access to all evaluation types including
G-Eval, RAG evaluation, response quality assessment, and batch processing.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_evaluation_results,
    print_json, print_table, print_progress_bar, format_timestamp
)


@click.group()
def eval_group():
    """Evaluation execution commands."""
    pass


@eval_group.command('geval')
@click.argument('text', required=True)
@click.argument('summary', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def geval(ctx, text, summary, provider, model, api_key, save, output_format):
    """
    Run G-Eval summarization assessment.
    
    Evaluates the quality of a summary against the original text using
    G-Eval methodology with multiple criteria including relevance,
    consistency, fluency, and coherence.
    
    Arguments:
        TEXT    Original text to evaluate against
        SUMMARY Summary text to evaluate
    
    Examples:
        tldw-evals eval geval "Original text..." "Summary text..."
        tldw-evals eval geval "file://input.txt" "file://summary.txt"
        tldw-evals eval geval "text" "summary" --provider anthropic --no-save
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load text from files if needed
        text_content = _load_text_content(text)
        summary_content = _load_text_content(summary)
        
        if not cli_context.quiet:
            print_info(f"Running G-Eval assessment with {provider}...")
        
        # Run evaluation
        result = _run_geval_assessment(
            text_content, summary_content, provider, model, api_key, save, cli_context.config
        )
        
        # Display results
        if output_format == 'json':
            print_json(result, "G-Eval Results")
        else:
            print_evaluation_results(result)
        
        print_success("G-Eval assessment completed successfully")
        
    except Exception as e:
        logger.exception("G-Eval assessment failed")
        print_error(f"G-Eval assessment failed: {e}")
        sys.exit(1)


@eval_group.command('rag')
@click.argument('query', required=True)
@click.argument('context', required=True)
@click.argument('response', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def rag_eval(ctx, query, context, response, provider, model, api_key, save, output_format):
    """
    Run RAG (Retrieval-Augmented Generation) evaluation.
    
    Evaluates RAG system performance by assessing context relevance,
    answer faithfulness, answer relevance, and overall quality.
    
    Arguments:
        QUERY    User query/question
        CONTEXT  Retrieved context/documents
        RESPONSE Generated response/answer
    
    Examples:
        tldw-evals eval rag "What is AI?" "Context about AI..." "AI response..."
        tldw-evals eval rag "file://query.txt" "file://context.txt" "file://response.txt"
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load content from files if needed
        query_content = _load_text_content(query)
        context_content = _load_text_content(context)
        response_content = _load_text_content(response)
        
        if not cli_context.quiet:
            print_info(f"Running RAG evaluation with {provider}...")
        
        # Run evaluation
        result = _run_rag_evaluation(
            query_content, context_content, response_content,
            provider, model, api_key, save, cli_context.config
        )
        
        # Display results
        if output_format == 'json':
            print_json(result, "RAG Evaluation Results")
        else:
            print_evaluation_results(result)
        
        print_success("RAG evaluation completed successfully")
        
    except Exception as e:
        logger.exception("RAG evaluation failed")
        print_error(f"RAG evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('quality')
@click.argument('text', required=True)
@click.argument('response', required=True)
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--criteria', multiple=True, help='Specific quality criteria to evaluate')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def quality_eval(ctx, text, response, provider, model, api_key, criteria, save, output_format):
    """
    Run response quality evaluation.
    
    Evaluates response quality across multiple dimensions including
    accuracy, completeness, clarity, relevance, and helpfulness.
    
    Arguments:
        TEXT     Original text/prompt
        RESPONSE Response to evaluate
    
    Examples:
        tldw-evals eval quality "Question..." "Answer..."
        tldw-evals eval quality "text" "response" --criteria accuracy --criteria clarity
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load content from files if needed
        text_content = _load_text_content(text)
        response_content = _load_text_content(response)
        
        if not cli_context.quiet:
            print_info(f"Running response quality evaluation with {provider}...")
        
        # Run evaluation
        result = _run_quality_evaluation(
            text_content, response_content, provider, model, api_key,
            list(criteria) if criteria else None, save, cli_context.config
        )
        
        # Display results
        if output_format == 'json':
            print_json(result, "Quality Evaluation Results")
        else:
            print_evaluation_results(result)
        
        print_success("Quality evaluation completed successfully")
        
    except Exception as e:
        logger.exception("Quality evaluation failed")
        print_error(f"Quality evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('batch')
@click.argument('input_file', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('--output', type=click.Path(path_type=Path), help='Output file for results')
@click.option('--provider', default='openai', help='Default LLM provider to use')
@click.option('--model', help='Default model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--parallel', type=int, default=1, help='Number of parallel evaluations')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def batch_eval(ctx, input_file, output, provider, model, api_key, parallel, output_format):
    """
    Run batch evaluations from input file.
    
    Processes multiple evaluations from a JSON or JSONL input file.
    Each line/entry should contain evaluation parameters.
    
    Input file format (JSONL):
        {"type": "geval", "text": "...", "summary": "...", "provider": "openai"}
        {"type": "rag", "query": "...", "context": "...", "response": "..."}
        {"type": "quality", "text": "...", "response": "...", "criteria": ["accuracy"]}
    
    Examples:
        tldw-evals eval batch evaluations.jsonl
        tldw-evals eval batch batch_input.json --output results.json --parallel 3
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load batch input
        batch_data = _load_batch_input(input_file)
        
        if not cli_context.quiet:
            print_info(f"Processing {len(batch_data)} evaluations with {parallel} parallel workers...")
        
        # Run batch evaluation
        results = _run_batch_evaluation(
            batch_data, provider, model, api_key, parallel, cli_context.config, not cli_context.quiet
        )
        
        # Save results if output file specified
        if output:
            _save_batch_results(results, output)
            if not cli_context.quiet:
                print_success(f"Results saved to {output}")
        
        # Display summary
        if output_format == 'json':
            print_json(results, "Batch Evaluation Results")
        else:
            _display_batch_summary(results)
        
        print_success(f"Batch evaluation completed: {len(results)} evaluations processed")
        
    except Exception as e:
        logger.exception("Batch evaluation failed")
        print_error(f"Batch evaluation failed: {e}")
        sys.exit(1)


@eval_group.command('custom')
@click.argument('metric_name', required=True)
@click.argument('text', required=True)
@click.option('--prompt', help='Custom evaluation prompt')
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', help='Specific model to use')
@click.option('--api-key', help='API key (overrides config)')
@click.option('--save/--no-save', default=True, help='Save evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def custom_eval(ctx, metric_name, text, prompt, provider, model, api_key, save, output_format):
    """
    Run custom metric evaluation.
    
    Evaluates text using a custom metric with optional custom prompt.
    Useful for domain-specific or specialized evaluation criteria.
    
    Arguments:
        METRIC_NAME Name of the custom metric
        TEXT        Text to evaluate
    
    Examples:
        tldw-evals eval custom "creativity" "Story text..." --prompt "Rate creativity 1-10"
        tldw-evals eval custom "technical_accuracy" "file://technical_doc.txt"
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load text content
        text_content = _load_text_content(text)
        
        if not cli_context.quiet:
            print_info(f"Running custom evaluation '{metric_name}' with {provider}...")
        
        # Run evaluation
        result = _run_custom_evaluation(
            metric_name, text_content, prompt, provider, model, api_key, save, cli_context.config
        )
        
        # Display results
        if output_format == 'json':
            print_json(result, f"Custom Evaluation Results: {metric_name}")
        else:
            print_evaluation_results(result)
        
        print_success(f"Custom evaluation '{metric_name}' completed successfully")
        
    except Exception as e:
        logger.exception("Custom evaluation failed")
        print_error(f"Custom evaluation failed: {e}")
        sys.exit(1)


def _load_text_content(input_str: str) -> str:
    """Load text content from string or file."""
    if input_str.startswith('file://'):
        file_path = Path(input_str[7:])  # Remove 'file://' prefix
        if not file_path.exists():
            raise click.ClickException(f"File not found: {file_path}")
        
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise click.ClickException(f"Failed to read file {file_path}: {e}")
    else:
        return input_str


def _run_geval_assessment(text: str, summary: str, provider: str, model: str, 
                         api_key: str, save: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run G-Eval assessment."""
    try:
        from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
        
        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)
        
        result = run_geval(
            transcript=text,
            summary=summary,
            api_key=api_key,
            api_name=provider,
            save=save
        )
        
        return {
            'evaluation_type': 'geval',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }
        
    except Exception as e:
        logger.error(f"G-Eval execution failed: {e}")
        raise


def _run_rag_evaluation(query: str, context: str, response: str, provider: str, 
                       model: str, api_key: str, save: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run RAG evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
        
        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)
        
        evaluator = RAGEvaluator(api_key=api_key, api_name=provider)
        
        result = asyncio.run(evaluator.evaluate(
            query=query,
            contexts=[context],  # RAGEvaluator expects a list
            response=response,
            save_results=save
        ))
        
        return {
            'evaluation_type': 'rag',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }
        
    except Exception as e:
        logger.error(f"RAG evaluation execution failed: {e}")
        raise


def _run_quality_evaluation(text: str, response: str, provider: str, model: str,
                           api_key: str, criteria: Optional[List[str]], save: bool, 
                           config: Dict[str, Any]) -> Dict[str, Any]:
    """Run response quality evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
        
        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)
        
        evaluator = ResponseQualityEvaluator(api_key=api_key, api_name=provider)
        
        result = asyncio.run(evaluator.evaluate(
            original_text=text,
            response_text=response,
            criteria=criteria,
            save_results=save
        ))
        
        return {
            'evaluation_type': 'quality',
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }
        
    except Exception as e:
        logger.error(f"Quality evaluation execution failed: {e}")
        raise


def _run_custom_evaluation(metric_name: str, text: str, prompt: str, provider: str,
                          model: str, api_key: str, save: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run custom metric evaluation."""
    try:
        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
        
        # Extract API key from config if not provided
        if not api_key:
            api_key = _get_api_key_from_config(config, provider)
        
        eval_manager = EvaluationManager()
        
        result = asyncio.run(eval_manager.evaluate_custom_metric(
            text=text,
            metric_name=metric_name,
            custom_prompt=prompt,
            api_key=api_key,
            api_name=provider,
            save_results=save
        ))
        
        return {
            'evaluation_type': 'custom',
            'metric_name': metric_name,
            'evaluation_id': result.get('evaluation_id'),
            'provider': provider,
            'model': model,
            'timestamp': format_timestamp(None),
            'metrics': result.get('metrics', {}),
            'details': result
        }
        
    except Exception as e:
        logger.error(f"Custom evaluation execution failed: {e}")
        raise


def _load_batch_input(input_file: Path) -> List[Dict[str, Any]]:
    """Load batch input from file."""
    try:
        content = input_file.read_text(encoding='utf-8')
        
        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError:
            pass
        
        # Try JSONL
        batch_data = []
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            if line.strip():
                try:
                    batch_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise click.ClickException(f"Invalid JSON on line {line_num}: {e}")
        
        return batch_data
        
    except Exception as e:
        raise click.ClickException(f"Failed to load batch input file: {e}")


def _run_batch_evaluation(batch_data: List[Dict[str, Any]], default_provider: str,
                         default_model: str, default_api_key: str, parallel: int,
                         config: Dict[str, Any], show_progress: bool) -> List[Dict[str, Any]]:
    """Run batch evaluation with progress tracking."""
    results = []
    
    if show_progress:
        with print_progress_bar(len(batch_data), "Evaluating") as progress:
            task = progress.add_task("Processing evaluations...", total=len(batch_data))
            
            for i, eval_spec in enumerate(batch_data):
                try:
                    result = _run_single_evaluation(eval_spec, default_provider, default_model, default_api_key, config)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation {i+1} failed: {e}")
                    results.append({
                        'evaluation_index': i + 1,
                        'status': 'failed',
                        'error': str(e),
                        'input': eval_spec
                    })
                
                progress.update(task, advance=1)
    else:
        for i, eval_spec in enumerate(batch_data):
            try:
                result = _run_single_evaluation(eval_spec, default_provider, default_model, default_api_key, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation {i+1} failed: {e}")
                results.append({
                    'evaluation_index': i + 1,
                    'status': 'failed',
                    'error': str(e),
                    'input': eval_spec
                })
    
    return results


def _run_single_evaluation(eval_spec: Dict[str, Any], default_provider: str,
                          default_model: str, default_api_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single evaluation from specification."""
    eval_type = eval_spec.get('type')
    provider = eval_spec.get('provider', default_provider)
    model = eval_spec.get('model', default_model)
    api_key = eval_spec.get('api_key', default_api_key)
    save = eval_spec.get('save', True)
    
    if eval_type == 'geval':
        return _run_geval_assessment(
            eval_spec['text'], eval_spec['summary'],
            provider, model, api_key, save, config
        )
    elif eval_type == 'rag':
        return _run_rag_evaluation(
            eval_spec['query'], eval_spec['context'], eval_spec['response'],
            provider, model, api_key, save, config
        )
    elif eval_type == 'quality':
        return _run_quality_evaluation(
            eval_spec['text'], eval_spec['response'],
            provider, model, api_key, eval_spec.get('criteria'), save, config
        )
    elif eval_type == 'custom':
        return _run_custom_evaluation(
            eval_spec['metric_name'], eval_spec['text'],
            eval_spec.get('prompt'), provider, model, api_key, save, config
        )
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")


def _save_batch_results(results: List[Dict[str, Any]], output_file: Path):
    """Save batch results to file."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
            
    except Exception as e:
        raise click.ClickException(f"Failed to save results to {output_file}: {e}")


def _display_batch_summary(results: List[Dict[str, Any]]):
    """Display batch evaluation summary."""
    successful = sum(1 for r in results if r.get('status') != 'failed')
    failed = len(results) - successful
    
    summary_data = [
        {'Metric': 'Total Evaluations', 'Count': len(results)},
        {'Metric': 'Successful', 'Count': successful},
        {'Metric': 'Failed', 'Count': failed},
        {'Metric': 'Success Rate', 'Count': f"{successful/len(results)*100:.1f}%" if results else "0%"}
    ]
    
    print_table(summary_data, "Batch Evaluation Summary")
    
    if failed > 0:
        print_info(f"Check logs for details on {failed} failed evaluation(s)")


def _get_api_key_from_config(config: Dict[str, Any], provider: str) -> Optional[str]:
    """Extract API key for provider from config."""
    # This is a simplified version - in reality, you'd access the config properly
    # Based on how the main config system works
    provider_key_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'groq': 'GROQ_API_KEY'
    }
    
    key_name = provider_key_map.get(provider.lower())
    if key_name:
        # Try to get from config
        for section, values in config.items():
            if isinstance(values, dict) and key_name in values:
                return values[key_name]
        
        # Try environment variable
        import os
        return os.getenv(key_name)
    
    return None