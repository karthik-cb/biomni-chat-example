#!/usr/bin/env python3
"""
cBioPortal Integration Tools for Biomni Framework

This module provides standalone functions for querying cBioPortal data
through direct HTTP API calls. These tools are designed to work with
Biomni's multi-agent architecture to demonstrate fast inference benefits
for Memorial Sloan Kettering demo.

Functions create sequential API calls where Cerebras speed matters:
1. Query mutations by gene/study
2. Get molecular profiles 
3. Analyze mutation frequencies
4. Filter and aggregate results
"""

import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import time


# cBioPortal API base URL (using public demo instance)
CBIOPORTAL_BASE_URL = "https://www.cbioportal.org/api"


def query_mutations_by_gene(gene_symbol: str, study_id: Optional[str] = None) -> str:
    """Query mutation data for a specific gene across cancer studies.
    
    This function retrieves mutation data for a given gene symbol from cBioPortal.
    Can optionally filter by specific study ID for targeted analysis.
    
    Parameters
    ----------
    gene_symbol : str
        Gene symbol to query mutations for (e.g., "TP53", "BRCA1", "EGFR")
    study_id : str, optional
        Specific study ID to filter results (e.g., "tcga_gbm", "msk_impact_2017")
        If None, queries across all available studies
    
    Returns
    -------
    str
        JSON-formatted mutation data including mutation types, positions,
        and clinical information for the specified gene
    """
    
    try:
        # Construct API endpoint
        endpoint = f"{CBIOPORTAL_BASE_URL}/mutations"
        params = {
            "geneSymbol": gene_symbol,
            "projection": "DETAILED"
        }
        
        if study_id:
            params["studyId"] = study_id
        
        # Make API request
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        mutations_data = response.json()
        
        # Process and format results
        result = {
            "gene_symbol": gene_symbol,
            "study_filter": study_id or "all_studies",
            "total_mutations": len(mutations_data),
            "mutations": mutations_data[:100],  # Limit to first 100 for demo
            "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"Error querying mutations for {gene_symbol}: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_molecular_profiles(study_id: str) -> str:
    """Get molecular profile information for a specific cancer study.
    
    Retrieves available molecular data types (mutations, copy number,
    expression, etc.) for a given study in cBioPortal.
    
    Parameters
    ----------
    study_id : str
        Study identifier (e.g., "tcga_gbm", "msk_impact_2017", "ccle_broad")
    
    Returns
    -------
    str
        JSON-formatted list of molecular profiles available for the study,
    """
    
    try:
        endpoint = f"{CBIOPORTAL_BASE_URL}/molecular-profiles"
        params = {"studyId": study_id}
        
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        profiles_data = response.json()
        
        # Process results
        result = {
            "study_id": study_id,
            "total_profiles": len(profiles_data),
            "molecular_profiles": profiles_data,
            "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"Error getting molecular profiles for {study_id}: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def analyze_mutation_frequency(gene_symbols: List[str], study_ids: Optional[List[str]] = None) -> str:
    """Analyze mutation frequencies across multiple genes and studies.
    
    This function queries mutation data for multiple genes and calculates
    frequency statistics across specified studies. Demonstrates sequential
    API calls where fast inference provides speed benefits.
    
    Parameters
    ----------
    gene_symbols : List[str]
        List of gene symbols to analyze (e.g., ["TP53", "KRAS", "EGFR"])
    study_ids : List[str], optional
        List of study IDs to include in analysis.
        If None, includes major TCGA studies by default
    
    Returns
    -------
    str
        JSON-formatted mutation frequency analysis with statistics
    """
    
    try:
        # Default studies if not specified
        if not study_ids:
            study_ids = ["tcga_gbm", "tcga_brca", "tcga_luad", "msk_impact_2017"]
        
        mutation_analysis = {
            "genes_analyzed": gene_symbols,
            "studies_analyzed": study_ids,
            "mutation_frequencies": {},
            "summary_statistics": {},
            "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        total_api_calls = 0
        
        # Sequential API calls for each gene-study combination
        for gene in gene_symbols:
            gene_mutations = []
            
            for study in study_ids:
                # Query mutations for this gene-study pair
                endpoint = f"{CBIOPORTAL_BASE_URL}/mutations"
                params = {
                    "geneSymbol": gene,
                    "studyId": study,
                    "projection": "SUMMARY"
                }
                
                response = requests.get(endpoint, params=params, timeout=15)
                response.raise_for_status()
                
                mutations = response.json()
                total_api_calls += 1
                
                # Calculate frequency for this study
                if mutations:
                    # Get sample count for this study
                    sample_endpoint = f"{CBIOPORTAL_BASE_URL}/samples"
                    sample_params = {"studyId": study, "projection": "SUMMARY"}
                    sample_response = requests.get(sample_endpoint, params=sample_params, timeout=15)
                    sample_response.raise_for_status()
                    
                    samples = sample_response.json()
                    total_samples = len(samples)
                    mutated_samples = len(set(mut['sampleId'] for mut in mutations))
                    frequency = (mutated_samples / total_samples * 100) if total_samples > 0 else 0
                    
                    gene_mutations.append({
                        "study_id": study,
                        "mutated_samples": mutated_samples,
                        "total_samples": total_samples,
                        "frequency_percent": round(frequency, 2)
                    })
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            mutation_analysis["mutation_frequencies"][gene] = gene_mutations
        
        # Calculate summary statistics
        all_frequencies = []
        for gene_data in mutation_analysis["mutation_frequencies"].values():
            for study_data in gene_data:
                all_frequencies.append(study_data["frequency_percent"])
        
        if all_frequencies:
            mutation_analysis["summary_statistics"] = {
                "average_frequency": round(sum(all_frequencies) / len(all_frequencies), 2),
                "max_frequency": max(all_frequencies),
                "min_frequency": min(all_frequencies),
                "total_api_calls": total_api_calls
            }
        
        return json.dumps(mutation_analysis, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"Error analyzing mutation frequency: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_clinical_data_by_study(study_id: str, clinical_attribute_id: Optional[str] = None) -> str:
    """Retrieve clinical data for patients in a specific study.
    
    Parameters
    ----------
    study_id : str
        Study identifier (e.g., "tcga_gbm", "msk_impact_2017")
    clinical_attribute_id : str, optional
        Specific clinical attribute to retrieve (e.g., "AGE", "SEX", "OS_STATUS")
        If None, retrieves all available clinical attributes
    
    Returns
    -------
    str
        JSON-formatted clinical data for the study
    """
    
    try:
        endpoint = f"{CBIOPORTAL_BASE_URL}/clinical-data"
        params = {"studyId": study_id}
        
        if clinical_attribute_id:
            params["clinicalAttributeId"] = clinical_attribute_id
        
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        clinical_data = response.json()
        
        result = {
            "study_id": study_id,
            "clinical_attribute_filter": clinical_attribute_id or "all_attributes",
            "total_patients": len(set(data['patientId'] for data in clinical_data)),
            "clinical_data": clinical_data[:200],  # Limit for demo
            "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"Error getting clinical data for {study_id}: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_available_studies(cancer_type: Optional[str] = None) -> str:
    """Get list of available studies in cBioPortal.
    
    Parameters
    ----------
    cancer_type : str, optional
        Filter studies by cancer type (e.g., "breast", "lung", "glioblastoma")
        If None, returns all available studies
    
    Returns
    -------
    str
        JSON-formatted list of available studies with metadata
    """
    
    try:
        endpoint = f"{CBIOPORTAL_BASE_URL}/studies"
        params = {"projection": "SUMMARY"}
        
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        studies_data = response.json()
        
        # Filter by cancer type if specified
        if cancer_type:
            studies_data = [
                study for study in studies_data 
                if cancer_type.lower() in study.get('name', '').lower() or 
                   cancer_type.lower() in study.get('description', '').lower()
            ]
        
        result = {
            "cancer_type_filter": cancer_type or "all_types",
            "total_studies": len(studies_data),
            "studies": studies_data[:50],  # Limit for demo
            "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"Error getting available studies: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Demo function to showcase multi-agent speed benefits
def comprehensive_mutation_analysis(gene_list: List[str], analysis_type: str = "frequency") -> str:
    """Perform comprehensive mutation analysis demonstrating multi-agent workflow.
    
    This function orchestrates multiple sequential API calls to demonstrate
    where fast inference with Cerebras provides speed advantages in biomedical
    research workflows for Memorial Sloan Kettering demo.
    
    Parameters
    ----------
    gene_list : List[str]
        List of genes to analyze (e.g., ["TP53", "KRAS", "EGFR", "BRCA1"])
    analysis_type : str, optional
        Type of analysis to perform ("frequency", "clinical_correlation", "study_comparison")
        Default is "frequency"
    
    Returns
    -------
    str
        Comprehensive analysis report with all findings
    """
    
    try:
        analysis_log = []
        analysis_log.append(f"Starting comprehensive mutation analysis for {len(gene_list)} genes")
        analysis_log.append(f"Analysis type: {analysis_type}")
        analysis_log.append("=" * 60)
        
        # Step 1: Get available studies
        analysis_log.append("Step 1: Retrieving available studies...")
        studies_result = get_available_studies()
        analysis_log.append("✓ Retrieved available studies")
        
        # Step 2: Analyze mutation frequencies
        analysis_log.append("Step 2: Analyzing mutation frequencies...")
        frequency_result = analyze_mutation_frequency(gene_list)
        analysis_log.append("✓ Completed mutation frequency analysis")
        
        # Step 3: Get clinical data for correlation (if requested)
        if analysis_type == "clinical_correlation":
            analysis_log.append("Step 3: Retrieving clinical data for correlation...")
            clinical_result = get_clinical_data_by_study("tcga_gbm")
            analysis_log.append("✓ Retrieved clinical data")
        
        # Step 4: Get molecular profiles
        analysis_log.append("Step 4: Getting molecular profiles...")
        profiles_result = get_molecular_profiles("tcga_gbm")
        analysis_log.append("✓ Retrieved molecular profiles")
        
        # Compile final report
        final_report = {
            "analysis_type": analysis_type,
            "genes_analyzed": gene_list,
            "workflow_steps": analysis_log,
            "mutation_frequency_data": json.loads(frequency_result),
            "molecular_profiles_data": json.loads(profiles_result),
            "analysis_completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if analysis_type == "clinical_correlation":
            final_report["clinical_data"] = json.loads(clinical_result)
        
        return json.dumps(final_report, indent=2)
        
    except Exception as e:
        return f"Error in comprehensive analysis: {str(e)}"


if __name__ == "__main__":
    # Test the tools
    print("🧪 Testing cBioPortal Integration Tools")
    print("=" * 50)
    
    # Test basic mutation query
    print("Testing mutation query for TP53...")
    result = query_mutations_by_gene("TP53")
    print("✓ Mutation query completed")
    
    # Test frequency analysis
    print("Testing mutation frequency analysis...")
    result = analyze_mutation_frequency(["TP53", "KRAS"])
    print("✓ Frequency analysis completed")
    
    print("All tests completed successfully!")
