"""
Exploratory Data Analysis Module
Creates comprehensive visualizations and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure directories exist
os.makedirs('reports/figures', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def perform_eda(data_path='data/dataset.csv'):
    """
    Perform exploratory data analysis and create visualizations.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"   Dataset shape: {data.shape}")
    
    # Basic statistics
    print("\n2. Dataset Statistics:")
    print(data.describe())
    
    # Correlation analysis
    print("\n3. Creating correlation heatmap...")
    numerical_cols = ['Age', 'Openness', 'Conscientiousness', 'Extraversion', 
                      'Agreeableness', 'Neuroticism', 'Optimism', 'Stress_Level', 
                      'Anxiety_Level', 'Emotional_Resilience', 'Placebo_Response']
    
    correlation_matrix = data[numerical_cols].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Correlation Heatmap of Psychological Factors', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/correlation_heatmap.png")
    plt.close()
    
    # Distribution plots
    print("\n4. Creating distribution plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    traits = ['Optimism', 'Stress_Level', 'Anxiety_Level', 'Emotional_Resilience',
              'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    palette = sns.color_palette("husl", len(traits))
    
    for i, trait in enumerate(traits):
        axes[i].hist(
            data[trait],
            bins=30,
            edgecolor='black',
            alpha=0.7,
            color=palette[i],
        )
        axes[i].set_title(f'Distribution of {trait}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(trait)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Psychological Traits', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('reports/figures/trait_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/trait_distributions.png")
    plt.close()
    
    # Placebo response analysis
    print("\n5. Analyzing placebo response...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Response distribution
    response_counts = data['Placebo_Response'].value_counts()
    axes[0, 0].bar(['Not Improved (0)', 'Improved (1)'], response_counts.values, 
                    color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2)
    axes[0, 0].set_title('Placebo Response Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Response by gender
    response_by_gender = pd.crosstab(data['Gender'], data['Placebo_Response'])
    response_by_gender.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'], edgecolor='black')
    axes[0, 1].set_title('Placebo Response by Gender', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Gender')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend(['Not Improved', 'Improved'])
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Optimism vs Placebo Response
    sns.boxplot(data=data, x='Placebo_Response', y='Optimism', ax=axes[1, 0], 
                palette=['#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Optimism Score by Placebo Response', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Placebo Response')
    axes[1, 0].set_ylabel('Optimism Score')
    axes[1, 0].set_xticklabels(['Not Improved', 'Improved'])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Stress vs Placebo Response
    sns.boxplot(data=data, x='Placebo_Response', y='Stress_Level', ax=axes[1, 1], 
                palette=['#e74c3c', '#2ecc71'])
    axes[1, 1].set_title('Stress Level by Placebo Response', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Placebo Response')
    axes[1, 1].set_ylabel('Stress Level')
    axes[1, 1].set_xticklabels(['Not Improved', 'Improved'])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Placebo Response Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('reports/figures/placebo_response_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/placebo_response_analysis.png")
    plt.close()
    
    # Feature importance correlation with target
    print("\n6. Feature correlation with Placebo Response:")
    correlations_with_target = correlation_matrix['Placebo_Response'].sort_values(ascending=False)
    print(correlations_with_target)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    correlations_with_target.drop('Placebo_Response').plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title('Feature Correlation with Placebo Response', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance_correlation.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/feature_importance_correlation.png")
    plt.close()
    
    # Age distribution by response
    print("\n7. Creating age analysis...")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Age', hue='Placebo_Response', bins=30, 
                 palette=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    plt.title('Age Distribution by Placebo Response', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(['Not Improved', 'Improved'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('reports/figures/age_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/age_distribution.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE! All visualizations saved.")
    print("=" * 60)
