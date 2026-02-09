"""
A/B Test Power Calculator
=========================

Calculate sample sizes, power, and minimum detectable effects
for A/B tests and causal inference experiments.

Author: Causal Impact Analysis Project
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class PowerAnalysisResult:
    """Results from power analysis."""
    sample_size_per_group: int
    total_sample_size: int
    power: float
    alpha: float
    effect_size: float
    mde_absolute: Optional[float]
    mde_relative: Optional[float]
    baseline_rate: Optional[float]


class PowerCalculator:
    """
    A/B Test Power and Sample Size Calculator.
    
    Supports:
    - Continuous outcomes (t-test)
    - Binary outcomes (proportions)
    - Minimum Detectable Effect (MDE)
    
    Example:
        >>> calc = PowerCalculator()
        >>> result = calc.sample_size_continuous(
        ...     effect_size=0.1,
        ...     baseline_std=10,
        ...     power=0.8,
        ...     alpha=0.05
        ... )
        >>> print(f"Need {result.total_sample_size} users")
    """
    
    def __init__(self, two_sided: bool = True):
        """
        Initialize power calculator.
        
        Args:
            two_sided: Use two-sided test (default True)
        """
        self.two_sided = two_sided
    
    def _get_z_scores(self, alpha: float, power: float) -> Tuple[float, float]:
        """Get z-scores for alpha and power."""
        if self.two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
        
        return z_alpha, z_beta
    
    def sample_size_continuous(
        self,
        effect_size: float,
        baseline_std: float,
        power: float = 0.8,
        alpha: float = 0.05,
        ratio: float = 1.0
    ) -> PowerAnalysisResult:
        """
        Calculate sample size for continuous outcome.
        
        Args:
            effect_size: Expected difference in means
            baseline_std: Standard deviation of the metric
            power: Desired statistical power (default 0.8)
            alpha: Significance level (default 0.05)
            ratio: Ratio of treatment to control (default 1.0)
        
        Returns:
            PowerAnalysisResult with sample sizes
        """
        z_alpha, z_beta = self._get_z_scores(alpha, power)
        
        # Cohen's d
        d = effect_size / baseline_std
        
        # Sample size formula
        n_per_group = 2 * ((z_alpha + z_beta) / d) ** 2
        
        # Adjust for unequal allocation
        if ratio != 1.0:
            n_per_group = n_per_group * (1 + ratio) ** 2 / (4 * ratio)
        
        n_per_group = int(np.ceil(n_per_group))
        
        return PowerAnalysisResult(
            sample_size_per_group=n_per_group,
            total_sample_size=int(n_per_group * (1 + ratio)),
            power=power,
            alpha=alpha,
            effect_size=d,
            mde_absolute=effect_size,
            mde_relative=effect_size / baseline_std if baseline_std else None,
            baseline_rate=None
        )
    
    def sample_size_proportion(
        self,
        baseline_rate: float,
        mde_relative: float,
        power: float = 0.8,
        alpha: float = 0.05,
        ratio: float = 1.0
    ) -> PowerAnalysisResult:
        """
        Calculate sample size for conversion rate test.
        
        Args:
            baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
            mde_relative: Minimum detectable relative lift (e.g., 0.1 for 10%)
            power: Desired power (default 0.8)
            alpha: Significance level (default 0.05)
            ratio: Ratio of treatment to control
        
        Returns:
            PowerAnalysisResult with sample sizes
        """
        z_alpha, z_beta = self._get_z_scores(alpha, power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde_relative)
        
        # Pooled proportion
        p_bar = (p1 + p2) / 2
        
        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Sample size
        n_per_group = 2 * ((z_alpha + z_beta) / h) ** 2
        
        # Adjust for unequal allocation
        if ratio != 1.0:
            n_per_group = n_per_group * (1 + ratio) ** 2 / (4 * ratio)
        
        n_per_group = int(np.ceil(n_per_group))
        
        return PowerAnalysisResult(
            sample_size_per_group=n_per_group,
            total_sample_size=int(n_per_group * (1 + ratio)),
            power=power,
            alpha=alpha,
            effect_size=h,
            mde_absolute=p2 - p1,
            mde_relative=mde_relative,
            baseline_rate=baseline_rate
        )
    
    def power_continuous(
        self,
        sample_size_per_group: int,
        effect_size: float,
        baseline_std: float,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate power for given sample size (continuous).
        
        Args:
            sample_size_per_group: Sample size per group
            effect_size: Expected difference in means
            baseline_std: Standard deviation
            alpha: Significance level
        
        Returns:
            Statistical power
        """
        d = effect_size / baseline_std
        z_alpha = stats.norm.ppf(1 - alpha / 2) if self.two_sided else stats.norm.ppf(1 - alpha)
        
        se = np.sqrt(2 / sample_size_per_group)
        z_beta = d / se - z_alpha
        
        power = stats.norm.cdf(z_beta)
        return min(power, 0.9999)
    
    def power_proportion(
        self,
        sample_size_per_group: int,
        baseline_rate: float,
        lift: float,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate power for given sample size (proportions).
        
        Args:
            sample_size_per_group: Sample size per group
            baseline_rate: Baseline conversion rate
            lift: Relative lift to detect
            alpha: Significance level
        
        Returns:
            Statistical power
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + lift)
        
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        z_alpha = stats.norm.ppf(1 - alpha / 2) if self.two_sided else stats.norm.ppf(1 - alpha)
        
        se = np.sqrt(2 / sample_size_per_group)
        z_beta = h / se - z_alpha
        
        power = stats.norm.cdf(z_beta)
        return min(power, 0.9999)
    
    def mde_continuous(
        self,
        sample_size_per_group: int,
        baseline_std: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate Minimum Detectable Effect for given sample.
        
        Args:
            sample_size_per_group: Sample size per group
            baseline_std: Standard deviation
            power: Desired power
            alpha: Significance level
        
        Returns:
            Minimum detectable absolute effect
        """
        z_alpha, z_beta = self._get_z_scores(alpha, power)
        
        mde = (z_alpha + z_beta) * baseline_std * np.sqrt(2 / sample_size_per_group)
        return mde
    
    def mde_proportion(
        self,
        sample_size_per_group: int,
        baseline_rate: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Calculate Minimum Detectable Effect for proportions.
        
        Returns:
            Tuple of (absolute_mde, relative_mde)
        """
        z_alpha, z_beta = self._get_z_scores(alpha, power)
        
        # Approximate MDE
        se_baseline = np.sqrt(baseline_rate * (1 - baseline_rate) / sample_size_per_group)
        mde_absolute = (z_alpha + z_beta) * se_baseline * np.sqrt(2)
        mde_relative = mde_absolute / baseline_rate
        
        return mde_absolute, mde_relative
    
    def sensitivity_analysis(
        self,
        baseline_rate: float = 0.05,
        power: float = 0.8,
        alpha: float = 0.05,
        sample_sizes: Optional[List[int]] = None,
        lifts: Optional[List[float]] = None
    ) -> Dict[str, List]:
        """
        Run sensitivity analysis across sample sizes and lifts.
        
        Returns grid of sample sizes, lifts, and achievable power.
        """
        if sample_sizes is None:
            sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        
        if lifts is None:
            lifts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        results = {
            'sample_size': [],
            'lift': [],
            'power': [],
            'detectable': []
        }
        
        for n in sample_sizes:
            for lift in lifts:
                power_achieved = self.power_proportion(n, baseline_rate, lift, alpha)
                results['sample_size'].append(n)
                results['lift'].append(lift)
                results['power'].append(power_achieved)
                results['detectable'].append(power_achieved >= power)
        
        return results


def main():
    """Demo power calculator."""
    print("=" * 60)
    print("A/B TEST POWER CALCULATOR DEMO")
    print("=" * 60)
    
    calc = PowerCalculator()
    
    # Example 1: Conversion rate test
    print("\n1. Conversion Rate Test")
    print("-" * 40)
    print("Baseline: 5%, Want to detect: 10% relative lift")
    
    result = calc.sample_size_proportion(
        baseline_rate=0.05,
        mde_relative=0.10,
        power=0.8,
        alpha=0.05
    )
    
    print(f"Sample size per group: {result.sample_size_per_group:,}")
    print(f"Total sample needed: {result.total_sample_size:,}")
    
    # Example 2: Revenue test
    print("\n2. Revenue Test (Continuous)")
    print("-" * 40)
    print("Detect $5 increase, baseline std=$50")
    
    result = calc.sample_size_continuous(
        effect_size=5,
        baseline_std=50,
        power=0.8,
        alpha=0.05
    )
    
    print(f"Sample size per group: {result.sample_size_per_group:,}")
    print(f"Total sample needed: {result.total_sample_size:,}")
    
    # Example 3: Power for given sample
    print("\n3. Power Calculation")
    print("-" * 40)
    print("10,000 users per group, 5% baseline, 10% lift")
    
    power = calc.power_proportion(
        sample_size_per_group=10000,
        baseline_rate=0.05,
        lift=0.10
    )
    
    print(f"Achievable power: {power:.1%}")
    
    # Example 4: MDE calculation
    print("\n4. Minimum Detectable Effect")
    print("-" * 40)
    print("50,000 users per group, 5% baseline")
    
    mde_abs, mde_rel = calc.mde_proportion(
        sample_size_per_group=50000,
        baseline_rate=0.05,
        power=0.8
    )
    
    print(f"MDE (absolute): {mde_abs:.4f}")
    print(f"MDE (relative): {mde_rel:.1%}")
    
    # Example 5: Sensitivity table
    print("\n5. Sensitivity Analysis")
    print("-" * 40)
    
    results = calc.sensitivity_analysis(
        baseline_rate=0.05,
        sample_sizes=[5000, 10000, 25000],
        lifts=[0.05, 0.10, 0.15]
    )
    
    print("Sample Size | Lift | Power | Detectable")
    print("-" * 45)
    for i in range(len(results['sample_size'])):
        check = "✓" if results['detectable'][i] else ""
        print(f"{results['sample_size'][i]:>10,} | {results['lift'][i]:>4.0%} | {results['power'][i]:>5.1%} | {check}")
    
    print("\n✓ Power calculator demo completed!")


if __name__ == '__main__':
    main()
