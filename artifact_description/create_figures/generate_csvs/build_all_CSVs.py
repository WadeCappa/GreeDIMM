import strong_scaling, imm_comparison, quality_drop_off, truncated_streaming, scaling_breakdown
import sys, math

def main():
    strong_scalilng = strong_scaling.StrongScaling()
    strong_scalilng.build_strong_scaling("../../results/strong_scaling/", sys.argv[1] + "/strong_scaling.csv")

    comparison = imm_comparison.IMMComparison()
    comparison.build_comparison("../../results/imm/", "../../results/strong_scaling/", sys.argv[1])

    quality = quality_drop_off.QualityDropoff()
    quality.build_quality_csv("../../results/strong_scaling/", sys.argv[1] + "/streaming_quality_dropoff.csv")

    truncated = truncated_streaming.TruncatedStreaming()
    truncated.build_truncated_csv("../../results/truncated/orkut_small/", sys.argv[1] + "/truncated_results.csv")

    breakdown = scaling_breakdown.ScalingBreakdown()
    breakdown.build_breakdown_csv("../../results/strong_scaling/wikipedia", [int(math.pow(2, x)) for x in range(6, 10 + 1)], sys.argv[1] + "/wikipedia_breakdown.csv")

if __name__ == '__main__':
    main()