import pandas as pd
import pickle
import math
import numpy as np
import random
import time
import os
import argparse


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MMF: Maximizing Marginal Fairness Re-ranking")

    parser.add_argument('--relevance_path', type=str, required=True,
                        help='Path to the relevance score CSV file.')
    parser.add_argument('--competition_path', type=str, required=True,
                        help='Path to the competition relationship matrix CSV file.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of items in the top-K ranking list.')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Probability of selecting relevance-based strategy (0 to 1).')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed for reproducibility.')
    parser.add_argument('--output_dir', type=str, default='./res',
                        help='Directory to save the output results.')

    return parser.parse_args()


def compute_average(x):
    """Compute the average of a list."""
    return sum(x) / len(x)


def compute_dcg(relevance_scores, k):
    """Compute DCG@k for given relevance scores."""
    dcg = 0
    for i in range(k):
        dcg += relevance_scores[i] / math.log(i + 2, 2)
    return dcg


def compute_exposure_value(position):
    """Compute the exposure value for a given position."""
    return 1 / math.log(position + 2, 2)


def compute_exposure_log(top_k):
    """Compute the exposure values for all positions in the ranking list."""
    exp_log = []
    for i in range(top_k):
        exp_log.append(compute_exposure_value(i))
    return exp_log


def compute_unfairness(exposure_merit_ratio, competition_matrix, regularization_coef):
    """Compute the unfairness metric based on exposure-merit disparity."""
    item_num = len(exposure_merit_ratio)
    disparity_matrix = np.zeros((item_num, item_num))
    for i in range(item_num):
        disparity_matrix[i] = exposure_merit_ratio[i] - exposure_merit_ratio
    unfairness = np.sum(disparity_matrix * disparity_matrix * competition_matrix)
    return unfairness / regularization_coef


def generate_ranking_mmf(relevance, exposure_merit_ratio, item_num, top_k, alpha):
    """
    Generate top-k ranking using MMF probabilistic sampling strategy.
    With probability alpha, select from relevance-sorted list;
    with probability (1-alpha), select from fairness-sorted list (lowest exposure-merit ratio).
    """
    # Sort by relevance (descending)
    relevance_sorted = sorted(zip(relevance, range(item_num)),
                              key=lambda x: x[0], reverse=True)

    # Sort by exposure-merit ratio (ascending, i.e., most under-exposed first)
    fairness_sorted = sorted(zip(exposure_merit_ratio, range(item_num)),
                             key=lambda x: x[0], reverse=False)

    ranking_list = []
    for _ in range(top_k):
        if random.random() < alpha:
            # Select from relevance-sorted list
            for j in range(item_num):
                if relevance_sorted[j][1] not in ranking_list:
                    ranking_list.append(relevance_sorted[j][1])
                    break
        else:
            # Select from fairness-sorted list
            for j in range(item_num):
                if fairness_sorted[j][1] not in ranking_list:
                    ranking_list.append(fairness_sorted[j][1])
                    break

    assert len(ranking_list) == top_k
    return ranking_list


def save_results(output_dir, alpha, top_k, time_elapsed, avg_ndcg, final_unfairness,
                 unfairness_list, ndcg_list, exposure_merit_ratio,
                 cumulative_exposure, ranking_dict):
    """Save all experimental results to files."""
    result_dir = os.path.join(output_dir, str(alpha))
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, f"summary_{alpha}_{top_k}.txt"), 'w') as f:
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Top-K: {top_k}\n")
        f.write(f"Time: {time_elapsed:.2f}s\n")
        f.write(f"Cumulative NDCG: {avg_ndcg:.4f}\n")
        f.write(f"Final Unfairness: {final_unfairness:.4f}\n")

    with open(os.path.join(result_dir, f"unfairness_list_{alpha}_{top_k}.list"), "wb") as f:
        pickle.dump(unfairness_list, f)

    with open(os.path.join(result_dir, f"ndcg_list_{alpha}_{top_k}.list"), "wb") as f:
        pickle.dump(ndcg_list, f)

    with open(os.path.join(result_dir, f"exposure_merit_ratio_{alpha}_{top_k}.list"), "wb") as f:
        pickle.dump(exposure_merit_ratio, f)

    with open(os.path.join(result_dir, f"cumulative_exposure_{alpha}_{top_k}.list"), "wb") as f:
        pickle.dump(cumulative_exposure, f)

    with open(os.path.join(result_dir, f"ranking_list_{alpha}_{top_k}.list"), "wb") as f:
        pickle.dump(ranking_dict, f)


def main():
    # ==================== Parse Arguments ====================
    args = parse_arguments()

    relevance_path = args.relevance_path
    competition_path = args.competition_path
    top_k = args.top_k
    alpha = args.alpha
    seed = args.seed
    output_dir = args.output_dir

    # Set random seed for reproducibility
    random.seed(seed)

    # ==================== Data Loading ====================
    print(f"Loading relevance data from: {relevance_path}")
    relevance_data = pd.read_csv(relevance_path, header=None)

    print(f"Loading competition matrix from: {competition_path}")
    competition_df = pd.read_csv(competition_path, header=None)
    competition_matrix = competition_df.to_numpy()
    competition_matrix = (competition_matrix + competition_matrix.T) / 2
    regularization_coef = np.sum(competition_matrix)

    user_num, item_num = relevance_data.shape

    print(f"Dataset: {user_num} users, {item_num} items")
    print(f"Parameters: top_k={top_k}, alpha={alpha}, seed={seed}")
    print(f"Output directory: {output_dir}")

    # ==================== Initialization ====================
    cumulative_exposure = np.zeros(item_num)
    merit_sum = np.ones(item_num)
    exposure_merit_ratio = np.zeros(item_num)
    exp_log = compute_exposure_log(top_k)

    ndcg_list = []
    unfairness_list = []
    ranking_dict = dict()

    time_start = time.time()

    # ==================== Main Loop ====================
    for user_idx in range(user_num):
        # Get relevance scores for current user
        relevance = relevance_data.iloc[user_idx, :]

        # Compute ideal DCG
        sorted_relevance = sorted(relevance, reverse=True)
        ideal_dcg = compute_dcg(sorted_relevance, top_k)

        # Generate ranking using MMF strategy
        if user_idx == 0:
            # First iteration: use pure relevance ranking
            relevance_sorted = sorted(zip(relevance, range(item_num)),
                                      key=lambda x: x[0], reverse=True)
            ranking_list = [relevance_sorted[i][1] for i in range(top_k)]
        else:
            ranking_list = generate_ranking_mmf(
                relevance, exposure_merit_ratio, item_num, top_k, alpha
            )

        ranking_dict[user_idx] = ranking_list

        # Compute DCG for the generated ranking and update cumulative exposure
        dcg = 0
        for i in range(top_k):
            item_index = ranking_list[i]
            cumulative_exposure[item_index] += exp_log[i]
            dcg += relevance[item_index] / math.log(i + 2, 2)

        # Compute NDCG
        ndcg = dcg / ideal_dcg
        ndcg_list.append(ndcg)

        # Update merit and exposure-merit ratio
        merit_sum += relevance
        merit = merit_sum / (user_idx + 1)
        exposure_merit_ratio = cumulative_exposure / merit

        # Compute unfairness
        unfairness = compute_unfairness(exposure_merit_ratio, competition_matrix, regularization_coef)
        unfairness_list.append(unfairness)

        # Print progress
        time_elapsed = time.time() - time_start
        if (user_idx + 1) % 100 == 0 or user_idx == 0:
            print(f"Iteration {user_idx}/{user_num}, "
                  f"Time: {time_elapsed:.2f}s, "
                  f"NDCG: {ndcg:.4f}, "
                  f"Cumulative NDCG: {compute_average(ndcg_list):.4f}, "
                  f"Unfairness: {unfairness:.4f}")

    # ==================== Save Results ====================
    time_elapsed = time.time() - time_start
    print(f"\nExperiment completed in {time_elapsed:.2f}s")
    print(f"Final Cumulative NDCG: {compute_average(ndcg_list):.4f}")
    print(f"Final Unfairness: {unfairness_list[-1]:.4f}")

    save_results(
        output_dir, alpha, top_k, time_elapsed,
        compute_average(ndcg_list), unfairness_list[-1],
        unfairness_list, ndcg_list, exposure_merit_ratio,
        cumulative_exposure, ranking_dict
    )
    print(f"Results saved to: {os.path.join(output_dir, str(alpha))}")


if __name__ == "__main__":
    main()
