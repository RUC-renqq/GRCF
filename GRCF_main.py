import pandas as pd
import pickle
import math
import numpy as np
import time
import os


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


def compute_total_exposure(top_k):
    """Compute the total exposure provided by a single ranking list."""
    total = 0
    for i in range(top_k):
        total += compute_exposure_value(i)
    return total


def compute_unfairness(exposure_merit_ratio, competition_matrix, regularization_coef):
    """Compute the unfairness metric based on exposure-merit disparity."""
    item_num = len(exposure_merit_ratio)
    disparity_matrix = np.zeros((item_num, item_num))
    for i in range(item_num):
        disparity_matrix[i] = exposure_merit_ratio[i] - exposure_merit_ratio
    unfairness = np.sum(disparity_matrix * disparity_matrix * competition_matrix)
    return unfairness / regularization_coef


def compute_gradient(item_num, relevance, competition_matrix, exposure_merit_ratio,
                     merit, alpha, regularization_coef):
    """Compute the gradient for each item."""
    gradient = np.zeros((item_num, 1))
    fairness_coef = alpha / regularization_coef

    for i in range(item_num):
        utility_term = (1 - alpha) * relevance[i]
        degree_vector = competition_matrix[i]
        exposure_merit_diff = exposure_merit_ratio[i] - exposure_merit_ratio
        fairness_term = np.sum(degree_vector * exposure_merit_diff) * 2 / merit[i] * fairness_coef * 2
        gradient[i] = utility_term - fairness_term

    return gradient


def generate_ranking(gradient, relevance, item_num, top_k, cumulative_exposure):
    """Generate top-k ranking based on gradient and update cumulative exposure."""
    # Sort by gradient (primary) and relevance (secondary) in descending order
    candidates = zip(gradient, relevance, range(item_num))
    sorted_candidates = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)

    ranking_list = []
    dcg = 0
    for i in range(top_k):
        item_index = sorted_candidates[i][2]
        ranking_list.append(item_index)
        dcg += relevance[item_index] / math.log(i + 2, 2)
        cumulative_exposure[item_index] += compute_exposure_value(i)

    return ranking_list, dcg, cumulative_exposure


def save_results(alpha, top_k, time_elapsed, avg_ndcg, final_unfairness,
                 unfairness_list, ndcg_list, exposure_merit_ratio,
                 cumulative_exposure, ranking_dict):
    """Save all experimental results to files."""
    output_dir = f"./res/{alpha}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/res_{alpha}_{top_k}.txt", 'w') as f:
        f.write(f"Time: {time_elapsed:.2f}s, "
                f"Cumulative NDCG: {avg_ndcg:.4f}, "
                f"Unfairness: {final_unfairness:.4f}")

    with open(f"{output_dir}/unfairness_list_{alpha}_{top_k}.list", "wb") as f:
        pickle.dump(unfairness_list, f)

    with open(f"{output_dir}/ndcg_list_{alpha}_{top_k}.list", "wb") as f:
        pickle.dump(ndcg_list, f)

    with open(f"{output_dir}/exposure_merit_ratio_{alpha}_{top_k}.list", "wb") as f:
        pickle.dump(exposure_merit_ratio, f)

    with open(f"{output_dir}/cumulative_exposure_{alpha}_{top_k}.list", "wb") as f:
        pickle.dump(cumulative_exposure, f)

    with open(f"{output_dir}/ranking_list_{alpha}_{top_k}.list", "wb") as f:
        pickle.dump(ranking_dict, f)


def main():
    # ==================== Configuration ====================
    relevance_path = '../../data/mt_4272_2395_pctr_2024_TOIS.csv'
    competition_path = '../../data/mt_j_sim_sym_01_TOIS_0.99.csv'
    top_k = 5
    alpha = 0.1

    # ==================== Data Loading ====================
    relevance_data = pd.read_csv(relevance_path, header=None)
    competition_df = pd.read_csv(competition_path, header=None)
    competition_matrix = competition_df.to_numpy()
    competition_matrix = (competition_matrix + competition_matrix.T) / 2

    user_num, item_num = relevance_data.shape
    total_exposure = compute_total_exposure(top_k)
    regularization_coef = np.sum(competition_matrix)

    # ==================== Initialization ====================
    cumulative_exposure = np.zeros(item_num)
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

        # Compute merit (average relevance up to current time step)
        merit = (relevance_data.iloc[0:user_idx + 1, :].sum(axis=0) + np.ones(item_num)) / (user_idx + 1)

        # Compute exposure-merit ratio
        exposure_merit_ratio = cumulative_exposure / merit

        # Compute gradient
        gradient = compute_gradient(
            item_num, relevance, competition_matrix,
            exposure_merit_ratio, merit, alpha, regularization_coef
        )

        # Generate ranking and update cumulative exposure
        ranking_list, dcg, cumulative_exposure = generate_ranking(
            gradient, relevance, item_num, top_k, cumulative_exposure
        )
        ranking_dict[user_idx] = ranking_list

        # Compute NDCG
        ndcg = dcg / ideal_dcg
        ndcg_list.append(ndcg)

        # Update exposure-merit ratio and compute unfairness
        exposure_merit_ratio = cumulative_exposure / merit
        unfairness = compute_unfairness(exposure_merit_ratio, competition_matrix, regularization_coef)
        unfairness_list.append(unfairness)

        # Print progress
        time_elapsed = time.time() - time_start
        print(f"Iteration {user_idx}, "
              f"Time: {time_elapsed:.2f}s, "
              f"NDCG: {ndcg:.4f}, "
              f"Cumulative NDCG: {compute_average(ndcg_list):.4f}, "
              f"Unfairness: {unfairness:.4f}")

    # ==================== Save Results ====================
    time_elapsed = time.time() - time_start
    save_results(
        alpha, top_k, time_elapsed,
        compute_average(ndcg_list), unfairness_list[-1],
        unfairness_list, ndcg_list, exposure_merit_ratio,
        cumulative_exposure, ranking_dict
    )


if __name__ == "__main__":
    main()
